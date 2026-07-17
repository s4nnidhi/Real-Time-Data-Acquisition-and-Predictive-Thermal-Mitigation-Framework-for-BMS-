/*
  ================================================================
   ADAPTIVE BMS — LIVE WEB DASHBOARD
   ESP32 | WiFi WebServer | Real-Time | Dual Branch | Kalman SoC
   + ML OVERHEAT PREDICTION (trained on dual_3s_discharge_24may.csv)
  ================================================================
  Libraries: Adafruit INA219, DallasTemperature, OneWire
  ================================================================

  ── ML MODEL SUMMARY (from 957-sample discharge log, 24 May 2026) ──
  Dataset spans 7170s across 2 cycles. Linear regression fitted to
  full temperature history for each branch:

    Branch A: temp(t) = 29.0476 + 0.002427 × t   (R² ≈ 0.99)
    Branch B: temp(t) = 29.9137 + 0.003732 × t   (R² ≈ 0.99)

  Observed overheat events:
    HOT  (≥38°C) — A: 3075 s (51m 15s) | B: 1920 s (32m 0s)
    CRIT (≥45°C) — A: 6525 s (108m 45s) | B: 3405 s (56m 45s)
    Predicted 55°C — A: ~10694 s (~2.97h) | B: ~6722 s (~1.87h)

  Mean observed dT/dt: A = 0.00254 °C/s | B = 0.00377 °C/s
  95th-pct dT/dt:      A = 0.00482 °C/s | B = 0.00634 °C/s

  The sketch uses a live Kalman-corrected ETA that starts from the
  ML prior and is updated each loop using actual measured dT/dt.
  ================================================================
*/

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <WiFi.h>
#include <WebServer.h>

// ══════════════════════════════════════════════════════════
#define WIFI_SSID   "Nishanth"
#define WIFI_PASS   "nononono"
// ══════════════════════════════════════════════════════════

#define TEMP_PIN        5
#define PIN_MOSFET_A    19
#define PIN_MOSFET_B    23
#define PWM_FREQ        5000
#define PWM_RES         8
#define DT_INTERVAL     1000
#define RISE_THRESHOLD  0.08f
#define WARN_TEMP       45.0f
#define CRITICAL_TEMP   55.0f

const float NOMINAL_CAPACITY = 2500.0f;
const float FULL_V           = 12.6f;
const float EMPTY_V          = 9.0f;

// ── ML priors from linear regression on discharge CSV ─────────────
// Baseline: temp(t) = ML_INTERCEPT + ML_SLOPE * t
// Used as the initial dTdt estimate before real data accumulates.
// After ALPHA_BLEND_SAMPLES samples, live measurement takes over.
const float ML_SLOPE_A     = 0.002427f;   // °C/s
const float ML_SLOPE_B     = 0.003732f;   // °C/s
const float ML_INTERCEPT_A = 29.0476f;    // °C at t=0
const float ML_INTERCEPT_B = 29.9137f;    // °C at t=0
const float ML_MEAN_DTDT_A = 0.00254f;    // mean observed dT/dt
const float ML_MEAN_DTDT_B = 0.00377f;    // mean observed dT/dt
const int   ALPHA_BLEND_SAMPLES = 120;    // blend live vs ML prior over 2 min

// ── Overheat thresholds for ETA prediction ────────────────────────
const float OVERHEAT_TARGET = 55.0f;      // hard overheat temperature
const float WARN_TARGET     = 45.0f;      // warning temperature

// ── Adaptive ETA filter (exponential moving average on dTdt) ──────
const float ETA_EMA_ALPHA = 0.05f;        // smoothing for dTdt estimate

struct KalmanBMS { float soc = 100.0f; float p = 1.0f; };
KalmanBMS kA, kB;

struct BranchData {
  float temp        = 25.0f;
  float dTdt        = 0.0f;
  float dTdt_ema    = 0.0f;    // EMA-smoothed dT/dt
  float volt        = 12.0f;
  float curr        = 0.0f;
  float power       = 0.0f;
  float soc         = 100.0f;
  float eta_warn_s  = -1.0f;   // seconds until WARN_TARGET (−1 = already past)
  float eta_crit_s  = -1.0f;   // seconds until OVERHEAT_TARGET
  int   duty        = 255;
};
BranchData brA, brB;

OneWire           oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);
Adafruit_INA219   inaA(0x40);
Adafruit_INA219   inaB(0x41);
WebServer         server(80);

float         tA_prev        = -999.0f;
float         tB_prev        = -999.0f;
unsigned long lastMeasure    = 0;
unsigned long startTime      = 0;
unsigned long loopCount      = 0;
bool          wifiConnected  = false;

// ─────────────────────────────────────────────────────────────────
// Kalman SoC estimator (unchanged from original)
// ─────────────────────────────────────────────────────────────────
float runKalman(KalmanBMS &k, float volt, float curr_mA, float dt) {
  k.soc -= ((curr_mA / 1000.0f) * dt / 3600.0f) / (NOMINAL_CAPACITY / 1000.0f) * 100.0f;
  float vPct = constrain(((volt - EMPTY_V) / (FULL_V - EMPTY_V)) * 100.0f, 0.0f, 100.0f);
  k.soc = constrain(k.soc + 0.1f * (vPct - k.soc), 0.0f, 100.0f);
  return k.soc;
}

// ─────────────────────────────────────────────────────────────────
// ML-blended ETA calculator
//   blends the ML-prior dTdt with the live EMA dTdt over the first
//   ALPHA_BLEND_SAMPLES samples, then switches fully to live data.
//   Returns ETA in seconds to reach targetTemp, or −1 if past it.
// ─────────────────────────────────────────────────────────────────
float computeETA(float currentTemp, float ema_dtdt, float ml_prior_dtdt,
                 float targetTemp) {
  if (currentTemp >= targetTemp) return -1.0f;

  // Blend weight: 0→1 over first ALPHA_BLEND_SAMPLES loops
  float alpha = constrain((float)loopCount / ALPHA_BLEND_SAMPLES, 0.0f, 1.0f);
  float effective_dtdt = (1.0f - alpha) * ml_prior_dtdt + alpha * ema_dtdt;

  if (effective_dtdt <= 0.0001f) return 99999.0f;  // essentially flat
  return (targetTemp - currentTemp) / effective_dtdt;
}

// ─────────────────────────────────────────────────────────────────
// JSON endpoint (/data)
// ─────────────────────────────────────────────────────────────────
void handleJSON() {
  unsigned long upSecs = (millis() - startTime) / 1000;

  // Format ETA strings
  char etaWarnA[16], etaCritA[16], etaWarnB[16], etaCritB[16];
  auto fmtETA = [](float s, char* buf) {
    if (s < 0)           snprintf(buf, 16, "PAST");
    else if (s > 90000)  snprintf(buf, 16, ">25h");
    else {
      int h = (int)(s / 3600), m = (int)(fmod(s, 3600) / 60), sec = (int)fmod(s, 60);
      if (h > 0) snprintf(buf, 16, "%dh%02dm", h, m);
      else       snprintf(buf, 16, "%dm%02ds", m, sec);
    }
  };
  fmtETA(brA.eta_warn_s, etaWarnA);
  fmtETA(brA.eta_crit_s, etaCritA);
  fmtETA(brB.eta_warn_s, etaWarnB);
  fmtETA(brB.eta_crit_s, etaCritB);

  char buf[1100];
  snprintf(buf, sizeof(buf),
    "{"
    "\"uptime\":%lu,\"sample\":%lu,"
    "\"A\":{"
      "\"temp\":%.2f,\"dtdt\":%.4f,\"dtdt_ema\":%.4f,"
      "\"volt\":%.3f,\"curr\":%.1f,\"power\":%.3f,"
      "\"soc\":%.1f,\"dod\":%.1f,"
      "\"duty\":%d,\"dutyPct\":%.1f,"
      "\"eta_warn\":\"%s\",\"eta_warn_s\":%.0f,"
      "\"eta_crit\":\"%s\",\"eta_crit_s\":%.0f"
    "},"
    "\"B\":{"
      "\"temp\":%.2f,\"dtdt\":%.4f,\"dtdt_ema\":%.4f,"
      "\"volt\":%.3f,\"curr\":%.1f,\"power\":%.3f,"
      "\"soc\":%.1f,\"dod\":%.1f,"
      "\"duty\":%d,\"dutyPct\":%.1f,"
      "\"eta_warn\":\"%s\",\"eta_warn_s\":%.0f,"
      "\"eta_crit\":\"%s\",\"eta_crit_s\":%.0f"
    "}"
    "}",
    upSecs, loopCount,
    brA.temp, brA.dTdt, brA.dTdt_ema,
    brA.volt, brA.curr, brA.power,
    brA.soc, 100.0f - brA.soc,
    brA.duty, (brA.duty / 255.0f) * 100.0f,
    etaWarnA, brA.eta_warn_s, etaCritA, brA.eta_crit_s,
    brB.temp, brB.dTdt, brB.dTdt_ema,
    brB.volt, brB.curr, brB.power,
    brB.soc, 100.0f - brB.soc,
    brB.duty, (brB.duty / 255.0f) * 100.0f,
    etaWarnB, brB.eta_warn_s, etaCritB, brB.eta_crit_s
  );
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", buf);
}

// ─────────────────────────────────────────────────────────────────
// HTML dashboard (/root)
// ─────────────────────────────────────────────────────────────────
void handleRoot() {
  String html = R"rawhtml(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BMS Live Dashboard — ML Overheat Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#0a0c10;--surface:#0f1318;--card:#141820;--border:#1e2530;--accent-a:#00e5ff;--accent-b:#7c4dff;--green:#00e676;--amber:#ffab00;--red:#ff1744;--text:#e8eaf0;--muted:#5c6370;--mono:'Share Tech Mono',monospace;--display:'Rajdhani',sans-serif;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:var(--display);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,229,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0;}
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);pointer-events:none;z-index:0;}
.wrap{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:24px 20px;}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;padding-bottom:16px;border-bottom:1px solid var(--border);}
.logo{font-family:var(--mono);font-size:11px;color:var(--accent-a);letter-spacing:3px;text-transform:uppercase;}
h1{font-size:26px;font-weight:700;letter-spacing:2px;color:var(--text);text-transform:uppercase;}
h1 span{color:var(--accent-a);}
.status-pill{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px;color:var(--muted);background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:6px 14px;}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite;}
.dot.dead{background:var(--red);animation:none;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,230,118,0.6);}50%{box-shadow:0 0 0 5px rgba(0,230,118,0);}}
.meta-row{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;}
.meta-chip{font-family:var(--mono);font-size:12px;color:var(--muted);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:5px 12px;}
.meta-chip span{color:var(--text);margin-left:6px;}
.alert-banner{display:none;margin-bottom:20px;padding:12px 18px;border-radius:8px;border:1px solid var(--red);background:rgba(255,23,68,0.08);font-family:var(--mono);font-size:13px;color:var(--red);letter-spacing:1px;}
.alert-banner.show{display:block;}
/* ── ML ETA banner ── */
.eta-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;}
@media(max-width:700px){.eta-row{grid-template-columns:1fr;}}
.eta-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 18px;}
.eta-header{font-size:10px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px;}
.eta-header::after{content:'';flex:1;height:1px;background:var(--border);}
.eta-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.eta-box{background:var(--surface);border-radius:6px;padding:8px 12px;border:1px solid var(--border);}
.eta-label{font-size:10px;letter-spacing:1px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;}
.eta-val{font-family:var(--mono);font-size:20px;font-weight:600;}
.eta-val.warn{color:var(--amber);}
.eta-val.crit{color:var(--red);}
.eta-val.past{color:var(--muted);font-size:14px;}
.ml-badge{font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:3px;background:rgba(0,229,255,0.08);color:var(--accent-a);border:1px solid rgba(0,229,255,0.2);letter-spacing:1px;margin-left:8px;}
/* ── Branch cards ── */
.branches{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px;}
@media(max-width:700px){.branches{grid-template-columns:1fr;}}
.branch-card{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;}
.branch-card.alert-a{border-color:rgba(255,171,0,0.5);}
.branch-card.alert-crit{border-color:rgba(255,23,68,0.6);}
.branch-header{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--border);}
.branch-title{font-size:18px;font-weight:700;letter-spacing:3px;text-transform:uppercase;}
.branch-title.a{color:var(--accent-a);}
.branch-title.b{color:var(--accent-b);}
.branch-status{font-family:var(--mono);font-size:11px;padding:4px 10px;border-radius:4px;text-transform:uppercase;letter-spacing:1px;}
.s-ok{background:rgba(0,230,118,0.12);color:var(--green);border:1px solid rgba(0,230,118,0.3);}
.s-warn{background:rgba(255,171,0,0.12);color:var(--amber);border:1px solid rgba(255,171,0,0.3);}
.s-crit{background:rgba(255,23,68,0.12);color:var(--red);border:1px solid rgba(255,23,68,0.3);}
.s-recv{background:rgba(0,229,255,0.08);color:var(--accent-a);border:1px solid rgba(0,229,255,0.3);}
.branch-body{padding:16px 18px;display:flex;flex-direction:column;gap:14px;}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.stat-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 14px;}
.stat-label{font-size:10px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:4px;}
.stat-value{font-family:var(--mono);font-size:22px;color:var(--text);}
.stat-value .unit{font-size:12px;color:var(--muted);margin-left:3px;}
.sec-head{font-size:10px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-bottom:6px;display:flex;align-items:center;gap:8px;}
.sec-head::after{content:'';flex:1;height:1px;background:var(--border);}
.bar-wrap{background:var(--surface);border-radius:4px;height:8px;overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;transition:width 0.6s ease;}
.bar-label-row{display:flex;justify-content:space-between;margin-bottom:4px;}
.bar-lbl{font-family:var(--mono);font-size:11px;color:var(--muted);}
.bar-val{font-family:var(--mono);font-size:11px;}
.temp-gauge{position:relative;height:6px;background:linear-gradient(90deg,#00e676 0%,#ffab00 60%,#ff1744 100%);border-radius:3px;margin-bottom:8px;}
.temp-needle{position:absolute;top:-4px;width:3px;height:14px;background:white;border-radius:2px;transform:translateX(-50%);transition:left 0.6s ease;box-shadow:0 0 6px rgba(255,255,255,0.6);}
.rise-bar-wrap{flex:1;background:var(--surface);border-radius:4px;height:6px;overflow:hidden;}
.rise-bar-fill{height:100%;border-radius:4px;transition:width 0.6s ease;}
.pwm-row{display:flex;align-items:center;gap:16px;}
.pwm-pct{font-family:var(--mono);font-size:36px;font-weight:600;line-height:1;}
.pwm-sub{font-size:11px;color:var(--muted);letter-spacing:1px;margin-top:2px;}
/* ── ETA inline strip inside branch card ── */
.eta-strip{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.eta-strip-box{background:var(--surface);border-radius:6px;padding:8px 12px;border:1px solid var(--border);}
.eta-strip-lbl{font-size:9px;letter-spacing:2px;color:var(--muted);margin-bottom:2px;}
.eta-strip-val{font-family:var(--mono);font-size:17px;}
/* ── Charts ── */
.charts-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}
@media(max-width:700px){.charts-row{grid-template-columns:1fr;}}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px;}
.chart-title{font-size:12px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:14px;}
.chart-canvas-wrap{position:relative;height:140px;}
footer{text-align:center;font-family:var(--mono);font-size:11px;color:var(--muted);padding-top:16px;border-top:1px solid var(--border);}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <div class="logo">ESP32 // Adaptive BMS + ML Predictor</div>
    <h1>Live <span>Dashboard</span></h1>
  </div>
  <div class="status-pill"><div class="dot" id="connDot"></div><span id="connLabel">CONNECTING...</span></div>
</header>
<div class="meta-row">
  <div class="meta-chip">UPTIME<span id="uptime">--:--:--</span></div>
  <div class="meta-chip">SAMPLE<span id="sampleN">--</span></div>
  <div class="meta-chip">REFRESH<span>1s</span></div>
  <div class="meta-chip">NOMINAL CAP<span>2500 mAh</span></div>
  <div class="meta-chip">ML MODEL<span>LinReg (957 samples)</span></div>
</div>
<div class="alert-banner" id="alertBanner"></div>

<!-- ── ML ETA panel ── -->
<div class="eta-row">
  <div class="eta-card">
    <div class="eta-header">Branch A — Overheat ETA <span class="ml-badge">ML</span></div>
    <div class="eta-grid">
      <div class="eta-box"><div class="eta-label">Warn (45°C)</div><div class="eta-val warn" id="etaWarnA">--</div></div>
      <div class="eta-box"><div class="eta-label">Critical (55°C)</div><div class="eta-val crit" id="etaCritA">--</div></div>
    </div>
  </div>
  <div class="eta-card">
    <div class="eta-header">Branch B — Overheat ETA <span class="ml-badge">ML</span></div>
    <div class="eta-grid">
      <div class="eta-box"><div class="eta-label">Warn (45°C)</div><div class="eta-val warn" id="etaWarnB">--</div></div>
      <div class="eta-box"><div class="eta-label">Critical (55°C)</div><div class="eta-val crit" id="etaCritB">--</div></div>
    </div>
  </div>
</div>

<div class="branches">
  <!-- Branch A -->
  <div class="branch-card" id="cardA">
    <div class="branch-header"><div class="branch-title a">Branch A</div><div class="branch-status s-ok" id="statusA">STABLE</div></div>
    <div class="branch-body">
      <div>
        <div class="sec-head">Thermal</div>
        <div class="temp-gauge"><div class="temp-needle" id="needleA"></div></div>
        <div class="stat-grid">
          <div class="stat-box"><div class="stat-label">Temperature</div><div class="stat-value" id="tempA">--<span class="unit">°C</span></div></div>
          <div class="stat-box"><div class="stat-label">Rise Rate (EMA)</div><div class="stat-value" id="dtdtA">--<span class="unit">°C/s</span></div></div>
        </div>
        <div style="margin-top:8px">
          <div class="bar-label-row"><span class="bar-lbl">dT/dt vs threshold</span><span class="bar-val" id="dtdtBarValA">0%</span></div>
          <div class="rise-bar-wrap"><div class="rise-bar-fill" id="dtdtBarA" style="width:0%;background:var(--green)"></div></div>
        </div>
      </div>
      <div>
        <div class="sec-head">Electrical</div>
        <div class="stat-grid">
          <div class="stat-box"><div class="stat-label">Voltage</div><div class="stat-value" id="voltA">--<span class="unit">V</span></div></div>
          <div class="stat-box"><div class="stat-label">Current</div><div class="stat-value" id="currA">--<span class="unit">mA</span></div></div>
        </div>
        <div class="stat-box" style="margin-top:10px"><div class="stat-label">Power</div><div class="stat-value" id="powerA">--<span class="unit">W</span></div></div>
      </div>
      <div>
        <div class="sec-head">PWM Throttle</div>
        <div class="pwm-row">
          <canvas id="arcA" width="80" height="80"></canvas>
          <div>
            <div class="pwm-pct" id="pwmPctA" style="color:var(--accent-a)">--%</div>
            <div class="pwm-sub">DUTY CYCLE</div>
            <div style="margin-top:8px;font-family:var(--mono);font-size:11px;color:var(--muted)">Raw: <span id="pwmRawA" style="color:var(--text)">--</span>/255</div>
          </div>
        </div>
      </div>
      <div>
        <div class="sec-head">State of Charge</div>
        <div class="bar-label-row"><span class="bar-lbl">SoC</span><span class="bar-val" id="socValA">--%</span></div>
        <div class="bar-wrap" style="margin-bottom:8px"><div class="bar-fill" id="socBarA" style="width:100%;background:var(--green)"></div></div>
        <div class="bar-label-row"><span class="bar-lbl">DoD</span><span class="bar-val" id="dodValA">--%</span></div>
        <div class="bar-wrap"><div class="bar-fill" id="dodBarA" style="width:0%;background:var(--accent-a)"></div></div>
      </div>
    </div>
  </div>
  <!-- Branch B -->
  <div class="branch-card" id="cardB">
    <div class="branch-header"><div class="branch-title b">Branch B</div><div class="branch-status s-ok" id="statusB">STABLE</div></div>
    <div class="branch-body">
      <div>
        <div class="sec-head">Thermal</div>
        <div class="temp-gauge"><div class="temp-needle" id="needleB"></div></div>
        <div class="stat-grid">
          <div class="stat-box"><div class="stat-label">Temperature</div><div class="stat-value" id="tempB">--<span class="unit">°C</span></div></div>
          <div class="stat-box"><div class="stat-label">Rise Rate (EMA)</div><div class="stat-value" id="dtdtB">--<span class="unit">°C/s</span></div></div>
        </div>
        <div style="margin-top:8px">
          <div class="bar-label-row"><span class="bar-lbl">dT/dt vs threshold</span><span class="bar-val" id="dtdtBarValB">0%</span></div>
          <div class="rise-bar-wrap"><div class="rise-bar-fill" id="dtdtBarB" style="width:0%;background:var(--green)"></div></div>
        </div>
      </div>
      <div>
        <div class="sec-head">Electrical</div>
        <div class="stat-grid">
          <div class="stat-box"><div class="stat-label">Voltage</div><div class="stat-value" id="voltB">--<span class="unit">V</span></div></div>
          <div class="stat-box"><div class="stat-label">Current</div><div class="stat-value" id="currB">--<span class="unit">mA</span></div></div>
        </div>
        <div class="stat-box" style="margin-top:10px"><div class="stat-label">Power</div><div class="stat-value" id="powerB">--<span class="unit">W</span></div></div>
      </div>
      <div>
        <div class="sec-head">PWM Throttle</div>
        <div class="pwm-row">
          <canvas id="arcB" width="80" height="80"></canvas>
          <div>
            <div class="pwm-pct" id="pwmPctB" style="color:var(--accent-b)">--%</div>
            <div class="pwm-sub">DUTY CYCLE</div>
            <div style="margin-top:8px;font-family:var(--mono);font-size:11px;color:var(--muted)">Raw: <span id="pwmRawB" style="color:var(--text)">--</span>/255</div>
          </div>
        </div>
      </div>
      <div>
        <div class="sec-head">State of Charge</div>
        <div class="bar-label-row"><span class="bar-lbl">SoC</span><span class="bar-val" id="socValB">--%</span></div>
        <div class="bar-wrap" style="margin-bottom:8px"><div class="bar-fill" id="socBarB" style="width:100%;background:var(--green)"></div></div>
        <div class="bar-label-row"><span class="bar-lbl">DoD</span><span class="bar-val" id="dodValB">--%</span></div>
        <div class="bar-wrap"><div class="bar-fill" id="dodBarB" style="width:0%;background:var(--accent-b)"></div></div>
      </div>
    </div>
  </div>
</div>

<div class="charts-row">
  <div class="chart-card"><div class="chart-title">Temperature History (°C)</div><div class="chart-canvas-wrap"><canvas id="chartTemp"></canvas></div></div>
  <div class="chart-card"><div class="chart-title">PWM Duty Cycle History (%)</div><div class="chart-canvas-wrap"><canvas id="chartPwm"></canvas></div></div>
</div>
<div class="charts-row">
  <div class="chart-card"><div class="chart-title">Voltage History (V)</div><div class="chart-canvas-wrap"><canvas id="chartVolt"></canvas></div></div>
  <div class="chart-card"><div class="chart-title">State of Charge (%)</div><div class="chart-canvas-wrap"><canvas id="chartSoc"></canvas></div></div>
</div>
<footer>ESP32 Adaptive BMS + ML Overheat Predictor &nbsp;|&nbsp; LinReg trained on 957 samples &nbsp;|&nbsp; /data for raw JSON</footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const MAX=60,labels=[],tAH=[],tBH=[],pAH=[],pBH=[],vAH=[],vBH=[],sAH=[],sBH=[];
function ds(label,data,color,dash=[]){return{label,data,borderColor:color,backgroundColor:'transparent',borderWidth:2,borderDash:dash,pointRadius:0,tension:0.4};}
function cfg(ds){return{type:'line',data:{labels,datasets:ds},options:{responsive:true,maintainAspectRatio:false,animation:{duration:400},plugins:{legend:{display:true,labels:{color:'#5c6370',font:{family:"'Share Tech Mono'",size:11},boxWidth:12,padding:16}}},scales:{x:{ticks:{color:'#5c6370',font:{size:10},maxTicksLimit:8},grid:{color:'#1e2530'}},y:{ticks:{color:'#5c6370',font:{size:10}},grid:{color:'#1e2530'}}}}};}
const cT=new Chart(document.getElementById('chartTemp'),cfg([ds('A',tAH,'#00e5ff'),ds('B',tBH,'#7c4dff',[4,3])]));
const cP=new Chart(document.getElementById('chartPwm'),cfg([ds('A',pAH,'#00e5ff'),ds('B',pBH,'#7c4dff',[4,3])]));
const cV=new Chart(document.getElementById('chartVolt'),cfg([ds('A',vAH,'#00e676'),ds('B',vBH,'#ffab00',[4,3])]));
const cS=new Chart(document.getElementById('chartSoc'),cfg([ds('A',sAH,'#00e676'),ds('B',sBH,'#ff1744',[4,3])]));
function push(a,v){a.push(v);if(a.length>MAX)a.shift();}

function drawArc(id,pct,color){
  const c=document.getElementById(id);if(!c)return;
  const ctx=c.getContext('2d'),cx=40,cy=44,r=32,s=Math.PI*0.75,e=Math.PI*2.25;
  ctx.clearRect(0,0,80,80);
  ctx.beginPath();ctx.arc(cx,cy,r,s,e);ctx.strokeStyle='#1e2530';ctx.lineWidth=8;ctx.lineCap='round';ctx.stroke();
  ctx.beginPath();ctx.arc(cx,cy,r,s,s+(e-s)*(pct/100));ctx.strokeStyle=color;ctx.lineWidth=8;ctx.lineCap='round';ctx.stroke();
}
function tempColor(t){return t>=55?'#ff1744':t>=45?'#ffab00':'#00e676';}
function needlePos(t){return(Math.min(Math.max((t-15)/(65-15),0),1)*100).toFixed(1)+'%';}
function statusCls(t,d,duty){
  if(t>=55||(d>0.08&&duty<200))return['s-crit','CRITICAL'];
  if(t>=45||d>0.08)return['s-warn','THROTTLING'];
  if(duty<255)return['s-recv','RECOVERING'];
  return['s-ok','STABLE'];
}
function socColor(s){return s>60?'#00e676':s>30?'#ffab00':'#ff1744';}
function etaColor(s){if(s<0||s==='PAST')return'var(--muted)';if(s<1800)return'var(--red)';if(s<3600)return'var(--amber)';return'var(--green)';}

function applyBranch(px,d,accent){
  const f2=v=>isNaN(v)?'--':v.toFixed(2),f1=v=>isNaN(v)?'--':v.toFixed(1),f3=v=>isNaN(v)?'--':v.toFixed(3),f4=v=>isNaN(v)?'--':v.toFixed(4);
  const tc=tempColor(d.temp);
  document.getElementById('temp'+px).innerHTML=`<span style="color:${tc}">${f2(d.temp)}</span><span class="unit">°C</span>`;
  document.getElementById('dtdt'+px).innerHTML=f4(d.dtdt_ema)+'<span class="unit">°C/s</span>';
  document.getElementById('volt'+px).innerHTML=f3(d.volt)+'<span class="unit">V</span>';
  document.getElementById('curr'+px).innerHTML=f1(d.curr)+'<span class="unit">mA</span>';
  document.getElementById('power'+px).innerHTML=f3(d.power)+'<span class="unit">W</span>';
  const dp=Math.min((d.dtdt/0.3)*100,100),dc=d.dtdt>0.08?'#ff1744':d.dtdt>0.04?'#ffab00':'#00e676';
  document.getElementById('dtdtBar'+px).style.cssText=`width:${dp.toFixed(1)}%;background:${dc}`;
  document.getElementById('dtdtBarVal'+px).textContent=dp.toFixed(0)+'%';
  document.getElementById('needle'+px).style.left=needlePos(d.temp);
  document.getElementById('pwmPct'+px).textContent=d.dutyPct.toFixed(1)+'%';
  document.getElementById('pwmRaw'+px).textContent=d.duty;
  drawArc('arc'+px,d.dutyPct,accent);
  const sc=socColor(d.soc);
  document.getElementById('socVal'+px).textContent=f1(d.soc)+'%';
  document.getElementById('socBar'+px).style.cssText=`width:${d.soc.toFixed(1)}%;background:${sc}`;
  document.getElementById('dodVal'+px).textContent=f1(d.dod)+'%';
  document.getElementById('dodBar'+px).style.width=d.dod.toFixed(1)+'%';
  const[sc2,sl]=statusCls(d.temp,d.dtdt,d.duty);
  const se=document.getElementById('status'+px);se.className='branch-status '+sc2;se.textContent=sl;
  const card=document.getElementById('card'+px);
  card.className='branch-card '+(d.temp>=55?'alert-crit':d.temp>=45||d.dtdt>0.08?'alert-a':'');
  // ETA top panel
  const warnEl=document.getElementById('etaWarn'+px);
  const critEl=document.getElementById('etaCrit'+px);
  warnEl.textContent=d.eta_warn; warnEl.style.color=d.eta_warn==='PAST'?'var(--muted)':etaColor(d.eta_warn_s);
  critEl.textContent=d.eta_crit; critEl.style.color=d.eta_crit==='PAST'?'var(--muted)':etaColor(d.eta_crit_s);
}

function fmtUp(s){const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(sec).padStart(2,'0');}
let failCount=0;

async function fetchData(){
  try{
    const res=await fetch('/data',{cache:'no-store'});
    if(!res.ok)throw new Error();
    const d=await res.json();
    failCount=0;
    document.getElementById('connDot').className='dot';
    document.getElementById('connLabel').textContent='LIVE';
    document.getElementById('uptime').textContent=' '+fmtUp(d.uptime);
    document.getElementById('sampleN').textContent=' #'+d.sample;
    applyBranch('A',d.A,'#00e5ff');
    applyBranch('B',d.B,'#7c4dff');
    const t=new Date().toLocaleTimeString();
    push(labels,t);push(tAH,+d.A.temp.toFixed(2));push(tBH,+d.B.temp.toFixed(2));
    push(pAH,+d.A.dutyPct.toFixed(1));push(pBH,+d.B.dutyPct.toFixed(1));
    push(vAH,+d.A.volt.toFixed(3));push(vBH,+d.B.volt.toFixed(3));
    push(sAH,+d.A.soc.toFixed(1));push(sBH,+d.B.soc.toFixed(1));
    cT.update();cP.update();cV.update();cS.update();
    const alerts=[];
    if(d.A.temp>=55)alerts.push('[A] CRITICAL: '+d.A.temp.toFixed(1)+'°C');
    if(d.B.temp>=55)alerts.push('[B] CRITICAL: '+d.B.temp.toFixed(1)+'°C');
    if(d.A.dtdt>0.08)alerts.push('[A] RAPID RISE: +'+d.A.dtdt.toFixed(4)+'°C/s');
    if(d.B.dtdt>0.08)alerts.push('[B] RAPID RISE: +'+d.B.dtdt.toFixed(4)+'°C/s');
    if(d.A.soc<15)alerts.push('[A] LOW SoC: '+d.A.soc.toFixed(1)+'%');
    if(d.B.soc<15)alerts.push('[B] LOW SoC: '+d.B.soc.toFixed(1)+'%');
    if(d.A.eta_crit_s>0&&d.A.eta_crit_s<1800)alerts.push('[A] OVERHEAT <30min: ETA '+d.A.eta_crit);
    if(d.B.eta_crit_s>0&&d.B.eta_crit_s<1800)alerts.push('[B] OVERHEAT <30min: ETA '+d.B.eta_crit);
    const bn=document.getElementById('alertBanner');
    if(alerts.length){bn.className='alert-banner show';bn.textContent='⚠ '+alerts.join('  |  ');}
    else{bn.className='alert-banner';}
  }catch(e){
    if(++failCount>3){document.getElementById('connDot').className='dot dead';document.getElementById('connLabel').textContent='DISCONNECTED';}
  }
}
fetchData();
setInterval(fetchData,1000);
</script>
</body>
</html>
)rawhtml";
  server.send(200, "text/html", html);
}

// ─────────────────────────────────────────────────────────────────
// setup()
// ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(300);

  ledcAttach(PIN_MOSFET_A, PWM_FREQ, PWM_RES);
  ledcAttach(PIN_MOSFET_B, PWM_FREQ, PWM_RES);
  ledcWrite(PIN_MOSFET_A, 255);
  ledcWrite(PIN_MOSFET_B, 255);

  // Seed EMA dTdt from ML prior so ETA is valid from sample #1
  brA.dTdt_ema = ML_MEAN_DTDT_A;
  brB.dTdt_ema = ML_MEAN_DTDT_B;

  sensors.begin();
  inaA.begin();
  inaB.begin();

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(200);

  bool connected = false;
  for (int attempt = 1; attempt <= 3 && !connected; attempt++) {
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
      delay(500);
    }
    if (WiFi.status() == WL_CONNECTED) {
      connected = true;
      wifiConnected = true;
    } else {
      WiFi.disconnect(true);
      delay(500);
    }
  }

  if (connected) {
    Serial.print("http://");
    Serial.println(WiFi.localIP());
  } else {
    WiFi.mode(WIFI_AP);
    WiFi.softAP("BMS_Dashboard", "bms12345");
    delay(300);
    Serial.print("HOTSPOT MODE — http://");
    Serial.println(WiFi.softAPIP());
  }

  server.on("/",     handleRoot);
  server.on("/data", handleJSON);
  server.begin();
  startTime = millis();
}

// ─────────────────────────────────────────────────────────────────
// loop()
// ─────────────────────────────────────────────────────────────────
void loop() {
  server.handleClient();

  if (wifiConnected && WiFi.status() != WL_CONNECTED) {
    static unsigned long lastReconnect = 0;
    if (millis() - lastReconnect > 10000) {
      lastReconnect = millis();
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  }

  unsigned long now = millis();
  if (now - lastMeasure < DT_INTERVAL) return;
  float dt = (now - lastMeasure) / 1000.0f;
  lastMeasure = now;
  loopCount++;

  // ── Temperature measurement ──────────────────────────────────
  sensors.requestTemperatures();
  float tA = sensors.getTempCByIndex(0);
  float tB = sensors.getTempCByIndex(1);
  if (tA < -50.0f) tA = tA_prev;
  if (tB < -50.0f) tB = tB_prev;

  // ── Electrical measurement ───────────────────────────────────
  float vA = inaA.getBusVoltage_V();
  float iA = inaA.getCurrent_mA();
  float vB = inaB.getBusVoltage_V();
  float iB = inaB.getCurrent_mA();

  // ── Instantaneous dT/dt ──────────────────────────────────────
  float dTdtA_raw = (tA_prev > -900.0f) ? (tA - tA_prev) / dt : ML_MEAN_DTDT_A;
  float dTdtB_raw = (tB_prev > -900.0f) ? (tB - tB_prev) / dt : ML_MEAN_DTDT_B;

  // ── EMA-smooth dT/dt (reduces sensor noise) ──────────────────
  brA.dTdt_ema = (1.0f - ETA_EMA_ALPHA) * brA.dTdt_ema + ETA_EMA_ALPHA * dTdtA_raw;
  brB.dTdt_ema = (1.0f - ETA_EMA_ALPHA) * brB.dTdt_ema + ETA_EMA_ALPHA * dTdtB_raw;

  // ── PWM throttle (unchanged logic) ───────────────────────────
  if (tA >= CRITICAL_TEMP || dTdtA_raw > RISE_THRESHOLD) brA.duty = max(0, brA.duty - 75);
  else if (tA < 35.0f && brA.duty < 255)                 brA.duty = min(255, brA.duty + 25);

  if (tB >= CRITICAL_TEMP || dTdtB_raw > RISE_THRESHOLD) brB.duty = max(0, brB.duty - 75);
  else if (tB < 35.0f && brB.duty < 255)                 brB.duty = min(255, brB.duty + 25);

  ledcWrite(PIN_MOSFET_A, brA.duty);
  ledcWrite(PIN_MOSFET_B, brB.duty);

  // ── Update branch structs ─────────────────────────────────────
  brA.temp  = tA;
  brA.dTdt  = dTdtA_raw;
  brA.volt  = vA;
  brA.curr  = iA;
  brA.power = (vA * iA) / 1000.0f;
  brA.soc   = runKalman(kA, vA, iA, dt);

  brB.temp  = tB;
  brB.dTdt  = dTdtB_raw;
  brB.volt  = vB;
  brB.curr  = iB;
  brB.power = (vB * iB) / 1000.0f;
  brB.soc   = runKalman(kB, vB, iB, dt);

  // ── ML-blended ETA to WARN (45°C) and CRITICAL (55°C) ────────
  brA.eta_warn_s = computeETA(tA, brA.dTdt_ema, ML_SLOPE_A, WARN_TARGET);
  brA.eta_crit_s = computeETA(tA, brA.dTdt_ema, ML_SLOPE_A, OVERHEAT_TARGET);
  brB.eta_warn_s = computeETA(tB, brB.dTdt_ema, ML_SLOPE_B, WARN_TARGET);
  brB.eta_crit_s = computeETA(tB, brB.dTdt_ema, ML_SLOPE_B, OVERHEAT_TARGET);

  // ── Serial log (useful for Serial Plotter) ───────────────────
  Serial.printf(
    "[%4lu] tA=%.2f tB=%.2f | dTdt_A=%.4f dTdt_B=%.4f "
    "| ETA_crit_A=%.0fs ETA_crit_B=%.0fs "
    "| dutyA=%d dutyB=%d | socA=%.1f socB=%.1f\n",
    loopCount, tA, tB,
    brA.dTdt_ema, brB.dTdt_ema,
    brA.eta_crit_s, brB.eta_crit_s,
    brA.duty, brB.duty,
    brA.soc, brB.soc
  );

  tA_prev = tA;
  tB_prev = tB;
}