# Real-Time-Data-Acquisition-and-Predictive-Thermal-Mitigation-Framework-for-BMS-


ESP32 Adaptive Battery Management System (BMS)

An intelligent, real-time Battery Management System designed for 3S2P
Lithium-ion battery packs. This project moves beyond standard "threshold-based"
monitoring by using Predictive Thermal Mitigation and Kalman Filter state
estimation to prevent battery failure before it occurs.

🌟 Key Features

  - Predictive Overheat Prevention: Monitors the rate of temperature rise
    (dT/dt) to calculate a "Time to Overheat" (TTO).
  - Adaptive PWM Throttling: Instead of cutting power completely, the system
    uses 5kHz Pulse Width Modulation to "derate" current, allowing the battery
    to cool down while maintaining operation (Limp-home mode).
  - Dual-Branch Monitoring: Independent voltage, current, and temperature
    sensing for two parallel branches using a 3S2P configuration.
  - Recursive State Estimation: Utilizes a Kalman Filter to accurately estimate
    State of Charge (SoC) and Depth of Discharge (DoD) by fusing current
    integration and voltage mapping.
  - Real-Time Web Dashboard: A live, dark-mode dashboard served directly from
    the ESP32 via WebSockets for sub-second telemetry updates.

🛠️ Hardware Requirements

  - Microcontroller: ESP32 (DevKit V1)
  - Sensors:
      - 2x INA219 High-side Current/Voltage Sensors (I2C)
      - 2x DS18B20 Digital Temperature Probes (1-Wire)
  - Switching: 2x Logic-level MOSFETs (e.g., IRLZ44N)
  - Power: 12V Battery Pack (3S Configuration)
  - Resistors: 4.7k Ohm (for 1-Wire pull-up)

📁 Project Structure

  - BMS_Master.ino: The main C++ backend logic for ESP32.
  - index_html: The frontend dashboard code (HTML/Tailwind CSS/JS) embedded in
    the firmware.
  - data/: Contains historical discharge logs used for ML prior calculations.

🔧 Software Setup

1.  Arduino IDE: Ensure you have the ESP32 board manager installed.
2.  Libraries: Install the following via the Library Manager:
      - Adafruit INA219
      - DallasTemperature & OneWire
      - ArduinoJson (by Benoit Blanchon)
      - WebSockets (by Markus Sattler)
3.  Configuration: Update the WIFI_SSID and WIFI_PASS in the code.
4.  Addressing: Ensure your second INA219 has the A0 pad soldered (Address
    0x41).

🚀 How to Use

1.  Upload the code to your ESP32.
2.  Open the Serial Monitor at 115200 baud to find the local IP address.
3.  Connect your laptop/phone to the same WiFi network.
4.  Enter the IP address in your web browser (e.g., http://192.168.1.15).
5.  Watch the telemetry update live. To test the adaptive throttling, apply heat
    to one of the temperature probes; the dashboard will signal "THROTTLING" and
    the supply duty cycle will drop autonomously.

📊 Logic & Algorithms

Predictive ETA

The system calculates the slope of the temperature curve. ETA (Seconds) = (55°C
- Current Temp) / (Rise Rate)

PWM Modulation

If the temperature rise exceeds 0.08°C/sec, the MOSFET duty cycle is reduced by
~25% per second until the temperature stabilizes, physically reducing the I^2R
heat generation within the cells.

🤝 Contribution

Contributions are welcome! If you have ideas for improving the Kalman Filter
accuracy or adding SOC-balancing logic, feel free to fork the repo and submit a
pull request.

📜 License

This project is licensed under the MIT License - see the LICENSE file for
details.

Disclaimer: Working with Lithium-ion batteries is dangerous. Ensure you have
physical fuses and a fire-safe testing environment.
