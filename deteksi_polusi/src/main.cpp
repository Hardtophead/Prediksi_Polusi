// main.cpp

#include <Arduino.h>           // Core Arduino functions (pinMode, digitalWrite, delay, etc.)
#include <WiFi.h>              // ESP32 Wi-Fi support
#include <WiFiClient.h>        // TCP client for network communications (used by ThingSpeak)
#include <MQUnifiedsensor.h>   // Abstraction layer for MQ-series gas sensors (MQ-7, MQ-135)
#include "Sensor.h"            // Helper class: reads DHT22 & SDS011 data
#include "ThingspeakManager.h" // Helper class: sends data to ThingSpeak

// ————————————————————————— Wi-Fi Credentials ———————————————————————————
#define WIFI_SSID "Medan Gaya"     // SSID jaringan Wi-Fi
#define WIFI_PASS "88888888" // Password jaringan Wi-Fi

// ————————————————————— ThingSpeak Channel Parameters ———————————————————
#define ENV_API_KEY "8HUN4CLTEV167KFA" // API key untuk channel lingkungan
#define ENV_CHANNEL_ID 2990169         // ID channel lingkungan (DHT22 + SDS011)

#define MQ7_API_KEY "D3U4K8SPWBQ4SN4Z" // API key untuk channel MQ-7
#define MQ7_CHANNEL_ID 2990179         // ID channel MQ-7

#define MQ135_API_KEY "7BE9Y69KWELBFERX" // API key untuk channel MQ-135
#define MQ135_CHANNEL_ID 2990177         // ID channel MQ-135

// ————————————————————— Hardware Pin Assignments & ADC Settings ——————————
#define DHT_PIN 4 // GPIO pin untuk data DHT22
#define SDS_RX 16 // RX pin untuk serial SDS011
#define SDS_TX 17 // TX pin untuk serial SDS011

#define MQ7_PIN 32   // ADC channel (GPIO32) untuk MQ-7
#define MQ135_PIN 35 // ADC channel (GPIO35) untuk MQ-135

#define VOLTAGE_RESOLUTION 3.3f // Tegangan referensi ADC (3.3V)
#define ADC_BIT_RESOLUTION 12   // Resolusi ADC (12 bit ⇒ 0–4095)

// —————————————————— Clean-air Ratios & Regression Coefficients —————————
#define MQ7_CLEAN_AIR_RATIO 27.0f  // Rs/R0 ratio di udara bersih MQ-7 (datasheet)
#define MQ135_CLEAN_AIR_RATIO 3.6f // Rs/R0 ratio di udara bersih MQ-135

const float MQ7_A = 36974.0f, MQ7_B = -3.109f;    // Regresi log-log PPM untuk CO (MQ-7)
const float MQ135_A = 110.47f, MQ135_B = -2.862f; // Regresi log-log PPM untuk CO₂ (MQ-135)

// ————————————————————— Instantiate MQ Sensors —————————————————————
MQUnifiedsensor MQ7("ESP-32", VOLTAGE_RESOLUTION, ADC_BIT_RESOLUTION, MQ7_PIN, "MQ-7");
MQUnifiedsensor MQ135("ESP-32", VOLTAGE_RESOLUTION, ADC_BIT_RESOLUTION, MQ135_PIN, "MQ-135");

// ————————————————————— Support Objects —————————————————————————
Sensor sensor(DHT_PIN, MQ135_PIN, MQ7_PIN, Serial2, SDS_RX, SDS_TX);
//   sensor: instance of Sensor class, reads DHT22 & SDS011, plus raw ADC for MQs
WiFiClient client; // TCP client used for ThingSpeak HTTP requests
// ThingSpeak managers for each channel:
ThingspeakManager tsEnv(ENV_API_KEY, ENV_CHANNEL_ID);
ThingspeakManager ts7(MQ7_API_KEY, MQ7_CHANNEL_ID);
ThingspeakManager ts135(MQ135_API_KEY, MQ135_CHANNEL_ID);

void setup()
{
  Serial.begin(115200);                     // Mulai Serial debug @115200bps
  analogReadResolution(ADC_BIT_RESOLUTION); // Set ADC ke 12-bit

  sensor.begin(); // Init DHT22 & SDS011

  // Hubungkan ke Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" OK");

  // Init ThingSpeak clients (harus setelah Wi-Fi connected)
  tsEnv.begin(client);
  ts7.begin(client);
  ts135.begin(client);

  // Konfigurasi regression method & koefisien untuk MQ sensors
  MQ7.setRegressionMethod(1);
  MQ7.setA(MQ7_A);
  MQ7.setB(MQ7_B);
  MQ135.setRegressionMethod(1);
  MQ135.setA(MQ135_A);
  MQ135.setB(MQ135_B);

  // Set load resistance RL dalam kΩ (datasheet)
  MQ7.setRL(21.00f);
  MQ135.setRL(21.00f);

  // Inisialisasi ADC sampling library
  MQ7.init();
  MQ135.init();

  // Kalibrasi R0 di udara bersih → rata-rata 10 sampel
  auto calibrateSensor = [&](MQUnifiedsensor &mqs, float cleanAirRatio, const char *name)
  {
    Serial.print("Calibrating ");
    Serial.print(name);
    Serial.print(" ...");
    float sum = 0;
    for (int i = 0; i < 10; i++)
    {
      mqs.update();                        // Update ADC reading
      sum += mqs.calibrate(cleanAirRatio); // Hitung R0 sampel
      Serial.print(".");
      delay(200);
    }
    mqs.setR0(sum / 10.0f); // Set R0 rata-rata
    Serial.println(" done");
  };
  calibrateSensor(MQ7, MQ7_CLEAN_AIR_RATIO, "MQ-7");
  calibrateSensor(MQ135, MQ135_CLEAN_AIR_RATIO, "MQ-135");
}

void loop()
{
  // ————————— MQ-7 Heat-Cycle & Measurement ————————
  float ppmCO = MQ7.readSensor(); // PPM mentah
  unsigned long start = millis();
  // High-heat 6 detik (5V)
  while (millis() - start < 6000)
  {
    analogWrite(5, 255);      // Output ~5V pada pin 5
    MQ7.update();             // Update ADC
    MQ7.readSensor(false, 0); // Baca nilai tanpa recalibrate
    MQ7.serialDebug();        // Print debug ke Serial
    delay(500);
  }
  // Low-heat 9 detik (~1.4V)
  start = millis();
  while (millis() - start < 9000)
  {
    analogWrite(5, 20); // Output ~1.4V
    MQ7.update();
    MQ7.readSensor(false, 0);
    MQ7.serialDebug();
    delay(500);
  }

  MQ7.update(); // Update sekali lagi
  // Hitung parameter:
  float volts7 = MQ7.getVoltage(false);                             // Vrl
  float rs7 = MQ7.getRL() * (VOLTAGE_RESOLUTION / volts7 - 1.0f);   // Rs = RL*(Vref/Vrl −1)
  float ro7 = MQ7.getR0();                                          // R0 hasil kalibrasi
  float ratio7 = rs7 / ro7;                                         // Rs/R0
  float ppmDS7 = MQ7_A * powf(ratio7, MQ7_B);                       // PPM dari kurva
  float error7 = (ppmDS7 != 0) ? fabs(ppmCO - ppmDS7) / ppmDS7 : 0; // error
  float accuracy7 = 1.0f - error7;                                  // akurasi

  // Kirim MQ-7 data ke ThingSpeak
  ts7.sendMQData(ratio7, ppmDS7, ppmCO, rs7 * 1000, ro7 * 1000, volts7, error7, accuracy7);
  Serial.printf("MQ-7 | Ratio: %.2f | PPM DS: %.2f | PPM Sensor: %.2f | Rs: %.0fΩ | Ro: %.0fΩ | VRL: %.2fV | Err: %.2f | Acc: %.2f\n\n",
                ratio7, ppmDS7, ppmCO, rs7 * 1000, ro7 * 1000, volts7, error7, accuracy7);

  // ————————— MQ-135 Heat-Cycle & Measurement ————————
  float rawCO2 = MQ135.readSensor() + 400; // PPM mentah CO₂
  start = millis();
  while (millis() - start < 6000)
  {
    analogWrite(5, 255);
    MQ135.update();
    MQ135.readSensor(false, 0);
    MQ135.serialDebug();
    delay(500);
  }
  start = millis();
  while (millis() - start < 9000)
  {
    analogWrite(5, 20);
    MQ135.update();
    MQ135.readSensor(false, 0);
    MQ135.serialDebug();
    delay(500);
  }

  MQ135.update();
  float volts135 = MQ135.getVoltage(false);
  float rs135 = MQ135.getRL() * (VOLTAGE_RESOLUTION / volts135 - 1.0f);
  float ro135 = MQ135.getR0();
  float ratio135 = rs135 / ro135;
  float ppmDS135 = 400 + MQ135_A * powf(ratio135, MQ135_B);
  float error135 = (ppmDS135 != 0) ? fabs(rawCO2 - ppmDS135) / ppmDS135 : 0;
  float accuracy135 = 1.0f - error135;

  // Kirim MQ-135 data ke ThingSpeak
  ts135.sendMQData(ratio135, ppmDS135, rawCO2, rs135 * 1000, ro135 * 1000, volts135, error135, accuracy135);
  Serial.printf("MQ-135 | Ratio: %.2f | PPM DS: %.2f | PPM Sensor: %.2f | Rs: %.0fΩ | Ro: %.0fΩ | VRL: %.2fV | Err: %.2f | Acc: %.2f\n\n",
                ratio135, ppmDS135, rawCO2, rs135 * 1000, ro135 * 1000, volts135, error135, accuracy135);

  // ————————— Read & Send Environmental Data —————————
  float temp = sensor.readTemperature(); // Baca suhu (°C)
  float hum = sensor.readHumidity();     // Baca kelembapan (%RH)
  float pm25 = 0, pm10 = 0;
  sensor.readSDS(pm25, pm10);                              // Baca PM2.5 & PM10 (µg/m³)
  tsEnv.sendEnvData(temp, hum, pm25, pm10, ppmCO, rawCO2); // Kirim ke ThingSpeak
  Serial.printf("T: %.1f°C | H: %.1f%% | PM2.5: %.1f | PM10: %.1f| PPM Sensor MQ-7: %.2f | PPM Sensor NQ-135: %.2f |\n\n",
                temp, hum, pm25, pm10, ppmCO, rawCO2);
  delay(15000); // Tunggu 15 detik sebelum siklus berikutnya
}
