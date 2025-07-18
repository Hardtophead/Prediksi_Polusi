// ThingspeakManager.cpp
#include "ThingspeakManager.h"

ThingspeakManager::ThingspeakManager(const char *apiKey, unsigned long channelId)
    : _apiKey(apiKey), _channelId(channelId), _client(nullptr) {}

void ThingspeakManager::begin(WiFiClient &client)
{
  _client = &client;
  ThingSpeak.begin(*_client); // Init ThingSpeak
}

void ThingspeakManager::sendEnvData(float temp, float hum, float pm25, float pm10, float ppmSensor7, float ppmSensor135)
{
  ThingSpeak.setField(1, temp);                // Field1 = suhu
  ThingSpeak.setField(2, hum);                 // Field2 = kelembapan
  ThingSpeak.setField(3, pm25);                // Field3 = PM2.5
  ThingSpeak.setField(4, pm10);                // Field4 = PM10
  ThingSpeak.setField(5, ppmSensor7);          // Field5 = PMMMQ7
  ThingSpeak.setField(6, ppmSensor135);        // Field6 = PMMMQ135
  ThingSpeak.writeFields(_channelId, _apiKey); // Kirim paket
}

void ThingspeakManager::sendMQData(float ratio,
                                   float ppmDatasheet,
                                   float ppmSensor,
                                   float rs,
                                   float ro,
                                   float vrl,
                                   float errorPct,
                                   float accuracyPct)
{
  ThingSpeak.setField(1, ratio);
  ThingSpeak.setField(2, ppmDatasheet);
  ThingSpeak.setField(3, ppmSensor);
  ThingSpeak.setField(4, rs);
  ThingSpeak.setField(5, ro);
  ThingSpeak.setField(6, vrl);
  ThingSpeak.setField(7, errorPct);
  ThingSpeak.setField(8, accuracyPct);
  ThingSpeak.writeFields(_channelId, _apiKey);
}
