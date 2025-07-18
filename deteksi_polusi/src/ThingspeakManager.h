// ThingspeakManager.h
#ifndef THINGSPEAK_MANAGER_H
#define THINGSPEAK_MANAGER_H

#include <ThingSpeak.h>
#include <WiFi.h>

class ThingspeakManager
{
public:
  ThingspeakManager(const char *apiKey, unsigned long channelId);
  void begin(WiFiClient &client);

  // Untuk data lingkungan (DHT22 & SDS011)
  void sendEnvData(float temp, float hum, float pm25, float pm10, float ppmSensor7, float ppmSensor135);

  // Untuk data gas (MQ-7 & MQ-135)
  // field1=Rs/R0, field2=PPM datasheet, field3=PPM sensor,
  // field4=Rs(Ω), field5=Ro(Ω), field6=Vrl(V), field7=Error, field8=Accuracy
  void sendMQData(float ratio,
                  float ppmDatasheet,
                  float ppmSensor,
                  float rs,
                  float ro,
                  float vrl,
                  float errorPct,
                  float accuracyPct);

private:
  const char *_apiKey;
  unsigned long _channelId;
  WiFiClient *_client;
};

#endif // THINGSPEAK_MANAGER_H
