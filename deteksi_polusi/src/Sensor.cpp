#include "Sensor.h"

// constructor: hanya inisialisasi members & objek DHT/SDS
Sensor::Sensor(uint8_t dhtPin,
               uint8_t mq135Pin,
               uint8_t mq7Pin,
               HardwareSerial &sdsSerial,
               uint8_t sdsRxPin,
               uint8_t sdsTxPin)
    : _dhtPin(dhtPin),
      _mq135Pin(mq135Pin),
      _mq7Pin(mq7Pin),
      dht(dhtPin, DHT22),
      _sdsSerial(sdsSerial),
      _sdsRxPin(sdsRxPin),
      _sdsTxPin(sdsTxPin),
      sds(sdsSerial, sdsRxPin, sdsTxPin)
{
}

void Sensor::begin()
{
  dht.begin();              // mulai DHT22
  analogReadResolution(12); // ADC 12-bit
  _sdsSerial.begin(9600, SERIAL_8N1, _sdsRxPin, _sdsTxPin);
  sds.begin();                 // mulai SDS011
  sds.setQueryReportingMode(); // mode query manual
}

float Sensor::readTemperature()
{
  return dht.readTemperature(); // baca °C
}

float Sensor::readHumidity()
{
  return dht.readHumidity(); // baca %RH
}

bool Sensor::readSDS(float &pm25, float &pm10)
{
  auto res = sds.queryPm(); // minta PM2.5 & PM10
  if (res.isOk())
  {
    pm25 = res.pm25;
    pm10 = res.pm10;
    return true;
  }
  pm25 = pm10 = 0;
  return false;
}
