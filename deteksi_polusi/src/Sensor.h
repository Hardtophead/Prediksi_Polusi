#ifndef SENSOR_H
#define SENSOR_H

#include <Arduino.h>
#include <DHT.h>           // untuk DHT22
#include <sdsdustsensor.h> // untuk SDS011

class Sensor
{
public:
  // constructor hanya untuk DHT22 & SDS011
  //  dhtPin:    pin DHT22
  //  mq135Pin:  ADC pin MQ-135 (jika butuh debug ADC)
  //  mq7Pin:    ADC pin MQ-7       (jika butuh debug ADC)
  //  sdsSerial: Serial hardware untuk SDS011
  //  sdsRxPin / sdsTxPin: pin RX/TX SDS011
  Sensor(uint8_t dhtPin,
         uint8_t mq135Pin,
         uint8_t mq7Pin,
         HardwareSerial &sdsSerial,
         uint8_t sdsRxPin,
         uint8_t sdsTxPin);

  void begin();                           // inisialisasi DHT22 & SDS011
  float readTemperature();                // °C
  float readHumidity();                   // %RH
  bool readSDS(float &pm25, float &pm10); // µg/m³

private:
  uint8_t _dhtPin, _mq135Pin, _mq7Pin;
  DHT dht;                    // objek DHT22
  HardwareSerial &_sdsSerial; // serial SDS011
  uint8_t _sdsRxPin, _sdsTxPin;
  SdsDustSensor sds; // objek SDS011
};

#endif // SENSOR_H
