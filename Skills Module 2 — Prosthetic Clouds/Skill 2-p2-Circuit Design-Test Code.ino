/*
 * Pulse Oximeter and Temperature Sensor with SMS Notification
 * This code reads data from a heart pulse sensor and a temperature sensor.
 * If the body temperature exceeds a threshold or the oxygen level drops
 * below a threshold, an SMS notification is sent via the ESP8266 WiFi module.
 */

#include<Wire.h>
#include<LiquidCrystal_I2C.h>
#include<SoftwareSerial.h>

// Pin definitions
const int pulsePin = A0;
const int tempPin = A1;
const int ledPin = 7;
const int espTx = 9;
const int espRx = 10;

// Thresholds
const float tempThreshold = 37.5; // Celsius
const int pulseThreshold = 60; // BPM

// LCD setup
LiquidCrystal_I2C lcd(0x27, 16, 4);

// ESP8266 setup
SoftwareSerial espSerial(espRx, espTx);

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  espSerial.begin(115200);

  // Initialize LCD
  lcd.begin();
  lcd.backlight();

  // Initialize LED pin
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);

  // Initialize ESP8266
  sendCommand("AT", 1000);
  sendCommand("AT+CWMODE=1", 1000);
  sendCommand("AT+CWJAP=\"yourSSID\",\"yourPASSWORD\"", 5000);
}

void loop() {
  // Read pulse sensor data
  int pulseValue = analogRead(pulsePin);
  int pulseBPM = map(pulseValue, 0, 1023, 0, 150); // Example mapping

  // Read temperature sensor data
  int tempValue = analogRead(tempPin);
  float temperature = (tempValue / 1024.0) * 5.0 * 100.0;

  // Display data on LCD
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  lcd.print(temperature);
  lcd.print(" C");
  lcd.setCursor(0, 1);
  lcd.print("Pulse: ");
  lcd.print(pulseBPM);
  lcd.print(" BPM");

  // Check thresholds and send SMS if needed
  if (temperature > tempThreshold || pulseBPM < pulseThreshold) {
    digitalWrite(ledPin, HIGH);
    sendSMS("Alert! Temp: " + String(temperature) + " C, Pulse: " + String(pulseBPM) + " BPM");
  } else {
    digitalWrite(ledPin, LOW);
  }

  delay(1000); // Wait for 1 second before next reading
}

void sendCommand(String command, int timeout) {
  espSerial.println(command);
  long int time = millis();
  while ((time + timeout) > millis()) {
    while (espSerial.available()) {
      char c = espSerial.read();
      Serial.print(c);
    }
  }
}

void sendSMS(String message) {
  sendCommand("AT+CMGF=1", 1000); // Set SMS mode
  sendCommand("AT+CMGS=\"+1234567890\"", 1000); // Replace with your phone number
  espSerial.print(message);
  espSerial.write(26); // ASCII code for CTRL+Z
}