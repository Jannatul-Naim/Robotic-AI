
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm(0x40);

#define SERVO_FREQ 50
#define SERVO_MIN 100   // calibrate
#define SERVO_MAX 520

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);   // ESP32 I2C
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);
  delay(10);
}

int angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    int ch, ang;
    if (sscanf(cmd.c_str(), "%d %d", &ch, &ang) == 2) {
      pwm.setPWM(ch, 0, angleToPulse(ang));
    }
  }
}

