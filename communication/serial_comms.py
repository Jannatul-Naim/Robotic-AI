import serial, time

ser = serial.Serial('/dev/ttyUSB0', 115200)
time.sleep(2)

def move(channel, angle):
    ser.write(f"{channel} {angle}\n".encode())
    time.sleep(0.1)