import serial
import pynmea2
import time

PORT = "/dev/serial0"   # thường là serial0 trên Raspberry Pi
BAUD = 9600             # GNSS L76K thường 9600

ser = serial.Serial(PORT, BAUD, timeout=1)

last_print = 0
while True:
    line = ser.readline().decode("ascii", errors="replace").strip()
    if not line.startswith("$"):
        continue

    # Lọc câu hay dùng để lấy vị trí
    if line.startswith("$GNRMC") or line.startswith("$GPRMC") or line.startswith("$GNGGA") or line.startswith("$GPGGA"):
        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            continue

        lat = getattr(msg, "latitude", None)
        lon = getattr(msg, "longitude", None)

        # Một số câu có status (RMC): A=valid, V=void
        status = getattr(msg, "status", None)

        if lat and lon:
            now = time.time()
            if now - last_print > 1:
                print(f"lat={lat:.6f}, lon={lon:.6f}, status={status}, sentence={msg.sentence_type}")
                last_print = now