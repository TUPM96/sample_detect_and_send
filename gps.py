import serial
import pynmea2
import time

PORT = "/dev/ttyAMA0"
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=1)

last = 0
last_seen = {"sats": None, "fix": None, "lat": None, "lon": None, "status": None}

while True:
    line = ser.readline().decode("ascii", errors="replace").strip()
    if not line.startswith("$"):
        continue

    try:
        msg = pynmea2.parse(line)
    except pynmea2.ParseError:
        continue

    # GGA: có fix quality + satellites
    if msg.sentence_type == "GGA":
        # pynmea2: gps_qual (0/1/2...), num_sats
        last_seen["fix"] = getattr(msg, "gps_qual", None)
        last_seen["sats"] = getattr(msg, "num_sats", None)
        last_seen["lat"] = getattr(msg, "latitude", None) or last_seen["lat"]
        last_seen["lon"] = getattr(msg, "longitude", None) or last_seen["lon"]

    # RMC: có status A/V
    if msg.sentence_type == "RMC":
        last_seen["status"] = getattr(msg, "status", None)
        last_seen["lat"] = getattr(msg, "latitude", None) or last_seen["lat"]
        last_seen["lon"] = getattr(msg, "longitude", None) or last_seen["lon"]

    now = time.time()
    if now - last > 1:
        lat, lon = last_seen["lat"], last_seen["lon"]
        if lat and lon:
            print(f"FIX OK lat={lat:.6f} lon={lon:.6f} sats={last_seen['sats']} fix={last_seen['fix']} rmc={last_seen['status']}")
        else:
            print(f"Waiting for fix... sats={last_seen['sats']} fix={last_seen['fix']} rmc={last_seen['status']}")
        last = now