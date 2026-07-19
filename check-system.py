from datetime import datetime
import socket

print(f"Hostname: {socket.gethostname()}")
print(f"Date/Time: {datetime.now().astimezone().isoformat(sep=' ', timespec='seconds')}")
print("Hello from Windows!")
