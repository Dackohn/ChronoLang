import ctypes

import os
import ctypes
import platform
print(platform.architecture())  # Must match your DLL build


# Construct the absolute path from this script to the DLL
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))

print("[INFO] Loading DLL from:", dll_path)
print("DLL exists:", os.path.exists(dll_path))
print("DLL path:", dll_path)
chrono = ctypes.CDLL(dll_path)

chrono.chrono_parse.argtypes = [ctypes.c_char_p]
chrono.chrono_parse.restype = ctypes.c_char_p

# Example use
code = b'LOAD data FROM "file.csv"\nSET WINDOW = 30d\n'

print("[PARSE]")
print(chrono.chrono_parse(code).decode())
