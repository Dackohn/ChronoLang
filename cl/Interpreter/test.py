from ctypes import cdll, c_char_p
import json
import os
import platform
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <chrono_code>")
        print("Error: No ChronoLang code provided")
        sys.exit(1)

    src = sys.argv[1]

    if platform.system() == "Windows":
        lib_extension = ".dll"
    elif platform.system() == "Darwin":
        lib_extension = ".dylib"
    elif platform.system() == "Linux":
        lib_extension = ".so"
    else:
        raise OSError(f"Unsupported platform: {platform.system()}")

    lib_name = f"chronolang{lib_extension}"
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), lib_name))

    try:
        chrono = cdll.LoadLibrary(dll_path)
        chrono.chrono_parse.argtypes = [c_char_p]
        chrono.chrono_parse.restype = c_char_p
    except OSError as e:
        print(f"Error loading library: {e}")
        print(f"Make sure you have compiled the library for {platform.system()} as {lib_name}")
        sys.exit(1)

    try:
        json_str = chrono.chrono_parse(src.encode('utf-8'))
        ast = json.loads(json_str)
        print(json.dumps(ast, indent=2))
    except Exception as e:
        print(f"Error parsing ChronoLang code: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()