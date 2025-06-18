from ctypes import cdll, c_char_p
import json
import os
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))
chrono = cdll.LoadLibrary(dll_path)
chrono.chrono_parse.argtypes = [c_char_p]
chrono.chrono_parse.restype = c_char_p

src = '''PLOT LINEPLOT(
    data=[$high_after_4],
    x_label="Index",
    y_label="High",
    title="High Price After 2019-01-04"
)

'''


src2 = '''LOAD sales_data FROM "data/sales.csv"
plot From 2 For 3'''
json_str = chrono.chrono_parse(src.encode('utf-8'))
ast = json.loads(json_str)

print(json.dumps(ast, indent=2))
