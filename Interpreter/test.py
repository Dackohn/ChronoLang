from ctypes import cdll, c_char_p
import json
import os
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))
chrono = cdll.LoadLibrary(dll_path)
chrono.chrono_parse.argtypes = [c_char_p]
chrono.chrono_parse.restype = c_char_p

src = '''SELECT sales.amount WHERE date > "2024-01-01" AS filtered
        SELECT $filtered WHERE date > "2024-02-01" AS more_filtered
        FORECAST $filtered USING ARIMA(param=1) AS predicted
        TREND($filtered) -> forecast_next(12m) AS trend_line
        EXPORT $predicted TO "pred.csv"
'''

src2 = '''LOAD sales_data FROM "data/sales.csv"
plot From 2 For 3'''
json_str = chrono.chrono_parse(src.encode('utf-8'))
ast = json.loads(json_str)

print(json.dumps(ast, indent=2))
