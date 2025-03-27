from ctypes import cdll, c_char_p
import json
import os
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))
chrono = cdll.LoadLibrary(dll_path)
chrono.chrono_parse.argtypes = [c_char_p]
chrono.chrono_parse.restype = c_char_p

src = '''TREND(sales_data.sales_amount) -> forecast_next(7d)
FORECAST sales_data.sales_amount USING ARIMA(model_order=2, seasonal_order=1)
EXPORT sales_data.sales_amount TO "results/sales_amount.csv"
'''

src2 = '''LOAD sales_data FROM "data/sales.csv"
plot From 2 For 3'''
json_str = chrono.chrono_parse(src.encode('utf-8'))
ast = json.loads(json_str)

print(json.dumps(ast, indent=2))
