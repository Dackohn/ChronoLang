from ctypes import cdll, c_char_p
import json
import os
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chronolang.dll'))
chrono = cdll.LoadLibrary(dll_path)
chrono.chrono_parse.argtypes = [c_char_p]
chrono.chrono_parse.restype = c_char_p

<<<<<<< HEAD
src = '''SELECT sales.amount WHERE date > "2024-01-01" AS filtered
        SELECT $filtered WHERE date > "2024-02-01" AS more_filtered
        FORECAST $filtered USING ARIMA(param=1) AS predicted
        TREND($filtered) -> forecast_next(12m) AS trend_line
        EXPORT $predicted TO "pred.csv"
=======
src = '''LOAD sales_data FROM "Interpreter\Amazon.csv"

        TREND(sales_data.Open) -> forecast_next(7d)
        FORECAST sales_data.Open USING ARIMA(model_order=2, seasonal_order=1)

        SELECT sales_data.ales_amount WHERE DATE > "2024-01-01"

        PLOT LINEPLOT(
            data=[[100, 200, 150], [120, 220, 170]],
            x_label="Days",
            y_label="Sales",
            title="Weekly Sales",
            legend=["Week 1", "Week 2"]
        )

        FOR i IN 1 TO 3 {
            FORECAST sales_data.Open USING Prophet(model_order=3, seasonal_order=2)
            EXPORT sales_data.Open TO "results/Open.csv"
        }
>>>>>>> 691a1ac0ede7274ab519db7ef5a8ad374c802fd1
'''

src2 = '''LOAD sales_data FROM "data/sales.csv"
plot From 2 For 3'''
json_str = chrono.chrono_parse(src.encode('utf-8'))
ast = json.loads(json_str)

print(json.dumps(ast, indent=2))
