#include <iostream>

extern "C" const char* chrono_parse(const char* input);

int main() {
    const char* result = chrono_parse("LOAD data FROM 'x.csv' as my_var\n"
                                      "SET WINDOW = 7d\n"
                                      "TREND(my_var.amount) -> forecast_next(14d)\n"
                                      "FORECAST $my_var USING ARIMA(model_order=2)\n"
                                     );
    std::cout << result << std::endl;
    return 0;
}
