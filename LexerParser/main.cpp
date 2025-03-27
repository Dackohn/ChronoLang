#include "src/lexer.cpp"
#include "src/parser.cpp"
//#include "src/ast_visualize_simplified.cpp"
#include "src/ast_visualize.cpp"
int main() {
    std::string code = R"(
        LOAD sales_data FROM "data/sales.csv"
STREAM live_data FROM "http://api.example.com/stream"

SET WINDOW = 30d

TREND(sales_amount) -> forecast_next(7d)
FORECAST sales_amount USING ARIMA(model_order=2, seasonal_order=1)

SELECT sales_amount WHERE DATE > "2024-01-01"

PLOT LINEPLOT(
    data=[[100, 200, 150], [120, 220, 170]],
    x_label="Days",
    y_label="Sales",
    title="Weekly Sales",
    legend=["Week 1", "Week 2"]
)

FOR i IN 1 TO 3 {
    FORECAST sales_amount USING Prophet(model_order=3, seasonal_order=2)
    EXPORT "run_${i}" TO "results/run_${i}.csv"
}
    )";

    Lexer lexer(code);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    //drawParseTreeSimplified(ast, "astSimplified.dot");
    drawParseTree(ast, "ast.dot");
    return 0;
}
