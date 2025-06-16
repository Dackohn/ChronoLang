#include "src/lexer.cpp"
#include "src/parser.cpp"
//#include "src/ast_visualize_simplified.cpp"
#include "src/ast_visualize.cpp"
int main() {
    std::string code = R"(
        TREND(sales_data.sales_amount) -> forecast_next(7d)
        FORECAST sales_data.sales_amount USING ARIMA(model_order=2, seasonal_order=1)
        EXPORT sales_data.sales_amount TO "results/sales_amount.csv"
    )";

    Lexer lexer(code);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    //drawParseTreeSimplified(ast, "astSimplified.dot");
    drawParseTree(ast, "ast.dot");
    return 0;
}
