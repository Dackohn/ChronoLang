#include "src/lexer.cpp"
#include "src/parser.cpp"
//#include "src/ast_visualize_simplified.cpp"
//#include "src/ast_visualize.cpp"

int main() {
    std::vector<std::pair<std::string, std::string>> snippets = {
        {R"(LOAD sales FROM "data.csv"
            SET WINDOW = 7d)", "ast1.dot"},

        {R"(TREND(sales.amount) -> forecast_next(14d))", "ast2.dot"},

        {R"(FORECAST sales.amount USING ARIMA(model_order=2))", "ast3.dot"},

        {R"(STREAM live FROM "http://stream.io/data")", "ast4.dot"},

        {R"(SELECT sales.amount WHERE DATE > "2023-01-01" as filtered)", "ast5.dot"},

        {R"(PLOT LINEPLOT(x_label="Day", y_label="Value"))", "ast6.dot"},

        {R"(EXPORT sales.amount TO "result.csv")", "ast7.dot"},

        {R"(FOR i IN 1 TO 3 {
                EXPORT sales TO "out_${i}.csv"
            })", "ast8.dot"},

        {R"(REMOVE MISSING FROM sales.amount
            REPLACE MISSING IN sales.amount WITH 0)", "ast9.dot"},
            
        {R"(For i in 1 to 3 {
            TREND(sales_data.sales_amount) -> forecast_next(7d)
        FORECAST sales_data.sales_amount USING ARIMA(model_order=2, seasonal_order=1)
        EXPORT sales_data.sales_amount TO "results/sales_amount.csv"})", "ast10.dot"},

        {R"chrono(LOAD sales_data FROM "InterpreterAmazon.csv"

            TREND(sales_data.Open) -> forecast_next(7d)
            FORECAST sales_data.Open USING ARIMA(model_order=2, seasonal_order=1)
            
            SELECT sales_data.ales_amount WHERE DATE > "2024-1-01"
            
            PLOT LINEPLOT(
                data=[[100, 200, 150], [120, 220, 170]],
                x_label="Days",
                y_label="Sales",
                title="Weekly Sales",
                legend=["Week 1", "Week 2"]
            )
            
            FOR i IN 1 TO 3 {
                FORECAST sales_data.Open USING Prophet(model_order=3, seasonal_order=2)
                EXPORT sales_data.Open TO "results/run_${i}.csv"
            })chrono", "ast11.dot"},
    };

    for (size_t i = 0; i < snippets.size(); ++i) {
        const auto& [code, filename] = snippets[i];

        try {
            Lexer lexer(code);
            auto tokens = lexer.tokenize();
            for (const auto& token : tokens) {
                std::cout << "Type: " << tokenTypeToString(token.type) << ", Value: " << token.value << std::endl;
            }
            std::cout << std::endl;
            //Parser parser(tokens);
            //auto ast = parser.parse();
            //drawParseTreeSimplified(ast, filename);
        } catch (const std::exception& e) {
            std::cerr << "Error in snippet " << i + 1 << ": " << e.what() << "\n";
        }
    }

    return 0;
}
