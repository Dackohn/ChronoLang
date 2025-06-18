#include "src/lexer.cpp"
#include "src/parser.cpp"
#include "src/ast_visualize_simplified.cpp"
//#include "src/ast_visualize.cpp"

int main() {
    std::vector<std::pair<std::string, std::string>> snippets = {
        {R"chrono(SELECT sales.amount WHERE date > "2024-01-01" AS filtered
        SELECT $filtered WHERE date > "2024-02-01" AS more_filtered
        FORECAST $filtered USING ARIMA(param=1) AS predicted
        TREND($filtered) -> forecast_next(12m) AS trend_line
        EXPORT $predicted TO "pred.csv")chrono", "ast11.dot"},
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

            Parser parser(tokens);
            auto ast = parser.parse();
            drawParseTreeSimplified(ast, filename);

        } catch (const std::exception& e) {
            std::cerr << "Error in snippet " << i + 1 << ": " << e.what() << "\n";
        }
    }

    return 0;
}
