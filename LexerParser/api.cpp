#include "include/lexer.h"
#include "include/parser.h"
#include "include/ast_visualize.h"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

extern "C" {

__declspec(dllexport) const char* chrono_parse(const char* input) {
    static std::string result;
    Lexer lexer(input);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);

    try {
        auto ast = parser.parse();
        drawParseTree(ast, "ast.dot");
        result = "Parsing successful!";
    } catch (const std::exception& e) {
        result = std::string("Parsing error: ") + e.what();
    }

    return result.c_str();
}

}
