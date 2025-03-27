#include "include/lexer.h"
#include "include/parser.h"
#include "include/ast_visualize.h"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include "include/astToJson.h"
using nlohmann::json;


extern "C" {

__declspec(dllexport) const char* chrono_parse(const char* input) {
    static std::string result;
    Lexer lexer(input);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);

    try {
        auto ast = parser.parse();
        json j = astToJson(ast.get());
        result = j.dump();  // serialize JSON to string
    } catch (const std::exception& e) {
        result = std::string("{\"error\": \"") + e.what() + "\"}";
    }

    return result.c_str();
}
}
