#include "include/lexer.h"
#include "include/parser.h"
#include "include/ast_visualize.h"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <cstring>
#include "include/astToJson.h"
using nlohmann::json;

// Define a platform-independent export macro
#ifdef _WIN32
    #define CHRONOLANG_EXPORT __declspec(dllexport)
#else
    #define CHRONOLANG_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

CHRONOLANG_EXPORT const char* chrono_parse(const char* input) {
    // Using a static variable is safer than return result.c_str()
    // but we need to make a copy of the result for thread safety
    static thread_local std::string result_storage;

    try {
        Lexer lexer(input);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        json j = astToJson(ast.get());
        result_storage = j.dump();  // serialize JSON to string
    } catch (const std::exception& e) {
        result_storage = std::string("{\"error\": \"") + e.what() + "\"}";
    }

    // Allocate persistent memory for the result
    // This avoids the string going out of scope issue
    char* result = new char[result_storage.length() + 1];
    std::strcpy(result, result_storage.c_str());

    return result;
}

} // extern "C"