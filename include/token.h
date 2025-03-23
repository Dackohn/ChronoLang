#pragma once
#include <string>
//Token types and their distribution
enum class TokenType {
    // Keywords
    LOAD, FROM, SET, WINDOW, TREND, FORECAST, USING, STREAM, SELECT, WHERE, DATE,
    PLOT, EXPORT, TO, FOR, IN, REMOVE, MISSING, REPLACE, WITH, ANALYZE,
    BASED_ON, BELOW, ABOVE, MEAN, MEDIAN, TENDENCY, ARIMA, PROPHET, LSTM,

    // Symbols
    EQUAL, ARROW, LBRACE, RBRACE, LPAREN, RPAREN, COMMA, LESS, GREATER,

    // Literals
    ID, STRING, INT, FLOAT, TIME_UNIT,

    // Special
    END_OF_FILE, INVALID
};

struct Token {
    TokenType type;
    std::string value;
    int line;
    int column;

    Token(TokenType type, const std::string& value, int line, int column)
        : type(type), value(value), line(line), column(column) {}
};
