#pragma once
#include "token.h"
#include <string>
#include <vector>

class Lexer {
public:
    explicit Lexer(const std::string& input);
    std::vector<Token> tokenize();
    const std::vector<Token>& getInvalidTokens() const;
    static void runREPL();

private:
    std::vector<Token> invalid_tokens;
    std::string input;
    size_t pos = 0;
    int line = 1;
    int column = 1;

    Token nextToken();
    void skipWhitespace();
    char peek() const;
    char peekNext() const;
    char advance();
    bool match(char expected);

    Token makeString();
    Token makeNumber();
    Token makeIdentifierOrKeyword();
    Token makeSymbol();
    Token makeToken(TokenType type, const std::string& value);
};
