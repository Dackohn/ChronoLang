#include "../include/lexer.h"
#include <iostream>
#include <unordered_map>
#include <cctype>

// === Lexer Core ===
Lexer::Lexer(const std::string& input) : input(input) {}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    while (true) {
        Token token = nextToken();
        if (token.type == TokenType::INVALID) {
            invalid_tokens.push_back(token);
            continue;
        }
        tokens.push_back(token);
        if (token.type == TokenType::END_OF_FILE) break;
    }
    return tokens;
}

Token Lexer::nextToken() {
    skipWhitespace();
    if (pos >= input.size()) return makeToken(TokenType::END_OF_FILE, "");

    char c = peek();
    if (std::isalpha(c) || c == '_') return makeIdentifierOrKeyword();
    if (std::isdigit(c)) return makeNumber();
    if (c == '"') return makeString();
    return makeSymbol();
}

void Lexer::skipWhitespace() {
    while (pos < input.size() && std::isspace(input[pos])) {
        if (input[pos] == '\n') { line++; column = 1; }
        else { column++; }
        pos++;
    }
}

char Lexer::peek() const { return pos < input.size() ? input[pos] : '\0'; }
char Lexer::peekNext() const { return pos + 1 < input.size() ? input[pos + 1] : '\0'; }
char Lexer::advance() { column++; return input[pos++]; }

bool Lexer::match(char expected) {
    if (peek() == expected) {
        advance();
        return true;
    }
    return false;
}

Token Lexer::makeToken(TokenType type, const std::string& value) {
    return Token(type, value, line, column - value.length());
}

Token Lexer::makeIdentifierOrKeyword() {
    size_t start = pos;
    int startColumn = column;
    while (std::isalnum(peek()) || peek() == '_') advance();
    std::string value = input.substr(start, pos - start);

    static std::unordered_map<std::string, TokenType> keywords = {
        {"LOAD", TokenType::LOAD}, {"FROM", TokenType::FROM}, {"SET", TokenType::SET},
        {"WINDOW", TokenType::WINDOW}, {"TREND", TokenType::TREND},
        {"FORECAST", TokenType::FORECAST}, {"USING", TokenType::USING},
        {"STREAM", TokenType::STREAM}, {"SELECT", TokenType::SELECT},
        {"WHERE", TokenType::WHERE}, {"DATE", TokenType::DATE},
        {"PLOT", TokenType::PLOT}, {"EXPORT", TokenType::EXPORT},
        {"TO", TokenType::TO}, {"FOR", TokenType::FOR}, {"IN", TokenType::IN},
        {"REMOVE", TokenType::REMOVE}, {"MISSING", TokenType::MISSING},
        {"REPLACE", TokenType::REPLACE}, {"WITH", TokenType::WITH},
        {"ANALYZE", TokenType::ANALYZE}, {"BASED_ON", TokenType::BASED_ON},
        {"BELOW", TokenType::BELOW}, {"ABOVE", TokenType::ABOVE},
        {"MEAN", TokenType::MEAN}, {"MEDIAN", TokenType::MEDIAN},
        {"TENDENCY", TokenType::TENDENCY}, {"ARIMA", TokenType::ARIMA},
        {"PROPHET", TokenType::PROPHET}, {"LSTM", TokenType::LSTM},
        {"LINEPLOT", TokenType::LINEPLOT}, {"HISTOGRAM", TokenType::HISTOGRAM},
        {"SCATTERPLOT", TokenType::SCATTERPLOT}, {"BARPLOT", TokenType::BARPLOT}
    };

    std::string upper;
    for (char ch : value) upper += std::toupper(ch);

    if (keywords.count(upper)) return makeToken(keywords[upper], value);
    return makeToken(TokenType::ID, value);
}

Token Lexer::makeNumber() {
    size_t start = pos;
    int startColumn = column;
    bool isFloat = false;

    while (std::isdigit(peek())) advance();
    if (peek() == '.' && std::isdigit(peekNext())) {
        isFloat = true;
        advance(); 
        while (std::isdigit(peek())) advance();
    }

    std::string value = input.substr(start, pos - start);

    if (peek() == 'd' || peek() == 'h' || peek() == 'm') {
        value += advance();
        return makeToken(TokenType::TIME_UNIT, value);
    }

    return makeToken(isFloat ? TokenType::FLOAT : TokenType::INT, value);
}

Token Lexer::makeString() {
    advance();  
    size_t start = pos;
    int startColumn = column;

    while (peek() != '"' && pos < input.size()) {
        if (peek() == '\n') line++;
        advance();
    }

    std::string value = input.substr(start, pos - start);
    advance();  
    return makeToken(TokenType::STRING, value);
}

Token Lexer::makeSymbol() {
    char c = advance();
    switch (c) {
        case '=': 
        if (match('=')) return makeToken(TokenType::EQUAL_EQUAL, "==");
        return makeToken(TokenType::EQUAL, "=");
        case '<': 
        if (match('=')) return makeToken(TokenType::LESS_EQUAL, "<=");
        return makeToken(TokenType::LESS, "<");
        case '>': 
        if (match('=')) return makeToken(TokenType::GREATER_EQUAL, ">=");
        return makeToken(TokenType::GREATER, ">");
        case '!':
        if (match('=')) return makeToken(TokenType::NOT_EQUAL, "!=");
        return makeToken(TokenType::INVALID, "!");
        case '{': return makeToken(TokenType::LBRACE, "{");
        case '.': return makeToken(TokenType::DOT, ".");
        case '}': return makeToken(TokenType::RBRACE, "}");
        case '(': return makeToken(TokenType::LPAREN, "(");
        case ')': return makeToken(TokenType::RPAREN, ")");
        case ',': return makeToken(TokenType::COMMA, ",");
        case '[': return makeToken(TokenType::LBRACKET, "[");
        case ']': return makeToken(TokenType::RBRACKET, "]");        
        case '-':
            if (match('>')) return makeToken(TokenType::ARROW, "->");
            break;
    }
    return makeToken(TokenType::INVALID, std::string(1, c));
}

const std::vector<Token>& Lexer::getInvalidTokens() const {
    return invalid_tokens;
}

static std::string tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::LOAD: return "LOAD";
        case TokenType::FROM: return "FROM";
        case TokenType::SET: return "SET";
        case TokenType::WINDOW: return "WINDOW";
        case TokenType::TREND: return "TREND";
        case TokenType::FORECAST: return "FORECAST";
        case TokenType::USING: return "USING";
        case TokenType::STREAM: return "STREAM";
        case TokenType::SELECT: return "SELECT";
        case TokenType::WHERE: return "WHERE";
        case TokenType::DATE: return "DATE";
        case TokenType::PLOT: return "PLOT";
        case TokenType::EXPORT: return "EXPORT";
        case TokenType::TO: return "TO";
        case TokenType::FOR: return "FOR";
        case TokenType::IN: return "IN";
        case TokenType::REMOVE: return "REMOVE";
        case TokenType::MISSING: return "MISSING";
        case TokenType::REPLACE: return "REPLACE";
        case TokenType::WITH: return "WITH";
        case TokenType::ANALYZE: return "ANALYZE";
        case TokenType::BASED_ON: return "BASED_ON";
        case TokenType::BELOW: return "BELOW";
        case TokenType::ABOVE: return "ABOVE";
        case TokenType::MEAN: return "MEAN";
        case TokenType::MEDIAN: return "MEDIAN";
        case TokenType::TENDENCY: return "TENDENCY";
        case TokenType::ARIMA: return "ARIMA";
        case TokenType::PROPHET: return "PROPHET";
        case TokenType::LSTM: return "LSTM";
        case TokenType::LINEPLOT: return "LINEPLOT";
        case TokenType::HISTOGRAM: return "HISTOGRAM";
        case TokenType::SCATTERPLOT: return "SCATTERPLOT";
        case TokenType::BARPLOT: return "BARPLOT";
        case TokenType::EQUAL: return "=";
        case TokenType::ARROW: return "->";
        case TokenType::LBRACE: return "{";
        case TokenType::RBRACE: return "}";
        case TokenType::LPAREN: return "(";
        case TokenType::RPAREN: return ")";
        case TokenType::LBRACKET: return "[";
        case TokenType::RBRACKET: return "]";        
        case TokenType::COMMA: return ",";
        case TokenType::LESS: return "<";
        case TokenType::GREATER: return ">";
        case TokenType::EQUAL_EQUAL: return "==";
        case TokenType::LESS_EQUAL: return "<=";
        case TokenType::GREATER_EQUAL: return ">=";
        case TokenType::NOT_EQUAL: return "!=";
        case TokenType::ID: return "ID";
        case TokenType::STRING: return "STRING";
        case TokenType::INT: return "INT";
        case TokenType::FLOAT: return "FLOAT";
        case TokenType::TIME_UNIT: return "TIME_UNIT";
        case TokenType::END_OF_FILE: return "EOF";
        case TokenType::INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

void Lexer::runREPL() {
    std::cout << "Enter ChronoLang code (Ctrl+D to end):\n";
    std::string input, line;
    while (std::getline(std::cin, line)) input += line + '\n';

    Lexer lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    for (const auto& token : tokens) {
        std::cout << tokenTypeToString(token.type) << "('" << token.value << "')"
                  << " at line " << token.line << ", col " << token.column << "\n";
    }

    const auto& errors = lexer.getInvalidTokens();
    if (!errors.empty()) {
        std::cout << "\nInvalid Tokens Detected:\n";
        for (const auto& token : errors) {
            std::cout << "INVALID('" << token.value << "') at line "
                      << token.line << ", col " << token.column << "\n";
        }
    }
}
