#pragma once
#include "token.h"
#include "ast.h"
#include <vector>
#include <memory>
#include <string>

class Parser {
public:
    explicit Parser(const std::vector<Token>& tokens);
    std::unique_ptr<ProgramNode> parse();

private:
    const std::vector<Token>& tokens;
    size_t current = 0;
    Reference parseReference();
    std::optional<std::string> parseOptionalAlias();


    const Token& peek() const;
    const Token& previous() const;
    bool match(TokenType type);
    bool check(TokenType type) const;
    const Token& advance();
    bool isAtEnd() const;
    void expect(TokenType type, const std::string& errorMessage);
    void synchronize();
    std::pair<std::string, std::optional<std::string>> parseTableAndColumn();
    ASTNodePtr parseStatement();

    ASTNodePtr parseLoadStatement();
    ASTNodePtr parseSetStatement();
    ASTNodePtr parseTransformStatement();
    ASTNodePtr parseForecastStatement();
    ASTNodePtr parseStreamStatement();
    ASTNodePtr parseSelectStatement();
    ASTNodePtr parsePlotStatement();
    ASTNodePtr parseExportStatement();
    ASTNodePtr parseLoopStatement();
    ASTNodePtr parseCleanStatement();

    std::pair<int, std::string> parseTimeInterval();
    std::pair<std::string, std::string> parseIDEqualsValue();
    std::vector<std::pair<std::string, int>> parseParams();
    std::vector<std::vector<double>> parseListOfLists();
    std::vector<double> parseListOfNumbers();
    std::vector<std::string> parseListOfStrings();
    std::string parseColumn();
    std::string parseValue();
};
