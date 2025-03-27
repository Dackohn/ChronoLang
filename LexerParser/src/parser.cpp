#include "../include/parser.h"
#include <stdexcept>
#include <cstdlib>
#include <sstream>

// === Core Parsing===

Parser::Parser(const std::vector<Token>& tokens) : tokens(tokens) {}

const Token& Parser::peek() const { return tokens[current]; }
const Token& Parser::previous() const { return tokens[current - 1]; }
bool Parser::check(TokenType type) const { return !isAtEnd() && peek().type == type; }
bool Parser::isAtEnd() const { return peek().type == TokenType::END_OF_FILE; }
const Token& Parser::advance() { if (!isAtEnd()) current++; return previous(); }

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

void Parser::expect(TokenType type, const std::string& errorMessage) {
    if (!match(type)) {
        throw std::runtime_error("Expected " + errorMessage + " at line " +
                                 std::to_string(peek().line) + ", column " +
                                 std::to_string(peek().column));
    }
}

std::string Parser::parseColumn() {
    expect(TokenType::ID, "table or column name");
    std::string name = previous().value;

    if (match(TokenType::DOT)) {
        expect(TokenType::ID, "column name after '.'");
        name += "." + previous().value;
    }

    return name;
}



std::string Parser::parseValue() {
    if (check(TokenType::STRING) || check(TokenType::INT) || check(TokenType::FLOAT)) {
        return advance().value;
    }
    throw std::runtime_error("Expected a value (STRING, INT, or FLOAT)");
}


std::unique_ptr<ProgramNode> Parser::parse() {
    auto program = std::make_unique<ProgramNode>();
    while (!isAtEnd()) {
        program->statements.push_back(parseStatement());
    }
    return program;
}

std::pair<std::string, std::optional<std::string>> Parser::parseTableAndColumn() {
    expect(TokenType::ID, "table name");
    std::string table = previous().value;
    std::optional<std::string> column;

    if (match(TokenType::DOT)) {
        expect(TokenType::ID, "column name after '.'");
        column = previous().value;
    }

    return { table, column };
}


// === Main Statement Dispatcher ===

ASTNodePtr Parser::parseStatement() {
    if (match(TokenType::LOAD)) return parseLoadStatement();
    if (match(TokenType::SET)) return parseSetStatement();
    if (match(TokenType::TREND)) return parseTransformStatement();
    if (match(TokenType::FORECAST)) return parseForecastStatement();
    if (match(TokenType::STREAM)) return parseStreamStatement();
    if (match(TokenType::SELECT)) return parseSelectStatement();
    if (match(TokenType::PLOT)) return parsePlotStatement();
    if (match(TokenType::EXPORT)) return parseExportStatement();
    if (match(TokenType::FOR)) return parseLoopStatement();
    if (match(TokenType::REMOVE) || match(TokenType::REPLACE) || match(TokenType::ANALYZE))
        return parseCleanStatement();

    throw std::runtime_error("Unexpected token at line " + std::to_string(peek().line));
}

ASTNodePtr Parser::parseLoadStatement() {
    Token id = advance();
    expect(TokenType::FROM, "'FROM'");
    Token path = advance();
    return std::make_unique<LoadStmtNode>(id.value, path.value, id.line, id.column);
}

ASTNodePtr Parser::parseSetStatement() {
    expect(TokenType::WINDOW, "'WINDOW'");
    expect(TokenType::EQUAL, "'='");
    auto [amount, unit] = parseTimeInterval();
    return std::make_unique<SetStmtNode>(amount, unit, peek().line, peek().column);
}

ASTNodePtr Parser::parseTransformStatement() {
    expect(TokenType::LPAREN, "'('");
    auto [table, column] = parseTableAndColumn();
    expect(TokenType::RPAREN, "')'");
    expect(TokenType::ARROW, "'->'");
    expect(TokenType::ID, "'forecast_next'");
    expect(TokenType::LPAREN, "'('");
    auto [amount, unit] = parseTimeInterval();
    expect(TokenType::RPAREN, "')'");

    if (!column.has_value())
        throw std::runtime_error("TREND requires a table.column reference");

    return std::make_unique<TransformStmtNode>(
        table, *column, amount, unit, peek().line, peek().column
    );
}

ASTNodePtr Parser::parseForecastStatement() {
    auto [table, column] = parseTableAndColumn();
    expect(TokenType::USING, "'USING'");
    Token model = advance();
    expect(TokenType::LPAREN, "'('");
    auto params = parseParams();
    expect(TokenType::RPAREN, "')'");

    if (!column.has_value())
        throw std::runtime_error("FORECAST requires a table.column reference");

    return std::make_unique<ForecastStmtNode>(
        table, *column, model.value, params, model.line, model.column
    );
}

ASTNodePtr Parser::parseStreamStatement() {
    Token id = advance();
    expect(TokenType::FROM, "'FROM'");
    Token path = advance();
    return std::make_unique<StreamStmtNode>(id.value, path.value, id.line, id.column);
}

ASTNodePtr Parser::parseSelectStatement() {
    auto [table, column] = parseTableAndColumn();
    std::optional<std::string> op, date;
    if (match(TokenType::WHERE)) {
        expect(TokenType::DATE, "'DATE'");
        Token oper = advance();
        Token expr = advance();
        op = oper.value;
        date = expr.value;
    }

    if (!column.has_value())
        throw std::runtime_error("SELECT requires a table.column reference");

    return std::make_unique<SelectStmtNode>(
        table, *column, op, date, peek().line, peek().column
    );
}

ASTNodePtr Parser::parsePlotStatement() {
    Token plotType = advance();
    expect(TokenType::LPAREN, "'('");

    std::vector<std::pair<std::string, std::string>> args;

    while (!check(TokenType::RPAREN)) {
        expect(TokenType::ID, "parameter key");
        std::string key = previous().value;
        expect(TokenType::EQUAL, "'='");

        std::string value;
        if (check(TokenType::STRING) || check(TokenType::INT) || check(TokenType::FLOAT)) {
            value = advance().value;
        } else if (check(TokenType::LBRACKET)) {
            int bracketCount = 0;
            std::ostringstream buffer;

            while (!isAtEnd()) {
                Token t = advance();

                if (t.type == TokenType::LBRACKET) bracketCount++;
                if (t.type == TokenType::RBRACKET) bracketCount--;

                buffer << t.value;

                if (bracketCount == 0) break;
            }

            if (bracketCount != 0) {
                throw std::runtime_error("Mismatched brackets in plot parameter value at line " + std::to_string(peek().line));
            }

            value = buffer.str();
        } else {
            throw std::runtime_error("Unexpected plot parameter value at line " + std::to_string(peek().line));
        }

        args.emplace_back(key, value);

        if (!check(TokenType::RPAREN)) {
            expect(TokenType::COMMA, "',' or ')'");
        }
    }

    expect(TokenType::RPAREN, "')'");

    return std::make_unique<PlotStmtNode>(plotType.value, args, plotType.line, plotType.column);
}


ASTNodePtr Parser::parseExportStatement() {
    expect(TokenType::ID, "table or column name");
    std::string table = previous().value;
    std::optional<std::string> column;

    if (match(TokenType::DOT)) {
        expect(TokenType::ID, "column name after '.'");
        column = previous().value;
    }

    expect(TokenType::TO, "'TO'");
    Token target = advance();

    return std::make_unique<ExportStmtNode>(table, column, target.value, table.length(), target.column);
}



ASTNodePtr Parser::parseLoopStatement() {
    expect(TokenType::ID, "loop variable");
    Token loopToken = previous();
    std::string loopVar = loopToken.value;

    expect(TokenType::IN, "'IN'");
    Token start = advance();
    expect(TokenType::TO, "'TO'");
    Token end = advance();
    expect(TokenType::LBRACE, "'{'");

    std::vector<ASTNodePtr> body;
    while (!check(TokenType::RBRACE) && !isAtEnd()) {
        body.push_back(parseStatement());
    }

    expect(TokenType::RBRACE, "'}'");

    return std::make_unique<LoopStmtNode>(
        loopVar,
        std::stoi(start.value),
        std::stoi(end.value),
        std::move(body),
        loopToken.line,
        loopToken.column
    );
}


ASTNodePtr Parser::parseCleanStatement() {
    Token first = previous();
    bool isRemove = first.type == TokenType::REMOVE;

    Token target = advance();

    if (isRemove) {
        expect(TokenType::FROM, "'FROM'");
        std::string column = parseColumn();
        return std::make_unique<CleanStmtNode>(
            CleanActionType::Remove,
            target.value,
            column,
            "",
            first.line,
            first.column
        );
    } else { 
        expect(TokenType::IN, "'IN'");
        std::string column = parseColumn();
        expect(TokenType::WITH, "'WITH'");
        std::string replacement = parseValue();
        return std::make_unique<CleanStmtNode>(
            CleanActionType::Replace,
            target.value,
            column,
            replacement,
            first.line,
            first.column
        );
    }
}


std::pair<int, std::string> Parser::parseTimeInterval() {
    expect(TokenType::TIME_UNIT, "time interval (e.g., 30d, 5h, 10m)");

    std::string value = previous().value;

    size_t i = 0;
    while (i < value.size() && std::isdigit(value[i])) ++i;

    if (i == 0 || i >= value.size()) {
        throw std::runtime_error("Invalid time interval format at line " +
            std::to_string(previous().line));
    }

    int amount = std::stoi(value.substr(0, i));
    std::string unit = value.substr(i);

    return { amount, unit };

}

std::pair<std::string, std::string> Parser::parseIDEqualsValue() {
    expect(TokenType::ID, "parameter key");
    std::string key = previous().value;

    expect(TokenType::EQUAL, "'='");

    std::string value = parseValue();

    return { key, value };
}

std::vector<std::pair<std::string, int>> Parser::parseParams() {
    std::vector<std::pair<std::string, int>> params;

    if (check(TokenType::ID)) {
        auto [key, val] = parseIDEqualsValue();
        params.emplace_back(key, std::stoi(val));

        while (match(TokenType::COMMA)) {
            auto [k, v] = parseIDEqualsValue();
            params.emplace_back(k, std::stoi(v));
        }
    }

    return params;
}

std::vector<std::vector<double>> Parser::parseListOfLists() {
    std::vector<std::vector<double>> outer;

    expect(TokenType::LBRACKET, "'['");

    while (!check(TokenType::RBRACKET)) {
        outer.push_back(parseListOfNumbers());

        if (!check(TokenType::RBRACKET)) {
            expect(TokenType::COMMA, "','");
        }
    }

    expect(TokenType::RBRACKET, "closing ']'");
    return outer;
}

std::vector<double> Parser::parseListOfNumbers() {
    std::vector<double> values;

    expect(TokenType::LBRACKET, "'['");

    while (!check(TokenType::RBRACKET)) {
        if (check(TokenType::INT)) {
            values.push_back(std::stoi(advance().value));
        } else if (check(TokenType::FLOAT)) {
            values.push_back(std::stod(advance().value));
        } else {
            throw std::runtime_error("Expected number inside list at line " + std::to_string(peek().line));
        }

        if (!check(TokenType::RBRACKET)) {
            expect(TokenType::COMMA, "','");
        }
    }

    expect(TokenType::RBRACKET, "closing ']'");
    return values;
}

std::vector<std::string> Parser::parseListOfStrings() {
    std::vector<std::string> values;

    expect(TokenType::LBRACKET, "'['");

    while (!check(TokenType::RBRACKET)) {
        expect(TokenType::STRING, "string literal");
        values.push_back(previous().value);

        if (!check(TokenType::RBRACKET)) {
            expect(TokenType::COMMA, "','");
        }
    }

    expect(TokenType::RBRACKET, "closing ']'");
    return values;
}
