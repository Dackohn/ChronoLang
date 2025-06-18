#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>

enum class ASTNodeType {
    Program,
    Load, Set, Transform, Forecast, Stream, Select, Plot, Export, Loop, Clean,
    Expression, Value,
};

enum class CleanActionType {
    Remove,
    Replace
};


struct ASTNode {
    ASTNodeType type;
    int line;
    int column;
    ASTNode(ASTNodeType type, int line, int column)
        : type(type), line(line), column(column) {}
    virtual ~ASTNode() = default;
};

using ASTNodePtr = std::unique_ptr<ASTNode>;

struct ProgramNode : public ASTNode {
    std::vector<ASTNodePtr> statements;
    ProgramNode() : ASTNode(ASTNodeType::Program, 0, 0) {}
};

struct LoadStmtNode : public ASTNode {
    std::string id;
    std::string path;
    std::optional<std::string> alias; 

    LoadStmtNode(const std::string& id, const std::string& path,
                 std::optional<std::string> alias,
                 int line, int col)
        : ASTNode(ASTNodeType::Load, line, col),
          id(id), path(path), alias(alias) {}
};



struct Reference {
    bool isVariable;                            
    std::string table;                         
    std::optional<std::string> column;         
    std::string variableName;                  
};


struct SetStmtNode : public ASTNode {
    int amount;
    std::string unit;
    SetStmtNode(int amount, const std::string& unit, int line, int col)
        : ASTNode(ASTNodeType::Set, line, col), amount(amount), unit(unit) {}
};

struct TransformStmtNode : public ASTNode {
    Reference ref;
    int intervalAmount;
    std::string intervalUnit;
    std::optional<std::string> alias;  // NEW

    TransformStmtNode(const Reference& ref,
                      int amt, const std::string& unit,
                      std::optional<std::string> alias, // NEW
                      int line, int col)
        : ASTNode(ASTNodeType::Transform, line, col),
          ref(ref), intervalAmount(amt), intervalUnit(unit), alias(alias) {}
};



struct ForecastStmtNode : public ASTNode {
    Reference ref;
    std::string model;
    std::vector<std::pair<std::string, int>> params;
    std::optional<std::string> alias;  // NEW

    ForecastStmtNode(const Reference& ref,
                     const std::string& model,
                     const std::vector<std::pair<std::string, int>>& params,
                     std::optional<std::string> alias, // NEW
                     int line, int col)
        : ASTNode(ASTNodeType::Forecast, line, col),
          ref(ref), model(model), params(params), alias(alias) {}
};


    

struct StreamStmtNode : public ASTNode {
    std::string id;
    std::string path;
    StreamStmtNode(const std::string& id, const std::string& path, int line, int col)
        : ASTNode(ASTNodeType::Stream, line, col), id(id), path(path) {}
};

struct SelectStmtNode : public ASTNode {
    Reference ref;
    std::optional<std::string> op;
    std::optional<std::string> dateExpr;
    std::optional<std::string> alias; // NEW

    SelectStmtNode(const Reference& ref,
                   std::optional<std::string> op,
                   std::optional<std::string> dateExpr,
                   std::optional<std::string> alias, // NEW
                   int line, int col)
        : ASTNode(ASTNodeType::Select, line, col),
          ref(ref), op(op), dateExpr(dateExpr), alias(alias) {}
};




struct PlotStmtNode : public ASTNode {
    std::string function;
    std::vector<std::pair<std::string, std::string>> args;
    PlotStmtNode(const std::string& fn,
        const std::vector<std::pair<std::string, std::string>>& args,
        int line, int col): ASTNode(ASTNodeType::Plot, line, col), function(fn), args(args) {}
};

struct ExportStmtNode : public ASTNode {
    Reference ref;
    std::string target;

    ExportStmtNode(const Reference& ref,
                   const std::string& target,
                   int line, int col)
        : ASTNode(ASTNodeType::Export, line, col),
          ref(ref), target(target) {}
};



struct LoopStmtNode : public ASTNode {
    std::string var;
    int from, to;
    std::vector<ASTNodePtr> body;
    LoopStmtNode(const std::string& var,
        int from, int to,
        std::vector<ASTNodePtr> body,
        int line, int col): ASTNode(ASTNodeType::Loop, line, col), var(var), from(from), to(to), body(std::move(body)) {}

};

struct CleanStmtNode : public ASTNode {
    CleanActionType action;
    std::string targetValue;  
    std::string column;       
    std::string replaceWith;  

    // In ast.h, CleanStmtNode
CleanStmtNode(CleanActionType action,
              const std::string& targetValue,
              const std::string& column,
              const std::string& replaceWith,
              int line, int col)
                : ASTNode(ASTNodeType::Clean, line, col),
          action(action),
          targetValue(targetValue),
          column(column),
          replaceWith(replaceWith) {}
};
