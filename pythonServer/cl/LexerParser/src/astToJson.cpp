#include "../include/astToJson.h"

json astToJson(const ASTNode* node) {
    switch (node->type) {
        case ASTNodeType::Program: {
            auto* n = dynamic_cast<const ProgramNode*>(node);
            json j = { {"type", "Program"}, {"statements", json::array()} };
            for (const auto& stmt : n->statements)
                j["statements"].push_back(astToJson(stmt.get()));
            return j;
        }
        case ASTNodeType::Load: {
            auto* n = dynamic_cast<const LoadStmtNode*>(node);
            json j = { {"type", "Load"}, {"id", n->id}, {"path", n->path} };
            if (n->alias.has_value())
                j["alias"] = *n->alias;
            return j;
        }

        case ASTNodeType::Set: {
            auto* n = dynamic_cast<const SetStmtNode*>(node);
            return { {"type", "Set"}, {"amount", n->amount}, {"unit", n->unit} };
        }
        case ASTNodeType::Transform: {
            auto* n = dynamic_cast<const TransformStmtNode*>(node);
            json j = { {"type", "Transform"} };
            if (n->ref.isVariable) {
                j["variable"] = n->ref.variableName;
            } else {
                j["table"] = n->ref.table;
                if (n->ref.column.has_value())
                    j["column"] = *n->ref.column;
            }
            j["interval"] = {
                {"amount", n->intervalAmount},
                {"unit", n->intervalUnit}
            };
            if (n->alias.has_value())
                j["alias"] = *n->alias;
            return j;
        }
       
        case ASTNodeType::Forecast: {
            auto* n = dynamic_cast<const ForecastStmtNode*>(node);
            json paramsJson;
            for (const auto& p : n->params)
                paramsJson[p.first] = p.second;

            json j = {
                {"type", "Forecast"},
                {"model", n->model},
                {"params", paramsJson}
            };
            if (n->ref.isVariable) {
                j["variable"] = n->ref.variableName;
            } else {
                j["table"] = n->ref.table;
                if (n->ref.column.has_value())
                    j["column"] = *n->ref.column;
            }
            if (n->alias.has_value())
                j["alias"] = *n->alias;
            return j;
        }


        
        case ASTNodeType::Stream: {
            auto* n = dynamic_cast<const StreamStmtNode*>(node);
            return { {"type", "Stream"}, {"id", n->id}, {"path", n->path} };
        }
        case ASTNodeType::Select: {
            auto* n = dynamic_cast<const SelectStmtNode*>(node);
            json j = { {"type", "Select"} };
            if (n->ref.isVariable) {
                j["variable"] = n->ref.variableName;
            } else {
                j["table"] = n->ref.table;
                if (n->ref.column.has_value())
                    j["column"] = *n->ref.column;
            }
            if (n->op && n->dateExpr) {
                j["condition"] = { {"op", *n->op}, {"date", *n->dateExpr} };
            }
            if (n->alias.has_value())
                j["alias"] = *n->alias;
            return j;
        }


        case ASTNodeType::Plot: {
            auto* n = dynamic_cast<const PlotStmtNode*>(node);
            json argsJson = json::object();
            for (auto& arg : n->args)
                argsJson[arg.first] = arg.second;
            return { {"type", "Plot"}, {"function", n->function}, {"args", argsJson} };
        }
        case ASTNodeType::Export: {
            auto* n = dynamic_cast<const ExportStmtNode*>(node);
            json j = {
                {"type", "Export"},
                {"to", n->target}
            };
            if (n->ref.isVariable) {
                j["variable"] = n->ref.variableName;
            } else {
                j["table"] = n->ref.table;
                if (n->ref.column.has_value())
                    j["column"] = *n->ref.column;
            }
            return j;
        }

        
        case ASTNodeType::Loop: {
            auto* n = dynamic_cast<const LoopStmtNode*>(node);
            json body = json::array();
            for (const auto& stmt : n->body)
                body.push_back(astToJson(stmt.get()));
            return {
                {"type", "Loop"},
                {"var", n->var},
                {"from", n->from},
                {"to", n->to},
                {"body", body}
            };
        }
        case ASTNodeType::Clean: {
            auto* n = dynamic_cast<const CleanStmtNode*>(node);
            if (n->action == CleanActionType::Remove) {
                return {
                    {"type", "Clean"},
                    {"action", "remove"},
                    {"target", n->targetValue},
                    {"column", n->column}
                };
            } else {
                return {
                    {"type", "Clean"},
                    {"action", "replace"},
                    {"target", n->targetValue},
                    {"column", n->column},
                    {"with", n->replaceWith}
                };
            }
        }
        default:
            return { {"type", "Unknown"} };
    }
}
