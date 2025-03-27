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
            return { {"type", "Load"}, {"id", n->id}, {"path", n->path} };
        }
        case ASTNodeType::Set: {
            auto* n = dynamic_cast<const SetStmtNode*>(node);
            return { {"type", "Set"}, {"amount", n->amount}, {"unit", n->unit} };
        }
        case ASTNodeType::Transform: {
            auto* n = dynamic_cast<const TransformStmtNode*>(node);
            return {
                {"type", "Transform"},
                {"column", n->column},
                {"interval", {{"amount", n->intervalAmount}, {"unit", n->intervalUnit}}}
            };
        }
        case ASTNodeType::Forecast: {
            auto* n = dynamic_cast<const ForecastStmtNode*>(node);
            json paramsJson = json::object();
            for (auto& p : n->params)
                paramsJson[p.first] = p.second;
            return {
                {"type", "Forecast"},
                {"column", n->column},
                {"model", n->model},
                {"params", paramsJson}
            };
        }
        case ASTNodeType::Stream: {
            auto* n = dynamic_cast<const StreamStmtNode*>(node);
            return { {"type", "Stream"}, {"id", n->id}, {"path", n->path} };
        }
        case ASTNodeType::Select: {
            auto* n = dynamic_cast<const SelectStmtNode*>(node);
            return {
                {"type", "Select"},
                {"column", n->column},
                {"condition", (n->op && n->dateExpr) ? json{
                    {"op", *n->op}, {"date", *n->dateExpr}
                } : nullptr}
            };
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
            return { {"type", "Export"}, {"from", n->source}, {"to", n->target} };
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
