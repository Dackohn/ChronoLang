#pragma once
#include "../include/ast.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

// Utility to escape labels for DOT
inline std::string escapeLabel(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

inline void drawNode(std::ostream& out, int id, const std::string& label) {
    out << "  node" << id << " [label=\"" << escapeLabel(label) << "\", shape=box, style=filled, fillcolor=lightyellow];\n";
}

inline int drawASTNodeSimplified(const ASTNode* node, std::ostream& out, int& nextId) {
    int thisId = nextId++;
    std::ostringstream label;

    switch (node->type) {
        case ASTNodeType::Program: {
            label << "Program";
            drawNode(out, thisId, label.str());
            const auto* program = dynamic_cast<const ProgramNode*>(node);
            for (const auto& stmt : program->statements) {
                int childId = drawASTNodeSimplified(stmt.get(), out, nextId);
                out << "  node" << thisId << " -> node" << childId << ";\n";
            }
            break;
        }
        case ASTNodeType::Load: {
            auto* n = dynamic_cast<const LoadStmtNode*>(node);
            label << "Load";
            drawNode(out, thisId, label.str());

            int idNode = nextId++;
            drawNode(out, idNode, "id: " + n->id);
            out << "  node" << thisId << " -> node" << idNode << ";\n";

            int pathNode = nextId++;
            drawNode(out, pathNode, "path: " + n->path);
            out << "  node" << thisId << " -> node" << pathNode << ";\n";

            if (n->alias.has_value()) {
                int aliasNode = nextId++;
                drawNode(out, aliasNode, "alias: " + *n->alias);
                out << "  node" << thisId << " -> node" << aliasNode << ";\n";
            }
            break;
        }
        case ASTNodeType::Set: {
            auto* n = dynamic_cast<const SetStmtNode*>(node);
            label << "Set";
            drawNode(out, thisId, label.str());

            int winNode = nextId++;
            drawNode(out, winNode, "window: " + std::to_string(n->amount) + n->unit);
            out << "  node" << thisId << " -> node" << winNode << ";\n";
            break;
        }
        case ASTNodeType::Transform: {
            auto* n = dynamic_cast<const TransformStmtNode*>(node);
            label << "Transform";
            drawNode(out, thisId, label.str());

            int refNode = nextId++;
            std::string ref = n->ref.isVariable ? n->ref.variableName : (n->ref.table + (n->ref.column ? ("." + *n->ref.column) : ""));
            drawNode(out, refNode, "ref: " + ref);
            out << "  node" << thisId << " -> node" << refNode << ";\n";

            int intvNode = nextId++;
            drawNode(out, intvNode, "interval: " + std::to_string(n->intervalAmount) + n->intervalUnit);
            out << "  node" << thisId << " -> node" << intvNode << ";\n";

            if (n->alias.has_value()) {
                int aliasNode = nextId++;
                drawNode(out, aliasNode, "alias: " + *n->alias);
                out << "  node" << thisId << " -> node" << aliasNode << ";\n";
            }
            break;
        }
        case ASTNodeType::Forecast: {
            auto* n = dynamic_cast<const ForecastStmtNode*>(node);
            label << "Forecast";
            drawNode(out, thisId, label.str());

            int refNode = nextId++;
            std::string ref = n->ref.isVariable ? n->ref.variableName : (n->ref.table + (n->ref.column ? ("." + *n->ref.column) : ""));
            drawNode(out, refNode, "ref: " + ref);
            out << "  node" << thisId << " -> node" << refNode << ";\n";

            int modelNode = nextId++;
            drawNode(out, modelNode, "model: " + n->model);
            out << "  node" << thisId << " -> node" << modelNode << ";\n";

            for (const auto& p : n->params) {
                int paramNode = nextId++;
                drawNode(out, paramNode, p.first + " = " + std::to_string(p.second));
                out << "  node" << thisId << " -> node" << paramNode << ";\n";
            }
            if (n->alias.has_value()) {
                int aliasNode = nextId++;
                drawNode(out, aliasNode, "alias: " + *n->alias);
                out << "  node" << thisId << " -> node" << aliasNode << ";\n";
            }
            break;
        }
        case ASTNodeType::Stream: {
            auto* n = dynamic_cast<const StreamStmtNode*>(node);
            label << "Stream";
            drawNode(out, thisId, label.str());

            int idNode = nextId++;
            drawNode(out, idNode, "id: " + n->id);
            out << "  node" << thisId << " -> node" << idNode << ";\n";

            int pathNode = nextId++;
            drawNode(out, pathNode, "path: " + n->path);
            out << "  node" << thisId << " -> node" << pathNode << ";\n";
            break;
        }
        case ASTNodeType::Select: {
            auto* n = dynamic_cast<const SelectStmtNode*>(node);
            label << "Select";
            drawNode(out, thisId, label.str());

            int refNode = nextId++;
            std::string ref = n->ref.isVariable ? n->ref.variableName : (n->ref.table + (n->ref.column ? ("." + *n->ref.column) : ""));
            drawNode(out, refNode, "ref: " + ref);
            out << "  node" << thisId << " -> node" << refNode << ";\n";

            if (n->op && n->dateExpr) {
                int condNode = nextId++;
                drawNode(out, condNode, "where: DATE " + *n->op + " " + *n->dateExpr);
                out << "  node" << thisId << " -> node" << condNode << ";\n";
            }
            if (n->alias.has_value()) {
                int aliasNode = nextId++;
                drawNode(out, aliasNode, "alias: " + *n->alias);
                out << "  node" << thisId << " -> node" << aliasNode << ";\n";
            }
            break;
        }
        case ASTNodeType::Plot: {
            auto* n = dynamic_cast<const PlotStmtNode*>(node);
            label << "Plot";
            drawNode(out, thisId, label.str());

            int typeNode = nextId++;
            drawNode(out, typeNode, "function: " + n->function);
            out << "  node" << thisId << " -> node" << typeNode << ";\n";

            for (const auto& arg : n->args) {
                int argNode = nextId++;
                drawNode(out, argNode, arg.first + " = " + arg.second);
                out << "  node" << thisId << " -> node" << argNode << ";\n";
            }
            break;
        }
        case ASTNodeType::Export: {
            auto* n = dynamic_cast<const ExportStmtNode*>(node);
            label << "Export";
            drawNode(out, thisId, label.str());

            int refNode = nextId++;
            std::string ref = n->ref.isVariable ? n->ref.variableName : (n->ref.table + (n->ref.column ? ("." + *n->ref.column) : ""));
            drawNode(out, refNode, "ref: " + ref);
            out << "  node" << thisId << " -> node" << refNode << ";\n";

            int tgtNode = nextId++;
            drawNode(out, tgtNode, "to: " + n->target);
            out << "  node" << thisId << " -> node" << tgtNode << ";\n";
            break;
        }
        case ASTNodeType::Loop: {
            auto* n = dynamic_cast<const LoopStmtNode*>(node);
            label << "Loop";
            drawNode(out, thisId, label.str());

            int varNode = nextId++;
            drawNode(out, varNode, "var: " + n->var);
            out << "  node" << thisId << " -> node" << varNode << ";\n";

            int fromNode = nextId++;
            drawNode(out, fromNode, "from: " + std::to_string(n->from));
            out << "  node" << thisId << " -> node" << fromNode << ";\n";

            int toNode = nextId++;
            drawNode(out, toNode, "to: " + std::to_string(n->to));
            out << "  node" << thisId << " -> node" << toNode << ";\n";

            for (const auto& stmt : n->body) {
                int bodyId = drawASTNodeSimplified(stmt.get(), out, nextId);
                out << "  node" << thisId << " -> node" << bodyId << ";\n";
            }
            break;
        }
        case ASTNodeType::Clean: {
            auto* n = dynamic_cast<const CleanStmtNode*>(node);
            label << "Clean";
            drawNode(out, thisId, label.str());

            int actionNode = nextId++;
            drawNode(out, actionNode, "action: " + std::string(n->action == CleanActionType::Remove ? "REMOVE" : "REPLACE"));
            out << "  node" << thisId << " -> node" << actionNode << ";\n";

            int targetNode = nextId++;
            drawNode(out, targetNode, "target: " + n->targetValue);
            out << "  node" << thisId << " -> node" << targetNode << ";\n";

            if (!n->column.empty()) {
                int columnNode = nextId++;
                drawNode(out, columnNode, "column: " + n->column);
                out << "  node" << thisId << " -> node" << columnNode << ";\n";
            }
            if (n->action == CleanActionType::Replace) {
                int replacementNode = nextId++;
                drawNode(out, replacementNode, "with: " + n->replaceWith);
                out << "  node" << thisId << " -> node" << replacementNode << ";\n";
            }
            break;
        }
        default: {
            drawNode(out, thisId, "Unknown Node");
            break;
        }
    }
    return thisId;
}

inline void drawParseTreeSimplified(const std::unique_ptr<ProgramNode>& root, const std::string& outputDotFile) {
    std::ofstream out(outputDotFile);
    out << "digraph AST {\n";
    out << "  node [fontname=\"Courier\"];\n";
    int nextId = 0;
    drawASTNodeSimplified(root.get(), out, nextId);
    out << "}\n";
    out.close();

    std::cout << "AST Graph written to " << outputDotFile << "\n";
}
