//#pragma once
#include "ast.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

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

inline int drawASTNode(const ASTNode* node, std::ostream& out, int& nextId) {
    int thisId = nextId++;

    std::ostringstream label;
    switch (node->type) {
        case ASTNodeType::Program: {
            label << "Program";
            drawNode(out, thisId, label.str());
            const auto* program = dynamic_cast<const ProgramNode*>(node);
            for (const auto& stmt : program->statements) {
                int childId = drawASTNode(stmt.get(), out, nextId);
                out << "  node" << thisId << " -> node" << childId << ";\n";
            }
            break;
        }
        case ASTNodeType::Load: {
            const auto* n = dynamic_cast<const LoadStmtNode*>(node);
            label << "Load\nID: " << n->id << "\nPath: " << n->path;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Set: {
            const auto* n = dynamic_cast<const SetStmtNode*>(node);
            label << "Set\nWindow: " << n->amount << n->unit;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Transform: {
            const auto* n = dynamic_cast<const TransformStmtNode*>(node);
            label << "Transform\nColumn: " << n->column << "\nNext: " << n->intervalAmount << n->intervalUnit;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Forecast: {
            const auto* n = dynamic_cast<const ForecastStmtNode*>(node);
            label << "Forecast\nColumn: " << n->column << "\nModel: " << n->model;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Stream: {
            const auto* n = dynamic_cast<const StreamStmtNode*>(node);
            label << "Stream\nID: " << n->id << "\nPath: " << n->path;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Select: {
            const auto* n = dynamic_cast<const SelectStmtNode*>(node);
            label << "Select\nColumn: " << n->column;
            if (n->op && n->dateExpr)
                label << "\nWhere DATE " << *n->op << " " << *n->dateExpr;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Plot: {
            const auto* n = dynamic_cast<const PlotStmtNode*>(node);
            label << "Plot\nFunction: " << n->function;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Export: {
            const auto* n = dynamic_cast<const ExportStmtNode*>(node);
            label << "Export\nFrom: " << n->source << "\nTo: " << n->target;
            drawNode(out, thisId, label.str());
            break;
        }
        case ASTNodeType::Loop: {
            const auto* n = dynamic_cast<const LoopStmtNode*>(node);
            label << "Loop\n" << n->var << " in " << n->from << " to " << n->to;
            drawNode(out, thisId, label.str());
            for (const auto& stmt : n->body) {
                int childId = drawASTNode(stmt.get(), out, nextId);
                out << "  node" << thisId << " -> node" << childId << ";\n";
            }
            break;
        }
        case ASTNodeType::Clean: {
            const auto* n = dynamic_cast<const CleanStmtNode*>(node);
            if (n->action == CleanActionType::Remove) {
                label << "Clean (REMOVE)\nTarget: " << n->targetValue << "\nFrom: " << n->column;
            } else if (n->action == CleanActionType::Replace) {
                label << "Clean (REPLACE)\nTarget: " << n->targetValue
                      << "\nIn: " << n->column << "\nWith: " << n->replaceWith;
            }
            drawNode(out, thisId, label.str());
            break;
        }
        default: {
            drawNode(out, thisId, "Unknown Node");
            break;
        }
    }

    return thisId;
}

inline void drawParseTree(const std::unique_ptr<ProgramNode>& root, const std::string& outputDotFile) {
    std::ofstream out(outputDotFile);
    out << "digraph AST {\n";
    out << "  node [fontname=\"Courier\"];\n";
    int nextId = 0;
    drawASTNode(root.get(), out, nextId);
    out << "}\n";
    out.close();

    std::cout << "AST Graph written to " << outputDotFile << "\n";
}
