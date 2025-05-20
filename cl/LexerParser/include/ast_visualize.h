#pragma once
#include "ast.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

inline std::string escapeLabel(const std::string& s);

inline void drawNode(std::ostream& out, int id, const std::string& label);
inline int drawASTNode(const ASTNode* node, std::ostream& out, int& nextId);
void drawParseTree(const std::unique_ptr<ProgramNode>& root, const std::string& outputDotFile);