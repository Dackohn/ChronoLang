#include "third_party/nlohmann/json.hpp"
#include "../include/ast.h"
using json = nlohmann::json;
json astToJson(const ASTNode* node);