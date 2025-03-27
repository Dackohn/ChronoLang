#include <iostream>

extern "C" const char* chrono_parse(const char* input);

int main() {
    const char* result = chrono_parse("LOAD data FROM 'x.csv'");
    std::cout << result << std::endl;
    return 0;
}
