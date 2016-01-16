#include "BackPropagationNN.hpp"

#include <iostream>

int main(int argc, const char * argv[]) {
    BackPropagationNN nn({2, 3, 2});

    nn.train({0.2, 0.3}, {0.1, 0.0});


    return 0;
}
