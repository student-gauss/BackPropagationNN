#include "BackPropagationNN.hpp"

#include <iostream>

using namespace std;

int main(int argc, const char * argv[]) {
    BackPropagationNN nn({2, 3, 2});

    auto start = std::clock();
    float error;
    for(int i = 0; i < 100000; i++) {
        error = nn.train({0.2, 0.3}, {1.0, 0.0});
        error = nn.train({0.5, 0.1}, {0.0, 1.0});
    }

    cout << "Elapsed " << (std::clock() - start) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    cout << "Test!" << endl;
    auto v1 = nn.test({0.2, 0.3});
    auto v2 = nn.test({0.5, 0.1});

    for (auto v : v1) cout << v << ' ';
    cout << endl;

    for (auto v : v2) cout << v << ' ';
    cout << endl;

    return 0;
}
