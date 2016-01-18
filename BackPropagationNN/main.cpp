#include "BackPropagationNN.hpp"

#include <iostream>

using namespace std;

int main(int argc, const char * argv[]) {
    BackPropagationNN nn({2, 10, 4});

    struct Train {
        std::vector<float> inputs;
        std::vector<float> outputs;
    };

    Train trainSet[] = {
        { {0.1, 0.2}, {1, 0, 0, 0} },
        { {0.3, 0.4}, {0, 1, 0, 0} },
        { {0.5, 0.6}, {0, 0, 1, 0} },
        { {0.7, 0.8}, {0, 0, 0, 1} },
    };

    auto start = std::clock();
    float error;
    for(int i = 0; i < 300000; i++) {
        int index = rand() % (sizeof(trainSet) / sizeof(trainSet[0]));
        error = nn.train(trainSet[index].inputs, trainSet[index].outputs);
//        cout << error << endl;

/*        for(int i = 0; i < 1; i++) {
            cout << "Weight for layer " << i << endl;
            nn.showWeights(i);
        }
*/
    }


    cout << "Elapsed " << (std::clock() - start) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    cout << "Test!" << endl;
    vector<float> v[4];
    v[0] = nn.test({0.1, 0.2});
    v[1] = nn.test({0.3, 0.4});
    v[2] = nn.test({0.5, 0.6});
    v[3] = nn.test({0.7, 0.8});

    for (auto const& result : v) {
        for (auto value : result) {
            cout << std::setprecision(5) << std::fixed << std::setw(12);
            cout << value;
        }
        cout << endl;
    }

    return 0;
}
