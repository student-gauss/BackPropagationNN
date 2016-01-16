#include <vector>
#include <random>
#include <cmath>

class Matrix
{
public:
    typedef std::vector<float> Vector;

private:
    std::vector<Vector> data;
public:
    friend std::vector<float> operator*(std::vector<float> const& v, Matrix const& M);

    Matrix(int numberOfRows, int numberOfColumns)
        :data(numberOfRows, Vector(numberOfColumns))
    {
    }

    void initializeWithRandom(std::mt19937& randomGenerator) {
        std::uniform_real_distribution<float> randomDistribution(-1.0f, +1.0f);
        for (auto& row : data) {
            for (auto& value : row) {
                value = randomDistribution(randomGenerator);
            }
        }
    }
};

inline std::vector<float> operator*(const std::vector<float>& v, const Matrix& M)
{
    auto data = M.data;
    std::vector<float> result(data[0].size(), 0.0f);

    for (int rowIndex = 0; rowIndex < data.size(); ++rowIndex) {
        auto const& row = data[rowIndex];
        for (int columnIndex = 0; columnIndex < row.size(); ++columnIndex) {
            result[columnIndex] += v[rowIndex] * data[rowIndex][columnIndex];
        }
    }
    return result;
}

class Layer
{
private:
    int layerIndex;
    std::vector<float> inputs;
    std::vector<float> outputs;
    float learningRate;
    Matrix weights;
public:
    Layer(int layerIndex,
          int numberOfInputs,
          int numberOfOutputs,
          float learningRate,
          std::mt19937& randomGenerator)
        :layerIndex(layerIndex),
         inputs(numberOfInputs),
         learningRate(learningRate),
         weights(numberOfInputs, numberOfOutputs)
    {
        // Initialize weight matrix randomly.
        weights.initializeWithRandom(randomGenerator);
    }

    void setInputs(std::vector<float> inputs)
    {
        this->inputs = inputs;

        std::vector<float> nonLinearizedInput(this->inputs.size());
        if (layerIndex != 0) {
            std::transform(inputs.begin(), inputs.end(), nonLinearizedInput.begin(), tanh);
        } else {
            std::copy(inputs.begin(), inputs.end(), nonLinearizedInput.begin());
        }

        this->outputs = nonLinearizedInput * weights;
    }

    std::vector<float> getOutputs() const
    {
        return this->outputs;
    }
};

class BackPropagationNN
{
private:
    std::vector<Layer> layers;

public:
    BackPropagationNN(std::vector<int> componentSizeList)
    {
        std::mt19937 randomGenerator;

        if (componentSizeList.size() < 2) {
            throw std::invalid_argument("invalid parameters.");
        }

        const size_t layerCount = componentSizeList.size() - 1;
        for (size_t layerIndex = 0; layerIndex != layerCount; ++layerIndex) {
            Layer layer(static_cast<int>(layerIndex),
                        componentSizeList[layerIndex],
                        componentSizeList[layerIndex + 1],
                        0.01,
                        randomGenerator);
            layers.push_back(layer);
        }
    }

    void train(std::vector<float> inputs, std::vector<float> expectedOutputs)
    {
        auto layerInputs = inputs;
        for (auto layer : layers) {
            layer.setInputs(layerInputs);
            layerInputs = layer.getOutputs();
        }

        auto actualOutputs = layerInputs;

    }


};
