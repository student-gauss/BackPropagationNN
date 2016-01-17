#include <vector>
#include <random>
#include <cmath>

class Matrix
{
public:
private:
    std::vector<std::vector<float>> data;
public:
    friend std::vector<float> operator*(std::vector<float> const& v, Matrix const& M);
    friend Matrix operator*(Matrix const& lhs, Matrix const& rhs);
    friend Matrix operator*(Matrix const& lhs, float v);

    Matrix(int numberOfRows, int numberOfColumns)
        :data(numberOfRows, std::vector<float>(numberOfColumns))
    {
    }

    Matrix(int numberOfRows, int numberOfColumns, std::vector<float> v)
        :data(numberOfRows, std::vector<float>(numberOfColumns))
    {
        int index = 0;
        for (int rowIndex = 0; rowIndex != numberOfRows; ++rowIndex) {
            for (int columnIndex = 0; columnIndex != numberOfColumns; ++columnIndex) {
                data[rowIndex][columnIndex] = v[index++];
            }
        }
    }

    void initializeWithRandom(std::mt19937& randomGenerator) {
        std::uniform_real_distribution<float> randomDistribution(-1.0f, +1.0f);
        for (auto& row : data) {
            for (auto& value : row) {
                value = randomDistribution(randomGenerator);
            }
        }
    }

    void set(int rowIndex, int columnIndex, float v) {
        data[rowIndex][columnIndex] = v;
    }

    float get(int rowIndex, int columnIndex) const {
        return data[rowIndex][columnIndex];
    }

    int rowCount() const
    {
        return static_cast<int>(data.size());
    }

    int columnCount() const
    {
        return static_cast<int>(data[0].size());
    }

    std::vector<float> asVector() const {
        // As optimization, if row count is 1, return the column
        // vector as is.
        if (rowCount() == 1) {
            return data[0];
        }

        std::vector<float> result(rowCount() * columnCount());
        int index = 0;
        for (int rowIndex = 0; rowIndex < rowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < columnCount(); ++columnIndex) {
                result[index++] = data[rowIndex][columnIndex];
            }
        }
        return result;
    }

    void operator-=(Matrix const& operand) {
        for (int rowIndex = 0; rowIndex != rowCount(); rowIndex++) {
            for (int columnIndex = 0; columnIndex != columnCount(); columnIndex++) {
                set(rowIndex, columnIndex, get(rowIndex, columnIndex) - operand.get(rowIndex, columnIndex));
            }
        }
    }
};

inline std::vector<float> operator*(const std::vector<float>& v, const Matrix& M)
{
    auto& data = M.data;
    std::vector<float> result(data[0].size(), 0.0f);

    for (int rowIndex = 0; rowIndex < data.size(); ++rowIndex) {
        auto const& row = data[rowIndex];
        for (int columnIndex = 0; columnIndex < row.size(); ++columnIndex) {
            result[columnIndex] += v[rowIndex] * data[rowIndex][columnIndex];
        }
    }
    return result;
}

inline Matrix operator*(Matrix const& lhs, Matrix const& rhs)
{
    if (lhs.columnCount() != rhs.rowCount()) {
        throw std::logic_error("incompatible matrices");
    }

    const int resultRowCount = lhs.rowCount();
    const int resultColumnCount = rhs.columnCount();

    Matrix result(lhs.rowCount(), rhs.columnCount());

    for (int resultRowIndex = 0; resultRowIndex != resultRowCount; ++resultRowIndex) {
        for (int resultColumnIndex = 0; resultColumnIndex != resultColumnCount; ++resultColumnIndex) {
            const int count = lhs.columnCount();
            float v = 0;
            for (int index = 0; index != count; ++index) {
                v += lhs.get(resultRowIndex, index) * rhs.get(index, resultColumnIndex);
            }
            result.set(resultRowIndex, resultColumnIndex, v);
        }
    }
    return result;
}

inline Matrix operator*(Matrix const& m, float multiplier)
{
    Matrix result = m;
    for (int rowIndex = 0; rowIndex != result.rowCount(); rowIndex++) {
        for (int columnIndex = 0; columnIndex != result.columnCount(); columnIndex++) {
            result.set(rowIndex, columnIndex, result.get(rowIndex, columnIndex) * multiplier);
        }
    }
    return result;
}

class Layer
{
private:
    int layerIndex;
    std::vector<float> currentInputs;
    std::vector<float> currentOutputs;
    std::vector<float> sensitivity;
    float learningRate;
    Matrix weights;
public:
    Layer(int layerIndex,
          int numberOfInputs,
          int numberOfOutputs,
          float learningRate,
          std::mt19937& randomGenerator)
        :layerIndex(layerIndex),
         currentInputs(numberOfInputs),
         learningRate(learningRate),
         weights(numberOfInputs, numberOfOutputs)
    {
        // Initialize weight matrix randomly.
        weights.initializeWithRandom(randomGenerator);
    }

    void setInputs(std::vector<float> inputs)
    {
        currentInputs = inputs;

        std::vector<float> nonLinearizedInput(currentInputs.size());
        if (layerIndex != 0) {
            std::transform(currentInputs.begin(), currentInputs.end(), nonLinearizedInput.begin(), tanh);
        } else {
            std::copy(currentInputs.begin(), currentInputs.end(), nonLinearizedInput.begin());
        }

        currentOutputs = nonLinearizedInput * weights;
    }

    std::vector<float> getOutputs() const
    {
        return currentOutputs;
    }

    std::vector<float> computeSensitivity(std::vector<float> nextLayerSensitivity)
    {
        Matrix nextLayerSensitivityMatrix(static_cast<int>(nextLayerSensitivity.size()), 1, nextLayerSensitivity);
        sensitivity = (weights * nextLayerSensitivityMatrix).asVector();

        std::transform(sensitivity.begin(), sensitivity.end(), sensitivity.begin(), [](float v) { return 1 - tanh(v) * tanh(v); });

        return sensitivity;
    }

    void updateWeight(std::vector<float> nextLayerSensitivity)
    {
        std::vector<float> nonLinearizedInput(currentInputs.size());
        if (layerIndex != 0) {
            std::transform(currentInputs.begin(), currentInputs.end(), nonLinearizedInput.begin(), tanh);
        } else {
            std::copy(currentInputs.begin(), currentInputs.end(), nonLinearizedInput.begin());
        }

        const int inputSize = static_cast<int>(currentInputs.size());
        const int outputSize = static_cast<int>(nextLayerSensitivity.size());

        weights -= Matrix(inputSize, 1, nonLinearizedInput) * Matrix(1, outputSize, nextLayerSensitivity) * learningRate;
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
                        0.01, // learning rate
                        randomGenerator);
            layers.push_back(layer);
        }
    }

    float train(std::vector<float> inputs, std::vector<float> expectedOutputs)
    {
        auto actualOutputs = test(inputs);

        std::vector<float> outputSensitivity(actualOutputs.size());
        std::transform(actualOutputs.begin(), actualOutputs.end(),
                       expectedOutputs.begin(),
                       outputSensitivity.begin(),
                       [](float actual, float expected){ return 2 * (actual - expected); });

        int layerCount = static_cast<int>(layers.size());

        std::vector<float> inputSensitivity;
        for (int layerIndex = layerCount - 1; layerIndex >= 0; --layerIndex) {
            Layer& layer = layers[layerIndex];

            // calculate sensitivity by the sensitivity of the next
            // box.
            inputSensitivity = layer.computeSensitivity(outputSensitivity);
            layer.updateWeight(outputSensitivity);

            outputSensitivity = inputSensitivity;
        }

        float error = inner_product(actualOutputs.begin(), actualOutputs.end(),
                                    expectedOutputs.begin(), 0.0,
                                    std::plus<float>(),
                                    [](float v1, float v2) { return (v1 - v2) * (v1 - v2); } );
        return error;
    }

    std::vector<float> test(std::vector<float> inputs)
    {
        auto layerInputs = inputs;
        for (auto& layer : layers) {
            layer.setInputs(layerInputs);
            layerInputs = layer.getOutputs();
        }
        return layerInputs;
    }
};
