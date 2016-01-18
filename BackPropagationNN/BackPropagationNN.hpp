#include <vector>
#include <random>
#include <cmath>
#include <future>
#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>

class Matrix
{
public:
private:
    std::vector<std::vector<float>> data;
public:
    friend std::vector<float> operator*(std::vector<float> const& v, Matrix const& M);
    friend Matrix operator*(Matrix const& lhs, Matrix const& rhs);
    friend Matrix operator+(Matrix const& lhs, Matrix const& rhs);
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

    void initializeWithRandom(std::mt19937* randomGenerator) {
        std::uniform_real_distribution<float> randomDistribution(-1.0f, +1.0f);
        for (auto& row : data) {
            for (auto& value : row) {
                value = randomDistribution(*randomGenerator);
            }
        }
    }

    std::vector<float>& operator[](int rowIndex) {
        return data[rowIndex];
    }

    std::vector<float> const& operator[](int rowIndex) const {
        return data[rowIndex];
    }

    std::string str() const {
        std::ostringstream stream;
        for (auto& row : data) {
            for (auto& value : row) {
                stream << std::setprecision(5) << std::fixed << std::setw(12);
                stream << value;
            }
            stream << std::endl;
        }
        return stream.str();
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
        if (rowCount() != operand.rowCount() ||
            columnCount() != operand.columnCount()) {
            throw std::logic_error("invalid operand.");
        }

        for (int rowIndex = 0; rowIndex != rowCount(); rowIndex++) {
            for (int columnIndex = 0; columnIndex != columnCount(); columnIndex++) {
                data[rowIndex][columnIndex] -= operand[rowIndex][columnIndex];
            }
        }
    }
};

inline std::vector<float> operator*(const std::vector<float>& v, const Matrix& M)
{
    const int rowCount = M.rowCount();
    const int columnCount = M.columnCount();

    if (v.size() != rowCount) {
        throw std::logic_error("invalid operand.");
    }

    auto& data = M.data;
    std::vector<float> result(columnCount, 0.0f);

    for (int rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
        for (int columnIndex = 0; columnIndex < columnCount; ++columnIndex) {
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
                v += lhs[resultRowIndex][index] * rhs[index][resultColumnIndex];
            }
            result[resultRowIndex][resultColumnIndex] = v;
        }
    }
    return result;
}

inline Matrix operator+(Matrix const& leftValue, Matrix const& rightValue)
{
    if (leftValue.columnCount() != rightValue.columnCount()) {
        throw std::logic_error("incompatible matrices");
    }

    if (leftValue.rowCount() != rightValue.rowCount()) {
        throw std::logic_error("incompatible matrices");
    }

    const int rowCount = leftValue.rowCount();
    const int columnCount = leftValue.columnCount();

    Matrix result(rowCount, columnCount);

    for (int rowIndex = 0; rowIndex != rowCount; ++rowIndex) {
        for (int columnIndex = 0; columnIndex != columnCount; ++columnIndex) {
            result[rowIndex][columnIndex] = leftValue[rowIndex][columnIndex] + rightValue[rowIndex][columnIndex];
        }
    }
    return result;
}

inline Matrix operator*(Matrix const& m, float multiplier)
{
    Matrix result = m;
    for (int rowIndex = 0; rowIndex != result.rowCount(); rowIndex++) {
        for (int columnIndex = 0; columnIndex != result.columnCount(); columnIndex++) {
            result[rowIndex][columnIndex] *= multiplier;
        }
    }
    return result;
}

class Expression
{
protected:
    Expression() {}
public:
    virtual ~Expression() {}
    virtual std::future<Matrix> value() = 0;
};

class BinaryOperator : public Expression
{
public:
    std::shared_ptr<Expression> leftOperand;
    std::shared_ptr<Expression> rightOperand;
protected:
    BinaryOperator(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :leftOperand(left), rightOperand(right)
    {
    }

    std::shared_ptr<Expression> left() const { return leftOperand; }
    std::shared_ptr<Expression> right() const { return rightOperand; }

};

class Multiply : public BinaryOperator
{
public:
    Multiply(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
    }

    virtual std::future<Matrix> value() {
        return std::async(std::launch::async, [this]() {
                Matrix leftValue = left()->value().get();
                Matrix rightValue = right()->value().get();
                return leftValue * rightValue;
            });
    }
};

class Plus : public BinaryOperator
{

public:
    Plus(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
    }

    virtual std::future<Matrix> value() {
        return std::async(std::launch::async, [this]() {
                Matrix leftValue = left()->value().get();
                Matrix rightValue = left()->value().get();
                return leftValue + rightValue;
            });
    }
};

class Term : public Expression
{
    Matrix matrix;
public:
    Term(Matrix matrix)
        :matrix(matrix)
    {
    }
    virtual std::future<Matrix> value() {
        return std::async(std::launch::deferred, [this]() {
                return matrix;
            });
    }
};

std::shared_ptr<Expression> operator+(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
{
    return std::make_shared<Plus>(left, right);
}

std::shared_ptr<Expression> operator+(Matrix left, Matrix right)
{
    std::shared_ptr<Term> leftTerm(new Term(left));
    std::shared_ptr<Term> rightTerm(new Term(right));
    return leftTerm + rightTerm;
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
          std::mt19937* randomGenerator)
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
        if (currentInputs.size() != inputs.size()) {
            throw std::logic_error("Input component count doesn't match.");
        }

        currentInputs = inputs;

        if (layerIndex != 0) {
            std::transform(inputs.begin(), inputs.end(), inputs.begin(), tanh);
        }

        currentOutputs = inputs * weights;
    }

    std::vector<float> getOutputs() const
    {
        return currentOutputs;
    }

    std::vector<float> computeSensitivity(std::vector<float> nextLayerSensitivity)
    {
        Matrix nextLayerSensitivityMatrix(static_cast<int>(nextLayerSensitivity.size()), 1, nextLayerSensitivity);
        sensitivity = (weights * nextLayerSensitivityMatrix).asVector();

        std::vector<float> nonLinearizedInput(currentInputs.size());
        std::transform(currentInputs.begin(), currentInputs.end(), nonLinearizedInput.begin(), [](float v) { return 1 - tanh(v) * tanh(v); });

        std::transform(nonLinearizedInput.begin(), nonLinearizedInput.end(), sensitivity.begin(), sensitivity.begin(), [](float i, float s) { return i * s; });

        return sensitivity;
    }

    void updateWeight(std::vector<float> nextLayerSensitivity)
    {
        std::vector<float> nonLinearizedInput(currentInputs.size());
//        if (layerIndex != 0) {
            std::transform(currentInputs.begin(), currentInputs.end(), nonLinearizedInput.begin(), tanh);
//        } else {
//            std::copy(currentInputs.begin(), currentInputs.end(), nonLinearizedInput.begin());
//        }

        const int inputSize = static_cast<int>(currentInputs.size());
        const int outputSize = static_cast<int>(nextLayerSensitivity.size());

        weights -= Matrix(inputSize, 1, nonLinearizedInput) * Matrix(1, outputSize, nextLayerSensitivity) * learningRate;
    }

    void showWeights() const {
        std::cout << weights.str() << std::endl;
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

        const int layerCount = static_cast<int>(componentSizeList.size() - 1);
        for (int layerIndex = 0; layerIndex != layerCount; ++layerIndex) {
            Layer layer(layerIndex,
                        componentSizeList[layerIndex],
                        componentSizeList[layerIndex + 1],
                        0.01, // learning rate
                        &randomGenerator);
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

    void showWeights(int layerIndex) const {
        layers[layerIndex].showWeights();
    }
};
