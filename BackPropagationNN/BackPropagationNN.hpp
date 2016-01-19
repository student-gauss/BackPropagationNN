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
    friend Matrix multiplyImmediately(Matrix const& lhs, Matrix const& rhs);

    Matrix(int numberOfRows, int numberOfColumns)
        :data(numberOfRows, std::vector<float>(numberOfColumns))
    {
    }

    Matrix(int numberOfRows, int numberOfColumns, std::vector<float> const& v)
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

    void applyTanh() {
        for (int rowIndex = 0; rowIndex < rowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < columnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] = tanh(data[rowIndex][columnIndex]);
            }
        }
    }

    void applyDerivativeOfTanh() {
        for (int rowIndex = 0; rowIndex < rowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < columnCount(); ++columnIndex) {
                float v = tanh(data[rowIndex][columnIndex]);
                data[rowIndex][columnIndex] = 1 - v * v;
            }
        }
    }

    void applyScale(float scale) {
        for (int rowIndex = 0; rowIndex < rowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < columnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] *= scale;
            }
        }
    }

    void subtractBy(Matrix const& matrix)
    {
        for (int rowIndex = 0; rowIndex < rowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < columnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] -= matrix[rowIndex][columnIndex];
            }
        }
    }

    void addBy(Matrix const& matrix)
    {
        for (int rowIndex = 0; rowIndex < rowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < columnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] += matrix[rowIndex][columnIndex];
            }
        }
    }
};

inline Matrix multiplyImmdiately(Matrix const& lhs, Matrix const& rhs)
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

class Expression
{
    std::shared_future<Matrix> computedValue;
protected:
    Expression() {}
public:
    virtual ~Expression() {}
    virtual int rowCount() const = 0;
    virtual int columnCount() const = 0;

    virtual Matrix value() = 0;

    std::string str() {
        Matrix v = value();
        return v.str();
    }
};

class BinaryOperator : public Expression
{
private:
    std::shared_ptr<Expression> leftOperand;
    std::shared_ptr<Expression> rightOperand;
protected:
    BinaryOperator(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :leftOperand(left), rightOperand(right)
    {
    }

    virtual int rowCount() const
    {
        return left()->rowCount();
    }

    virtual int columnCount() const
    {
        return left()->columnCount();
    }

    std::shared_ptr<Expression> left() const { return leftOperand; }
    std::shared_ptr<Expression> right() const { return rightOperand; }
};

class UnaryOperator : public Expression
{
private:
    std::shared_ptr<Expression> operand;
public:
    UnaryOperator(std::shared_ptr<Expression> argument)
        :operand(argument) {
    }

    virtual int rowCount() const
    {
        return argument()->rowCount();
    }

    virtual int columnCount() const
    {
        return argument()->columnCount();
    }

    std::shared_ptr<Expression> argument() const { return operand; }
};

class Multiply : public BinaryOperator
{
public:
    Multiply(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
    }

    virtual int rowCount() const
    {
        return left()->rowCount();
    }

    virtual int columnCount() const
    {
        return right()->columnCount();
    }

    virtual Matrix value() {
        Matrix leftValue = left()->value();
        Matrix rightValue = right()->value();
        return multiplyImmdiately(leftValue, rightValue);
    }
};

class Plus : public BinaryOperator
{
public:
    Plus(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
        if (left->rowCount() != right->rowCount() ||
            left->columnCount() != right->columnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual Matrix value() {
        Matrix leftValue = left()->value();
        Matrix rightValue = right()->value();
        leftValue.addBy(rightValue);
        return leftValue;
    }
};

class Minus : public BinaryOperator
{
public:
    Minus(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
        if (left->rowCount() != right->rowCount() ||
            left->columnCount() != right->columnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual Matrix value() {
        Matrix leftValue = left()->value();
        Matrix rightValue = right()->value();
        leftValue.subtractBy(rightValue);
        return leftValue;
    }
};

class ElementByElementMultiply : public BinaryOperator
{
public:
    ElementByElementMultiply(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
        if (left->rowCount() * left->columnCount() != right->rowCount() * right->columnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual Matrix value() {
        Matrix leftValue = left()->value();
        Matrix rightValue = right()->value();

        int rowCount = leftValue.rowCount();
        int columnCount = leftValue.columnCount();
        int index = 0;
        for (int rowIndex = 0; rowIndex != rowCount; rowIndex++) {
            for (int columnIndex = 0; columnIndex != columnCount; columnIndex++) {
                int rightRowIndex = index / rightValue.columnCount();
                int rightColumnIndex = index % rightValue.columnCount();
                leftValue[rowIndex][columnIndex] *= rightValue[rightRowIndex][rightColumnIndex];
                index++;
            }
        }

        return leftValue;
    }
};

class OutputSensitivity : public BinaryOperator
{
public:
    OutputSensitivity(std::shared_ptr<Expression> prediction, std::shared_ptr<Expression> truth)
        :BinaryOperator(prediction, truth)
    {
        if (prediction->rowCount() != 1 ||
            truth->rowCount() != 1 ||
            prediction->columnCount() != truth->columnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual int rowCount() const
    {
        return left()->columnCount();
    }

    virtual int columnCount() const
    {
        return 1;
    }

    virtual Matrix value() {
        Matrix prediction = left()->value();
        Matrix truth = right()->value();
        Matrix sensitivity(prediction.columnCount(), 1);

        int columnCount = prediction.columnCount();
        for (int columnIndex = 0; columnIndex != columnCount; columnIndex++) {
            sensitivity[columnIndex][0] = 2 * (prediction[0][columnIndex] - truth[0][columnIndex]);
        }

        return sensitivity;
    }
};

class Tanh : public UnaryOperator
{
public:
    Tanh(std::shared_ptr<Expression> operand)
        :UnaryOperator(operand) {
    }

    virtual Matrix value() {
        Matrix argumentValue = argument()->value();
        argumentValue.applyTanh();
        return argumentValue;
    }
};

class Scale : public UnaryOperator
{
private:
    float scale;
public:
    Scale(std::shared_ptr<Expression> operand, float scale)
        :UnaryOperator(operand),
         scale(scale){
    }

    virtual Matrix value() {
        Matrix argumentValue = argument()->value();
        argumentValue.applyScale(scale);
        return argumentValue;
    }
};

class DerivativeOfTanh : public UnaryOperator
{
public:
    DerivativeOfTanh(std::shared_ptr<Expression> operand)
        :UnaryOperator(operand) {
    }

    virtual Matrix value() {
        Matrix argumentValue = argument()->value();
        argumentValue.applyDerivativeOfTanh();
        return argumentValue;
    }
};

class Transpose : public UnaryOperator
{
public:
    Transpose(std::shared_ptr<Expression> argument)
        :UnaryOperator(argument) {
    }

    virtual int rowCount() const
    {
        return argument()->columnCount();
    }

    virtual int columnCount() const
    {
        return argument()->rowCount();
    }

    virtual Matrix value() {
        Matrix source = argument()->value();
        Matrix transposed(source.columnCount(), source.rowCount());

        for (int rowIndex = 0; rowIndex != source.rowCount(); rowIndex++) {
            for (int columnIndex = 0; columnIndex != source.columnCount(); columnIndex++) {
                transposed[columnIndex][rowIndex] = source[rowIndex][columnIndex];
            }
        }
        return transposed;
    }
};

class Term : public Expression
{
    Matrix matrix;
public:
    Term(Matrix initMatrix)
        :matrix(initMatrix)
    {
    }

    Term(std::vector<float> const& v)
        :matrix(1, static_cast<int>(v.size()), v)
    {
    }

    virtual int rowCount() const
    {
        return matrix.rowCount();
    }

    virtual int columnCount() const
    {
        return matrix.columnCount();
    }

    virtual Matrix value() {
        return matrix;
    }
};

std::shared_ptr<Expression> operator+(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
{
    return std::make_shared<Plus>(left, right);
}

std::shared_ptr<Expression> operator-(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
{
    return std::make_shared<Minus>(left, right);
}

std::shared_ptr<Expression> operator*(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
{
    return std::make_shared<Multiply>(left, right);
}

std::shared_ptr<Expression> operator*(std::shared_ptr<Expression> argument, float multiplier)
{
    return std::make_shared<Scale>(argument, multiplier);
}

std::shared_ptr<Expression> operator+(Matrix left, Matrix right)
{
    return std::make_shared<Term>(left) + std::make_shared<Term>(right);
}

std::shared_ptr<Expression> operator*(Matrix left, Matrix right)
{
    return std::make_shared<Term>(left) * std::make_shared<Term>(right);
}

std::shared_ptr<Expression> transpose(std::shared_ptr<Expression> argument)
{
    return std::make_shared<Transpose>(argument);
}

std::shared_ptr<Expression> derivativeOfTanh(std::shared_ptr<Expression> argument)
{
    return std::make_shared<DerivativeOfTanh>(argument);
}

std::shared_ptr<Expression> elementByElementMultiply(std::shared_ptr<Expression> argument1, std::shared_ptr<Expression> argument2)
{
    return std::make_shared<ElementByElementMultiply>(argument1, argument2);
}

std::shared_ptr<Expression> errorDerivative(std::shared_ptr<Expression> argument1, std::shared_ptr<Expression> argument2)
{
    return std::make_shared<OutputSensitivity>(argument1, argument2);
}

std::shared_ptr<Expression> compute(std::shared_ptr<Expression> v)
{
    return std::make_shared<Term>(v->value());
}

class Layer
{
private:
    const int layerIndex;
    const int numberOfInputs;
    const int numberOfOutputs;
    const float learningRate;

    std::shared_ptr<Expression> currentInputs;
    std::shared_ptr<Expression> currentOutputs;
    std::shared_ptr<Expression> sensitivity;
    std::shared_ptr<Expression> weights;
public:
    Layer(int layerIndex,
          int numberOfInputs,
          int numberOfOutputs,
          float learningRate,
          std::mt19937* randomGenerator)
        :layerIndex(layerIndex),
         numberOfInputs(numberOfInputs),
         numberOfOutputs(numberOfOutputs),
         learningRate(learningRate)
    {
        // Initialize weight matrix randomly.
        Matrix weightMatrix(numberOfInputs, numberOfOutputs);
        weightMatrix.initializeWithRandom(randomGenerator);

        weights = std::make_shared<Term>(weightMatrix);
    }

    void setInputs(std::shared_ptr<Expression> inputs)
    {
        if (inputs->rowCount() != 1 ||
            inputs->columnCount() != numberOfInputs ) {
            throw std::logic_error("Input component count doesn't match.");
        }

        currentInputs = compute(inputs);

        if (layerIndex != 0) {
            inputs = std::make_shared<Tanh>(inputs);
        }

        currentOutputs = compute(inputs * weights);
    }

    std::shared_ptr<Expression> getOutputs() const
    {
        return currentOutputs;
    }

    std::shared_ptr<Expression> computeSensitivity(std::shared_ptr<Expression> nextLayerSensitivity)
    {
        if (nextLayerSensitivity->rowCount()    != numberOfOutputs ||
            nextLayerSensitivity->columnCount() != 1 ) {
            throw std::invalid_argument("invalid matrix.");
        }

        sensitivity = compute(elementByElementMultiply(weights * nextLayerSensitivity, derivativeOfTanh(currentInputs)));

        return sensitivity;
    }

    void updateWeight(std::shared_ptr<Expression> nextLayerSensitivity)
    {
        auto nonLinearizedInput = std::make_shared<Tanh>(currentInputs);
        auto newWeight = compute(weights - transpose(nonLinearizedInput) * transpose(nextLayerSensitivity) * learningRate);
        weights = newWeight;
    }

    void showWeights() const {
        std::cout << weights->str() << std::endl;
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

    float train(std::vector<float> inputs, std::vector<float> truthOutputs)
    {
        auto prediction = test(inputs);
        auto outputSensitivity = errorDerivative(prediction, std::make_shared<Term>(truthOutputs));

        int layerCount = static_cast<int>(layers.size());

        std::shared_ptr<Expression> inputSensitivity;
        for (int layerIndex = layerCount - 1; layerIndex >= 0; --layerIndex) {
            Layer& layer = layers[layerIndex];

            // calculate sensitivity by the sensitivity of the next
            // box.
            inputSensitivity = layer.computeSensitivity(outputSensitivity);
            layer.updateWeight(outputSensitivity);

            outputSensitivity = inputSensitivity;
        }
/*
        float error = inner_product(actualOutputs.begin(), actualOutputs.end(),
                                    expectedOutputs.begin(), 0.0,
                                    std::plus<float>(),
                                    [](float v1, float v2) { return (v1 - v2) * (v1 - v2); } );
*/
        return 0;
    }

    std::shared_ptr<Expression> test(std::vector<float> inputs)
    {
        std::shared_ptr<Expression> layerInputs = std::make_shared<Term>(inputs);
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
