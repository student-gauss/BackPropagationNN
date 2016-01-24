#include <vector>
#include <random>
#include <cmath>
#include <future>
#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <dispatch/dispatch.h>

class Matrix
{
public:
private:
    std::vector<std::vector<float>> data;
public:
    friend Matrix multiplyImmediately(Matrix const& lhs, Matrix const& rhs);

    void swap(Matrix& other) {
        std::swap(data, other.data);
    }

    Matrix()
    {
    }

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

    Matrix(Matrix&& other)
        :data(std::move(other.data))
    {
    }

    Matrix(Matrix const& other)
        :data(other.data)
    {
    }

    Matrix& operator=(Matrix m) {
        swap(m);
        return *this;
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

    int getRowCount() const
    {
        return static_cast<int>(data.size());
    }

    int getColumnCount() const
    {
        return static_cast<int>(data[0].size());
    }

    void applyTanh() {
        for (int rowIndex = 0; rowIndex < getRowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < getColumnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] = tanh(data[rowIndex][columnIndex]);
            }
        }
    }

    void applyDerivativeOfTanh() {
        for (int rowIndex = 0; rowIndex < getRowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < getColumnCount(); ++columnIndex) {
                float v = tanh(data[rowIndex][columnIndex]);
                data[rowIndex][columnIndex] = 1 - v * v;
            }
        }
    }

    void applyScale(float scale) {
        for (int rowIndex = 0; rowIndex < getRowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < getColumnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] *= scale;
            }
        }
    }

    void subtractBy(Matrix const& matrix)
    {
        for (int rowIndex = 0; rowIndex < getRowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < getColumnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] -= matrix[rowIndex][columnIndex];
            }
        }
    }

    void addBy(Matrix const& matrix)
    {
        for (int rowIndex = 0; rowIndex < getRowCount(); ++rowIndex) {
            for (int columnIndex = 0; columnIndex < getColumnCount(); ++columnIndex) {
                data[rowIndex][columnIndex] += matrix[rowIndex][columnIndex];
            }
        }
    }
};

inline Matrix multiplyImmdiately(Matrix lhs, Matrix rhs)
{
    if (lhs.getColumnCount() != rhs.getRowCount()) {
        throw std::logic_error("incompatible matrices");
    }

    const int resultRowCount = lhs.getRowCount();
    const int resultColumnCount = rhs.getColumnCount();

    Matrix result(resultRowCount, resultColumnCount);

    for (int resultRowIndex = 0; resultRowIndex != resultRowCount; ++resultRowIndex) {
        for (int resultColumnIndex = 0; resultColumnIndex != resultColumnCount; ++resultColumnIndex) {
            const int count = lhs.getColumnCount();
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
private:
    std::mutex computeOnceMutex;
    bool wasComputed;
    Matrix computedValue;

protected:
    Expression() : wasComputed(false) {}
public:
    virtual ~Expression() {}
    virtual int getRowCount() const = 0;
    virtual int getColumnCount() const = 0;

    virtual std::string getDescription(int indent = 0) = 0;

    std::string str() {
        Matrix v = value();
        return v.str();
    }

    Matrix value() {
        std::lock_guard<std::mutex> lock(computeOnceMutex);
        if (!wasComputed) {
            computedValue = computeValue();
        }
        return computedValue;
    }
private:
    static void computeValueOnce(void *context) {
        Expression *expression = reinterpret_cast<Expression *>(context);
        expression->computedValue = expression->computeValue();
    }
    virtual Matrix computeValue() = 0;
};

class BinaryOperator : public Expression
{
private:
    std::shared_ptr<Expression> leftOperand;
    std::shared_ptr<Expression> rightOperand;

    dispatch_queue_t dispatchQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

protected:
    BinaryOperator(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :leftOperand(left), rightOperand(right)
    {
    }

    virtual int getRowCount() const
    {
        return left()->getRowCount();
    }

    virtual int getColumnCount() const
    {
        return left()->getColumnCount();
    }

    std::shared_ptr<Expression> left() const { return leftOperand; }
    std::shared_ptr<Expression> right() const { return rightOperand; }

    std::pair<Matrix, Matrix> computeOperands() {
        dispatch_group_t group = dispatch_group_create();

        __block Matrix leftValue;
        dispatch_group_async( group, dispatchQueue, ^{
                auto leftExpression = left();
                leftValue = leftExpression->value();
            });

        __block Matrix rightValue;
        dispatch_group_async( group, dispatchQueue, ^{
                auto rightExpression = right();
                rightValue = rightExpression->value();
            });
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        dispatch_release(group);

        return std::make_pair(std::move(leftValue), std::move(rightValue));
    }
};

class UnaryOperator : public Expression
{
private:
    std::shared_ptr<Expression> operand;
public:
    UnaryOperator(std::shared_ptr<Expression> argument)
        :operand(argument) {
    }

    virtual int getRowCount() const
    {
        return argument()->getRowCount();
    }

    virtual int getColumnCount() const
    {
        return argument()->getColumnCount();
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

    virtual int getRowCount() const
    {
        return left()->getRowCount();
    }

    virtual int getColumnCount() const
    {
        return right()->getColumnCount();
    }

    virtual Matrix computeValue() {
        auto operands = computeOperands();
        return multiplyImmdiately(std::move(operands.first), std::move(operands.second));
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Multiply[" << this << "]" << std::endl;
        sstream << left()->getDescription(indent + 2);
        sstream << right()->getDescription(indent + 2);

        return sstream.str();
    }

};

class Plus : public BinaryOperator
{
public:
    Plus(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
        if (left->getRowCount() != right->getRowCount() ||
            left->getColumnCount() != right->getColumnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual Matrix computeValue() {
        auto operands = computeOperands();
        Matrix leftValue = std::move(operands.first);
        Matrix rightValue = std::move(operands.second);
        leftValue.addBy(rightValue);
        return leftValue;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Plus[" << this << "]" << std::endl;
        sstream << left()->getDescription(indent + 2);
        sstream << right()->getDescription(indent + 2);

        return sstream.str();
    }
};

class Minus : public BinaryOperator
{
public:
    Minus(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
        if (left->getRowCount() != right->getRowCount() ||
            left->getColumnCount() != right->getColumnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual Matrix computeValue() {
        auto operands = computeOperands();
        Matrix leftValue = std::move(operands.first);
        Matrix rightValue = std::move(operands.second);
        leftValue.subtractBy(rightValue);
        return leftValue;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Minus[" << this << "]" << std::endl;
        sstream << left()->getDescription(indent + 2);
        sstream << right()->getDescription(indent + 2);

        return sstream.str();
    }
};

class ElementByElementMultiply : public BinaryOperator
{
public:
    ElementByElementMultiply(std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        :BinaryOperator(left, right)
    {
        if (left->getRowCount() * left->getColumnCount() != right->getRowCount() * right->getColumnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual Matrix computeValue() {
        Matrix leftValue = left()->value();
        Matrix rightValue = right()->value();

        int rowCount = leftValue.getRowCount();
        int columnCount = leftValue.getColumnCount();
        int index = 0;
        for (int rowIndex = 0; rowIndex != rowCount; rowIndex++) {
            for (int columnIndex = 0; columnIndex != columnCount; columnIndex++) {
                int rightRowIndex = index / rightValue.getColumnCount();
                int rightColumnIndex = index % rightValue.getColumnCount();
                leftValue[rowIndex][columnIndex] *= rightValue[rightRowIndex][rightColumnIndex];
                index++;
            }
        }

        return leftValue;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Element-By-Element Multiply[" << this << "]" << std::endl;
        sstream << left()->getDescription(indent + 2);
        sstream << right()->getDescription(indent + 2);

        return sstream.str();
    }
};

class OutputSensitivity : public BinaryOperator
{
public:
    OutputSensitivity(std::shared_ptr<Expression> prediction, std::shared_ptr<Expression> truth)
        :BinaryOperator(prediction, truth)
    {
        if (prediction->getRowCount() != 1 ||
            truth->getRowCount() != 1 ||
            prediction->getColumnCount() != truth->getColumnCount()) {
            throw std::invalid_argument("incompatible matrix.");
        }
    }

    virtual int getRowCount() const
    {
        return left()->getColumnCount();
    }

    virtual int getColumnCount() const
    {
        return 1;
    }

    virtual Matrix computeValue() {
        Matrix prediction = left()->value();
        Matrix truth = right()->value();
        Matrix sensitivity(prediction.getColumnCount(), 1);

        int columnCount = prediction.getColumnCount();
        for (int columnIndex = 0; columnIndex != columnCount; columnIndex++) {
            sensitivity[columnIndex][0] = 2 * (prediction[0][columnIndex] - truth[0][columnIndex]);
        }

        return sensitivity;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "OutputSensitibity[" << this << "]" << std::endl;
        sstream << left()->getDescription(indent + 2);
        sstream << right()->getDescription(indent + 2);

        return sstream.str();
    }
};

class Tanh : public UnaryOperator
{
public:
    Tanh(std::shared_ptr<Expression> operand)
        :UnaryOperator(operand) {
    }

    virtual Matrix computeValue() {
        Matrix argumentValue = argument()->value();
        argumentValue.applyTanh();
        return argumentValue;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Tanh[" << this << "]" << std::endl;
        sstream << argument()->getDescription(indent + 2);

        return sstream.str();
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

    virtual Matrix computeValue() {
        Matrix argumentValue = argument()->value();
        argumentValue.applyScale(scale);
        return argumentValue;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Scale[" << this << "]" << std::endl;
        sstream << argument()->getDescription(indent + 2);

        return sstream.str();
    }
};

class DerivativeOfTanh : public UnaryOperator
{
public:
    DerivativeOfTanh(std::shared_ptr<Expression> operand)
        :UnaryOperator(operand) {
    }

    virtual Matrix computeValue() {
        Matrix argumentValue = argument()->value();
        argumentValue.applyDerivativeOfTanh();
        return argumentValue;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "DTanh[" << this << "]" << std::endl;
        sstream << argument()->getDescription(indent + 2);

        return sstream.str();
    }
};

class Transpose : public UnaryOperator
{
public:
    Transpose(std::shared_ptr<Expression> argument)
        :UnaryOperator(argument) {
    }

    virtual int getRowCount() const
    {
        return argument()->getColumnCount();
    }

    virtual int getColumnCount() const
    {
        return argument()->getRowCount();
    }

    virtual Matrix computeValue() {
        Matrix source = argument()->value();
        Matrix transposed(source.getColumnCount(), source.getRowCount());

        for (int rowIndex = 0; rowIndex != source.getRowCount(); rowIndex++) {
            for (int columnIndex = 0; columnIndex != source.getColumnCount(); columnIndex++) {
                transposed[columnIndex][rowIndex] = source[rowIndex][columnIndex];
            }
        }
        return transposed;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Transpose[" << this << "]" << std::endl;
        sstream << argument()->getDescription(indent + 2);

        return sstream.str();
    }
};

class Term : public Expression
{
    int rowCount;
    int columnCount;
    Matrix matrix;
public:
    Term(Matrix initMatrix)
        :matrix(initMatrix),
         rowCount(initMatrix.getRowCount()),
         columnCount(initMatrix.getColumnCount())
    {
    }

    Term(std::vector<float> const& v)
        :matrix(1, static_cast<int>(v.size()), v),
         rowCount(1),
         columnCount(static_cast<int>(v.size()))
    {
    }

    virtual int getRowCount() const
    {
        return rowCount;
    }

    virtual int getColumnCount() const
    {
        return columnCount;
    }

    virtual Matrix computeValue() {
        return matrix;
    }

    virtual std::string getDescription(int indent)
    {
        std::ostringstream sstream;
        sstream << std::string(indent, ' ');
        sstream << "Matrix" << std::endl;
        return sstream.str();
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
    using namespace std;
//    cout << v->getDescription() << std::endl;
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
        if (inputs->getRowCount() != 1 ||
            inputs->getColumnCount() != numberOfInputs ) {
            throw std::logic_error("Input component count doesn't match.");
        }

        currentInputs = inputs;

        if (layerIndex != 0) {
            inputs = std::make_shared<Tanh>(inputs);
        }

        currentOutputs = inputs * weights;
    }

    std::shared_ptr<Expression> getOutputs() const
    {
        return currentOutputs;
    }

    std::shared_ptr<Expression> computeSensitivity(std::shared_ptr<Expression> nextLayerSensitivity)
    {
        if (nextLayerSensitivity->getRowCount()    != numberOfOutputs ||
            nextLayerSensitivity->getColumnCount() != 1 ) {
            throw std::invalid_argument("invalid matrix.");
        }

        sensitivity = elementByElementMultiply(weights * nextLayerSensitivity, derivativeOfTanh(currentInputs));

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
