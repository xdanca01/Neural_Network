#include "Neuron.hpp"
#include "Matrix.h"

//#define DATA_SIZE 60000
#define DATA_SIZE 4
//#define INPUTS 784
#define INPUTS 2
#define CLASSES 2

class NeuralNetwork {
private:
	//Training data
	std::vector<std::vector<float>> data;

	//Training data labels
	int labels[DATA_SIZE];

	//[Layer][To][From]
	std::vector<Matrix> Weights;

	//[Layer][Neuron][0] Vec
	std::vector<Matrix> Y;

	//[Layer][Neuron][0] Vec
	std::vector<Matrix> Biases;

public:
	NeuralNetwork(std::string trainingDataFile, std::string labelsFile, std::vector<int>& hiddenNeuronsInLayer);

	//fetch data from csv file with ',' delimiter
	void readData(std::string filename);

	//fetch expectedOutput from cs file with ',' delimiter
	void readExpectedOutput(std::string filename);

	//predict output based on input
	void forwardPropagation(std::vector<float>& inputNeurons);
	std::vector<float> predict();
	std::vector<Matrix> backpropagation(float expectedOutput);

	//Compute gradient with gradient descent method
	std::vector<std::vector<float>> gradientDescent(int expectedOutput);

	void trainNetwork();
	//ReLU activation function
};