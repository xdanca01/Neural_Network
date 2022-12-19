#include "Matrix.h"
#include <omp.h>
#include <random>
#include <ctime>

#define DATA_SIZE 60000
#define PREDICT_SIZE 10000
#define INPUTS 784
#define CLASSES 10
#define THREAD_NUM 8 // for Aisa

class NeuralNetwork {
private:
	//Training data
	std::vector<std::vector<float> > data;
	std::vector<std::vector<float> > dataForCompare;

	//Training data labels
	std::vector<float> labels;
	//Data labels to predict
	std::vector<float> labelsForCompare;

	//[Layer][To][From]
	std::vector<Matrix> Weights;

	//[Layer][Neuron][0] Vec
	std::vector<Matrix> Y;
	std::vector<std::vector<Matrix> > Y2;

	//[Layer][Neuron][0] Vec
	std::vector<Matrix> Biases;

public:
	NeuralNetwork(std::string trainingDataFile, std::string labelsFile, std::vector<int>& hiddenNeuronsInLayer);

	//fetch data from csv file with ',' delimiter
	void readData(std::string filename, bool compare);

	//fetch expectedOutput from cs file with ',' delimiter
	void readExpectedOutput(std::string filename, bool compare);

	//predict output based on input
	void forwardPropagation(std::vector<float>& inputNeurons);
	void forwardPropagation(std::vector<float>& inputNeurons, int thread);
	float predict();
	std::vector<Matrix> backpropagation(float expectedOutput);
	std::vector<Matrix> backpropagation(float expectedOutput, int thread);

	void trainNetworkThreads();
	void writeLabelToFile(std::string filename, std::vector<float> writeData, int dataSize);

	unsigned argMax();
	unsigned argMaxThreads(int thread);
	//ReLU activation function
};