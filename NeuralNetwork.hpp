#include "Neuron.hpp"
#include "Matrix.h"

#define DATA_SIZE 60000
#define INPUTS 784

class NeuralNetwork {
	private:
		//Training data
		std::vector<std::vector<float>> data;
		//Training data labels
		int labels[DATA_SIZE];
		//Hidden layers and output layer with neurons saved in it
		std::vector<std::vector<Neuron>> Layers;
		//[Layer][To][From]
		std::vector<Matrix*> Weights;
		std::vector<Matrix*> Y;
		std::vector<Matrix*> Biases;

	public:
		NeuralNetwork(std::string trainingDataFile, std::string labelsFile, std::vector<int> &hiddenNeuronsInLayer);

		//fetch data from csv file with ',' delimiter
		void readData(std::string filename);

		//fetch expectedOutput from cs file with ',' delimiter
		void readExpectedOutput(std::string filename);

		//predict output based on input
		void forwardPropagation(std::vector<float> &inputNeurons);
		std::vector<float> predict();
		void backpropagation();
		//Compute gradient with gradient descent method
		std::vector<std::vector<float>> gradientDescent(int expectedOutput);

		//computes prediction percentage for each neuron output
		void computeSoftMax(std::vector<float> outputNeurons, std::vector<Neuron> &outputLayer);

		void trainNetwork();
		//ReLU activation function


};