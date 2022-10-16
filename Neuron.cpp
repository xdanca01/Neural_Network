#include "Neuron.hpp"

Neuron::Neuron(std::vector<float> w, float b) {
	W = w;
	B = b;
}

void Neuron::computeState(std::vector<int> &X) {
	float innerPotential = B;
	//Compute dot product of Weights * Input
	for (int i = 0; i < X.size(); ++i) {
		innerPotential += X[i] * W[i];
	}
	//TODO set function specific for Neuron
	//Compute Y as Func(innerPotential)
	Y = LogicSigmoid(innerPotential, 1);
}

float Neuron::innerPotential(std::vector<Neuron> &X) {
	float innerPotential = B;
	//Compute dot product of Weights * Input
	for (int i = 0; i < X.size(); ++i) {
		innerPotential += X[i].Y * W[i];
	}
	return innerPotential;
}

void Neuron::computeState(std::vector<Neuron> &X) {
	float innerPotential = B;
	//Compute dot product of Weights * Input
	for (int i = 0; i < X.size(); ++i) {
		innerPotential += X[i].Y * W[i];
	}
	//TODO set function specific for Neuron
	//Compute Y as Func(innerPotential)
	Y = LogicSigmoid(innerPotential, 1);
}

//ReLU activation function
float Neuron::ReLU(float innerPotential) {
	return std::fmax(0.0f, innerPotential);
}

float ReLuDerivative(float innerPotential) {
	return innerPotential > 0.0f ? 1.0f : 0.0f;
}

float Neuron::softPlus(float innerPotential) {
	return std::log(1 + std::exp(innerPotential));
}

float softPlusDerivative(float innerPotential) {
	return 1 / (1 + exp(-innerPotential));
}

float Neuron::LogicSigmoid(float innerPotential, float lambda) {
	return 1 / (1 + exp(-lambda * innerPotential));
}

float Neuron::LogicSigmoidDerivative(float lambda) {
	return lambda * Y * (1 - Y);
}

void Neuron::updateWeights(std::vector<float> &gradient, float learningRate) {
	for (int i = 0; i < W.size(); ++i) {
		W[i] -= gradient[i] * learningRate;
	}
}

