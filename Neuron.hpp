#pragma once

#include <vector>
#include <algorithm>
#include <string_view>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

class Neuron {
	float B;

public:
	float Y;
	std::vector<float> W;
	Neuron(std::vector<float> w, float b);
	void computeState(std::vector<int>& X);
	void computeState(std::vector<Neuron>& X);
	float innerPotential(std::vector<Neuron>& X);
	float ReLU(float innerPotential);
	float ReLuDerivative(float innerPotential);
	float LogicSigmoid(float innerPotential, float lambda);
	float LogicSigmoidDerivative(float lambda);
	float softPlus(float innerPotential);
	float softPlusDerivative(float innerPotential);
	void updateWeights(std::vector<float>& gradient, float learningRate);
};
