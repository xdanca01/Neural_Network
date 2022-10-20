#include "NeuralNetwork.hpp"

#define Fsigm ReLU
#define Fderivative ReLuDerivative

using namespace std;

string file1 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_vectors.csv";
string file2 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_labels.csv";
string XOR_DATA = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\XOR_DATA.txt";
string XOR_LABEL = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\XOR_LABEL.txt";

//ReLU activation function
float ReLU(float& innerPotential) {
	return std::fmax(0.0f, innerPotential);
}

float ReLuDerivative(float& innerPotential) {
	return innerPotential > 0.0f ? 1.0f : 0.0f;
}

float softPlus(float& innerPotential) {
	return std::log(1 + std::exp(innerPotential));
}

float softPlusDerivative(float& innerPotential) {
	return 1 / (1 + exp(-innerPotential));
}

float LogicSigmoid(float& innerPotential) {
	return 1 / (1 + exp(-innerPotential));
}

float LogicSigmoidDerivative(float& Y) {
	return Y * (1 - Y);
}

NeuralNetwork::NeuralNetwork(std::string trainingDataFile, std::string labelsFile, std::vector<int> &hiddenNeuronsInLayer){
	//Fetch training data
	readData(trainingDataFile);
	//Fetch labels data
	readExpectedOutput(labelsFile);
	std::vector<float> *weightVec;
	std::vector<float> *biasesVec;
	float bias;
	std::vector<Neuron> Layer;
	//Count of layers
	int Layers = hiddenNeuronsInLayer.size();
	weightVec = new std::vector<float>(INPUTS);
	Weights.reserve(Layers);
	Matrix* M = new Matrix();
	//Input layer outcoming weights
	for (int j = 0; j < hiddenNeuronsInLayer[0]; ++j) {
		for (int i = 0; i < INPUTS; ++i) {
		//RAND_MAX is max number that rand can return so the randomNumber is <-0.05,0.05>
			(*weightVec)[i] =((float)rand() / RAND_MAX - 0.5);
		}
		M->addRow(*weightVec);
	}
	Weights.push_back(M);
	//Each hidden layer outcoming weights 
	for (int layer = 0; layer < Layers-1; ++layer) {
		M = new Matrix();
		weightVec = new std::vector<float>(hiddenNeuronsInLayer[layer]);
		for (int j = 0; j < hiddenNeuronsInLayer[layer + 1]; ++j) {
			for (int i = 0; i < hiddenNeuronsInLayer[layer]; ++i) {
			//RAND_MAX is max number that rand can return so the randomNumber is <-0.05,0.05>
				(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5);
			}
			M->addRow(*weightVec);

		}
		Weights.push_back(M);
	}
	
	//Set biases
	for (int layer = 0; layer < Layers; ++layer) {
		biasesVec = new std::vector<float>(hiddenNeuronsInLayer[layer]);
		for (int i = 0; i < hiddenNeuronsInLayer[layer]; ++i) {
			(*biasesVec)[i] = ((float)rand() / RAND_MAX);
		}
		Biases.push_back(new Matrix(*biasesVec));
	}

	//Set output values to 0
	for (int layer = 0; layer < Layers; ++layer) {
		Y.push_back(new Matrix(hiddenNeuronsInLayer[layer], 1));
	}
}

void NeuralNetwork::readData(string filename) {
	data = vector<vector<float>>(DATA_SIZE, vector<float>(INPUTS, 0));
	//open file for read-only
	fstream file(filename, ios::in);
	string word, line;
	if (!file.is_open()) {
		cout << "File " << filename << " couldn't be open." << endl;
		return;
	}

	int cnt;
	for (int dataSet = 0; getline(file, line); ++dataSet) {
		word.clear();
		cnt = 0;
		for (int i = 0; i < line.size(); ++i) {
			//delimiter
			if (line[i] == ',') {
				this->data[dataSet][cnt] = stof(word);
				++cnt;
				word.clear();
			}
			else {
				word.push_back(line[i]);
			}
		}
		this->data[dataSet][cnt] = stof(word);
		//TODO remove
		if (dataSet == 128) {
			break;
		}
		//break;
	}
	return;
}

void NeuralNetwork::readExpectedOutput(std::string filename) {
	fstream file(filename, ios::in);
	string line;
	if (!file.is_open()) {
		cout << "File " << filename << " couldn't be open." << endl;
		return;
	}

	for (int i = 0; getline(file, line); ++i) {
		this->labels[i] = stof(line);
	}
}

void NeuralNetwork::forwardPropagation(std::vector<float> &inputNeurons) {
	//std::vector<float> InnerPotential = inputNeurons * weights + biases;
	vector<Matrix*> innerPotentials;
	Matrix* innerPotential;
	Matrix input(inputNeurons);
	for (int layer = 0; layer < Weights.size(); ++layer) {
		//W * X + B
		//input layer
		if (layer == 0) {
			innerPotential = *Weights[layer]->dot(*input.transpose()) + *Biases[layer]->transpose();
			innerPotentials.push_back(innerPotential);
		}
		//W * X + B
		else {
			innerPotential = *Weights[layer]->dot(*Y[layer - 1]) +*Biases[layer]->transpose();
			innerPotentials.push_back(innerPotential);
		}
		//Sigma(innerState)
		if (layer < Weights.size() - 1) {
			Y[layer] = innerPotential->computeOutput(Fsigm);
		}
		else {
			Y[layer] = innerPotential->softMax();
		}
	}
	return;
}

vector<Matrix*> NeuralNetwork::backpropagation(float expectedOutput) {
	vector<float> Dkj;
	vector<Matrix*> EderY(Y.size());
	vector<float> sums;
	//derivative = Yj - Dkj ####### output layer 
	EderY[EderY.size() - 1] = Y.back()->subExpectedOutput(expectedOutput);
	for (int i = Y.size() - 2; i >= 0; --i) {
		EderY[i] = EderY[i + 1]->multiply(*Y[i + 1]->computeOutput(Fderivative))->transpose()->dot(*Weights[i + 1])->transpose();
	}
	return EderY;
}

void NeuralNetwork::trainNetwork() {
	//1. forward pass
	//2. backward pass (backpropagation)
	//3. compute Der_Ek/Der_Wji
	//4. sum
	//[Layer][To][From] weights WITH INPUT LAYER!!!!!!!
	vector<Matrix*> Eji;
	vector<Matrix*> dE_dY;
	vector<Matrix*> dE_dW;
	vector<Matrix*> previousEji;
	Matrix* tmp;
	unsigned batchSize = 2;
	float stepSize = 0.5, stepSize0 = stepSize;
	unsigned dataSet;
	for (unsigned cycles = 0; cycles < 200000; ++cycles) {
		for (unsigned k = 0; k < batchSize; ++k) {
			dataSet = rand() % data.size();
			forwardPropagation(data[dataSet]);
			dE_dY = backpropagation(labels[dataSet]);
			for (int layer = 0; layer < Y.size(); ++layer) {
				if (k == 0) {
					if (layer == 0) {
						Eji.push_back(dE_dY[layer]->multiply(*Y[layer]->computeOutput(Fderivative))->dot((data[dataSet])));
					}
					else {
						Eji.push_back(dE_dY[layer]->multiply(*Y[layer]->computeOutput(Fderivative))->dot(*Y[layer - 1]->transpose()));
					}
				}
				else {
					tmp = Eji[layer];
					if (layer == 0) {
						Eji[layer] = *Eji[layer] + (*dE_dY[layer]->multiply(*Y[layer]->computeOutput(Fderivative))->dot((data[dataSet])));
					}
					else {
						Eji[layer] = *Eji[layer] + (*dE_dY[layer]->multiply(*Y[layer]->computeOutput(Fderivative))->dot(*Y[layer - 1]->transpose()));
					}
					delete tmp;
				}
			}
			for (auto p : dE_dY) {
				delete p;
			}
		}
		for (int weight = 0; weight < Weights.size(); ++weight) {
			tmp = Weights[weight];
			Weights[weight] = *Weights[weight] - *(Eji[weight]->multiply(stepSize));
			delete tmp;
		}
		predict();
		stepSize = stepSize0 / (1 + cycles);
	}
	//predict();
	for (auto p : Eji) {
		delete p;
	}
	return;
}

std::vector<float> NeuralNetwork::predict() {
	vector<float> D = data[0];
	//forwardPropagation(data[50]);
	cout << "next cycle" << endl;
	forwardPropagation(D);
	cout << "input " << this->data[0][0] << "," << this->data[0][1] << " has outputs : " << Y[1]->at(0, 0) << " " << Y[1]->at(1, 0) << endl;
	D = data[1];
	forwardPropagation(D);
	cout << "input " << this->data[1][0] << "," << this->data[1][1] << " has outputs: " << Y[1]->at(0, 0) << " " << Y[1]->at(1, 0) << endl;
	D = data[2];
	forwardPropagation(D);
	cout << "input " << this->data[2][0] << "," << this->data[2][1] << " has outputs: " << Y[1]->at(0, 0) << " " << Y[1]->at(1, 0) << endl;
	D = data[3];
	forwardPropagation(D);
	cout << "input " << this->data[3][0] << "," << this->data[3][1] << " has outputs: " << Y[1]->at(0, 0) << " " << Y[1]->at(1, 0) << endl;
	return D;
}

int main() {
	/*vector<int> layers{100, 10, 10};
	NeuralNetwork obj(file1, file2, layers);
	obj.trainNetwork();*/
	vector<int> layers{ 2, 2 };
	NeuralNetwork obj(XOR_DATA, XOR_LABEL, layers);
	cout << "Before:" << endl;
	obj.predict();
	cout << "After:" << endl;
	obj.trainNetwork();
	//obj.predict();
}