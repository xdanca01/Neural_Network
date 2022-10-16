#include "NeuralNetwork.hpp"


using namespace std;

string file1 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_vectors.csv";
string file2 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_labels.csv";

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
			(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5);
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
			(*biasesVec)[i] = ((float)rand() / RAND_MAX - 0.5);
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
		break;
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
	Matrix* innerPotential;
	for (int layer = 0; layer < Weights.size(); ++layer) {
		//W * X + B
		//input layer
		if (layer == 0) {
			innerPotential = *(*(Weights[layer]) * inputNeurons) + *(Biases[layer]->transpose());
		}
		//W * X + B
		else {
			innerPotential = *(*(Weights[layer]) * *Y[layer - 1]) + *Biases[layer]->transpose();
		}
		//Sigma(innerState)
		if (layer < Weights.size() - 1) {
			Y[layer] = innerPotential->computeOutput(LogicSigmoid);
		}
		else {
			Y[layer] = innerPotential->softMax();
		}
	}
}



void NeuralNetwork::backpropagation(float expectedOutput) {
	vector<float> Dkj;
	vector<Matrix*> EderY(Y.size());
	vector<float> sums;
	Matrix* M;
	for (int i = Y.size() - 1; i >= 0; --i) {
		//derivative = Yj - Dkj ####### output layer 
		if (i + 1 == Y.size()) {
			EderY[i] = Y[i]->subExpectedOutput(expectedOutput);
		}
		//TODO derivative of softmax
		/*else if (i + 2 == Y.size()) {

		}*/
		//derivative = sum<ReJ->>(dE/dYr * sigma'(Epsilonr) * Wrj)
		else {
			M = *(EderY[i + 1]) * *(Y[i+1]->computeOutput(LogicSigmoidDerivative)->transpose());
			M = *M * *Weights[i+1];
		}
	}
}
/*
vector<vector<float>> NeuralNetwork::gradientDescent(int expectedOutput) {
	vector<vector<float>> EderY(Layers.size());
	//Compute dE/dY
	float lambda = 1.0f;
	for (int i = Layers.size() - 1; i >= 0; --i) {
		//Output layer has different calc
		if (i == Layers.size() - 1) {
			//go throught Y values from softMaxedOutput
			for (int j = 0; j < Layers[i].size(); ++j) {
				//if the index is same as label, then Y - 1, else Y
				EderY[i].push_back(j == expectedOutput ? Layers[i][j].Y - 1.0f : Layers[i][j].Y);
			}
		}
		else {
			for (int j = 0; j < Layers[i].size(); ++j) {
				EderY[i].push_back(0);
				Neuron *neuronAbove;
				//compute der_Ek/der_Yj = SUM_r_e_j->(Ek/Yr)*derivation_of_sigma*Wrj
				for (int r = 0; r < Layers[i + 1].size(); ++r) {
					neuronAbove = &Layers[i + 1][r];
					EderY[i][j] += EderY[i + 1][r] * neuronAbove->LogicSigmoidDerivative(lambda) * neuronAbove->W[j];
				}
			}
		}
	}
	return EderY;
}
*/
void NeuralNetwork::trainNetwork() {
	//1. forward pass
	//2. backward pass (backpropagation)
	//3. compute Der_Ek/Der_Wji
	//4. sum
	//[Layer][To][From] weights WITH INPUT LAYER!!!!!!!
	forwardPropagation(data[0]);
	backpropagation(labels[0]);
	/*
	vector<vector<vector<float>>> E(Layers.size());
	vector<vector<float>> gd;
	float lambda = 1.0f;
	//TODO implement learning in scale of batchSize
	unsigned batchSize = 128;
	int n = data[0].size();

	//Initialize E
	for (int Layer = 0; Layer < Layers.size() - 1; ++Layer) {
		if (Layer == 0) {
			for (int j = 0; j < Layers[0].size(); ++j) {
				E[Layer].push_back(vector<float>());
				for (int i = 0; i < n; ++i) {
					E[0][j].push_back(0.0f);
				}
			}
		}
		//To each layer add its Neuron count vectors
		for (int j = 0; j < Layers[Layer + 1].size(); ++j) {
			E[Layer+1].push_back(vector<float>());
			for (int i = 0; i < Layers[Layer].size(); ++i) {
				//For each To Neuron initialize value 0
				E[Layer+1][j].push_back(0.0f);
			}
		}
	}

	//Training process over all data
	for (unsigned k = 0; k < batchSize; ++k) {
		forwardPropagation(data[k]);
		gd = gradientDescent(labels[k]);
		//Input layer to first hidden layer
		//Neuron From
		for (int i = 0; i < n; ++i) {
			//Neuron To
			for (int j = 0; j < Layers[0].size(); ++j) {
				//clean previous values from previous batches
				if (k == 0) {
					E[0][j][i] = gd[0][j] * Layers[0][j].LogicSigmoidDerivative(lambda) * data[k][i];
				}
				else {
					E[0][j][i] += gd[0][j] * Layers[0][j].LogicSigmoidDerivative(lambda) * data[k][i];
				}
			}
		}
		//Layers without last layer (Output layer doesn't have weights to next neurons)
		for (int Layer = 0; Layer < Layers.size() - 1; ++Layer) {
			//Neuron From
			for (int i = 0; i < Layers[Layer].size(); ++i) {
				//Neuron To
				for (int j = 0; j < Layers[Layer+1].size(); ++j) {
					//clean previous values from previous batches
					if (k == 0) {
						E[Layer + 1][j][i] = gd[Layer + 1][j] * Layers[Layer + 1][j].LogicSigmoidDerivative(lambda) * Layers[Layer][i].Y;
					}
					else {
						E[Layer + 1][j][i] += gd[Layer + 1][j] * Layers[Layer + 1][j].LogicSigmoidDerivative(lambda) * Layers[Layer][i].Y;
					}
				}
			}
		}
	}

	//TODO update weights
	for (int Layer = 0; Layer < Layers.size(); ++Layer) {
		for (int i = 0; i < Layers[Layer].size(); ++i) {
			Layers[Layer][i].updateWeights(E[Layer][i], 1);
		}
	}*/
}/*

std::vector<float> NeuralNetwork::predict() {
	vector<float> output;
	forwardPropagation(data[50]);
	for (Neuron N : Layers.back()) {
		output.push_back(N.Y);
	}
	return output;
}*/

int main() {
	vector<int> layers{100, 20, 10};
	NeuralNetwork obj(file1, file2, layers);
	obj.trainNetwork();
	//obj.predict();
}