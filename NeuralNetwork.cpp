#include "NeuralNetwork.hpp"


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
			(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5)/10;
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
				(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5)/10;
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

vector<Matrix*> NeuralNetwork::forwardPropagation(std::vector<float> &inputNeurons) {
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
			innerPotential = *Weights[layer]->dot(*input.transpose()) + *Biases[layer]->transpose();
			innerPotentials.push_back(innerPotential);
		}
		//Sigma(innerState)
		if (layer < Weights.size() - 1) {
			Y[layer] = innerPotential->computeOutput(ReLU);
		}
		else {
			Y[layer] = innerPotential->softMax();
		}
	}
	return innerPotentials;
}



vector<Matrix*> NeuralNetwork::backpropagation(float expectedOutput, vector<Matrix*> innerPotentials) {
	vector<float> Dkj;
	vector<Matrix*> EderY(Y.size());
	vector<float> sums;
	Matrix* M;
	//derivative = Yj - Dkj ####### output layer 
	EderY[EderY.size() - 1] = Y.back()->subExpectedOutput(expectedOutput);
	for (int i = Y.size() - 2; i >= 0; --i) {
		//TODO derivative of softmax
		/*else if (i + 2 == Y.size()) {

		}*/
		//derivative = sum<ReJ->>(dE/dYr * sigma'(Epsilonr) * Wrj)
		EderY[i] = EderY[i + 1]->multiply(*innerPotentials[i + 1]->computeOutput(ReLuDerivative))->transpose()->dot(*Weights[i + 1])->transpose();
	}
	return EderY;
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
	vector<Matrix*> Eji;
	vector<Matrix*> innerPotentials;
	vector<Matrix*> dE_dY;
	vector<Matrix*> dE_dW;
	unsigned batchSize = 4;
	float stepSize = 0.05;
	for (unsigned cycles = 0; cycles < 1500; ++cycles) {
		for (unsigned k = 0; k < batchSize; ++k) {
			innerPotentials = forwardPropagation(data[k]);
			dE_dY = backpropagation(labels[k], innerPotentials);
			for (int layer = 0; layer < Y.size(); ++layer) {
				if (k == 0) {
					if (layer == 0) {
						Eji.push_back(dE_dY[layer]->multiply(*innerPotentials[layer]->computeOutput(ReLuDerivative))->dot((data[k])));
					}
					else {
						Eji.push_back(dE_dY[layer]->multiply(*innerPotentials[layer]->computeOutput(ReLuDerivative))->dot(*Y[layer - 1]->transpose()));
					}
				}
				else {
					if (layer == 0) {
						Eji[layer] = *Eji[layer] + *dE_dY[layer]->multiply(*innerPotentials[layer]->computeOutput(ReLuDerivative))->dot((data[k]));
					}
					else {
						Eji[layer] = *Eji[layer] + *dE_dY[layer]->multiply(*innerPotentials[layer]->computeOutput(ReLuDerivative))->dot(*Y[layer - 1]->transpose());
					}
				}
			}
		}
		for (int weight = 0; weight < Weights.size(); ++weight) {
			Weights[weight] = *Weights[weight] - *(Eji[weight]->multiply(stepSize));
		}
	}
	predict();
	return;
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
 */
std::vector<float> NeuralNetwork::predict() {
	vector<float> data{0,0};
	//forwardPropagation(data[50]);
	forwardPropagation(data);
	cout << "input 0,0 has output 0 with probability:" <<Y[1]->at(0, 0) << endl;
	data = { 0,1 };
	forwardPropagation(data);
	cout << "input 0,1 has output 1 with probability:" << Y[1]->at(1, 0) << endl;
	data = { 1,0 };
	forwardPropagation(data);
	cout << "input 1,0 has output 1 with probability:" << Y[1]->at(1, 0) << endl;
	data = { 1,1 };
	forwardPropagation(data);
	cout << "input 1,1 has output 0 with probability:" << Y[1]->at(0, 0) << endl;
	return data;
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