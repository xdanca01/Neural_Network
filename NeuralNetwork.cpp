#include "NeuralNetwork.hpp"

#define Fsigm ReLU
#define Fderivative ReLuDerivative

using namespace std;

string file1 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_vectors.csv";
string file2 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_labels.csv";
string XOR_DATA = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\XOR_DATA.txt";
string XOR_LABEL = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\XOR_LABEL.txt";
string resultData = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_test_vectors.csv";
string resultLabels = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_test_labels.csv";
//string file1 = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\fashion_mnist_train_vectors.csv";
//string file2 = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\fashion_mnist_train_labels.csv";
//string XOR_DATA = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\XOR_DATA.txt";
//string XOR_LABEL = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\XOR_LABEL.txt";

//ReLU activation function
float ReLU(float& innerPotential) {
	return fmax(0.0f, innerPotential);
}

float ReLuDerivative(float& innerPotential) {
	return innerPotential > 0.0f ? 1.0f : 0.0f;
}

float softPlus(float& innerPotential) {
	return log(1 + exp(innerPotential));
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

unsigned NeuralNetwork::argMax() {
	return Y.back().argMax();
}

unsigned NeuralNetwork::argMaxThreads(int thread) {
	return Y2[thread].back().argMax();
}

NeuralNetwork::NeuralNetwork(string trainingDataFile, string labelsFile, vector<int>& hiddenNeuronsInLayer) {
	//Fetch training data
	readData(trainingDataFile, false);
	//Fetch predict data
	readData(resultData, true);
	//Fetch training labels
	readExpectedOutput(labelsFile, false);
	//Fetch predict labels
	readExpectedOutput(resultLabels, true);
	vector<float>* weightVec;
	vector<float>* biasesVec;
	vector<Neuron> Layer;
	//Count of layers
	int Layers = hiddenNeuronsInLayer.size();
	weightVec = new vector<float>(INPUTS);
	Weights.reserve(Layers);
	Matrix M = Matrix::Matrix();

	//Input layer outcoming weights
	for (int j = 0; j < hiddenNeuronsInLayer[0]; ++j) {
		for (int i = 0; i < INPUTS; ++i) {
			//RAND_MAX is max number that rand can return so the randomNumber is <-0.05,0.05>
			(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5f)/10;
		}
		M.addRow(*weightVec);
	}
	Weights.push_back(M);

	//Each hidden layer outcoming weights 
	for (int layer = 0; layer < Layers - 1; ++layer) {
		M = Matrix::Matrix();
		weightVec = new vector<float>(hiddenNeuronsInLayer[layer]);
		for (int j = 0; j < hiddenNeuronsInLayer[layer + 1]; ++j) {
			for (int i = 0; i < hiddenNeuronsInLayer[layer]; ++i) {
				//RAND_MAX is max number that rand can return so the randomNumber is <-0.05,0.05>
				(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5f)/10;
			}
			M.addRow(*weightVec);
		}
		Weights.push_back(M);
	}

	//Set biases
	for (int layer = 0; layer < Layers; ++layer) {
		biasesVec = new vector<float>(hiddenNeuronsInLayer[layer]);
		for (int i = 0; i < hiddenNeuronsInLayer[layer]; ++i) {
			(*biasesVec)[i] = ((float)rand() / RAND_MAX - 0.5f)/10;
		}
		Biases.push_back(Matrix::Matrix(*biasesVec));
	}

	
	//Set output values to 0
	for (int layer = 0; layer < Layers; ++layer) {
		Y.push_back(Matrix::Matrix(hiddenNeuronsInLayer[layer], 1));
	}
	for (int thread = 0; thread < THREAD_NUM; ++thread) {
		Y2.push_back(Y);
	}
}

void NeuralNetwork::readData(string filename, bool compare) {
	
	if (compare == false) {
		data = vector<vector<float>>(DATA_SIZE, vector<float>(INPUTS, 0));
	}
	else {
		dataForCompare = vector<vector<float>>(PREDICT_SIZE, vector<float>(INPUTS, 0));
	}
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
		for (unsigned i = 0; i < line.size(); ++i) {
			//delimiter
			if (line[i] == ',') {
				if (compare == false) {
					this->data[dataSet][cnt] = stof(word);
				}
				else {
					this->dataForCompare[dataSet][cnt] = stof(word);
				}
				++cnt;
				word.clear();
			}
			else {
				word.push_back(line[i]);
			}
		}
		if (compare == false) {
			this->data[dataSet][cnt] = stof(word);
		}
		else {
			this->dataForCompare[dataSet][cnt] = stof(word);
		}
		//TODO - remove for dataset, shorter reading only for testing
		if ((dataSet+1 == DATA_SIZE && compare == false) || (dataSet + 1 == PREDICT_SIZE && compare == true)) {
			break;
		}
	}
	file.close();
	return;
}

void NeuralNetwork::readExpectedOutput(string filename, bool compare) {
	fstream file(filename, ios::in);
	string line;
	if (!file.is_open()) {
		cout << "File " << filename << " couldn't be open." << endl;
		return;
	}

	for (int i = 0; getline(file, line); ++i) {
		// reading predictions (as float)
		if (compare == false) {
			this->labels[i] = stof(line);
		}
		else {
			this->labelsForCompare[i] = stof(line);
		}
		if ((i + 1 == DATA_SIZE && compare == false) || (i + 1 == PREDICT_SIZE && compare == true)) {
			break;
		}
	}
	file.close();
}

void NeuralNetwork::forwardPropagation(vector<float>& inputNeurons) {
	//vector<float> InnerPotential = inputNeurons * weights + biases;
	Matrix innerPotential;
	Matrix tmp;
	Matrix tmp2;
	Matrix input(inputNeurons);

	for (unsigned layer = 0; layer < Weights.size(); ++layer) {
		//W * X + B
		//input layer
		if (layer == 0) {
			tmp = input.transpose(); 
			tmp2 = Weights[layer].dot(tmp);
			tmp = Biases[layer].transpose();
			innerPotential = tmp2 + tmp;
			//innerPotential = *Weights[layer]->dot(*input.transpose()) + *Biases[layer]->transpose();
		}
		//W * X + B
		else {
			tmp = Weights[layer].dot(Y[layer - 1]);
			tmp2 = Biases[layer].transpose();
			innerPotential = tmp + tmp2;
			//innerPotential = *Weights[layer]->dot(*Y[layer - 1]) +*Biases[layer]->transpose();
		}
		//Sigma(innerState)
		tmp = Y[layer];
		if (layer < Weights.size() - 1) {
			Y[layer] = innerPotential.computeOutput(Fsigm);
		}
		else {
			Y[layer] = innerPotential.softMax();
		}
	}
	return;
}

void NeuralNetwork::forwardPropagation(vector<float>& inputNeurons, int thread) {
	//vector<float> InnerPotential = inputNeurons * weights + biases;
	Matrix innerPotential;
	Matrix tmp;
	Matrix tmp2;
	Matrix input(inputNeurons);

	for (unsigned layer = 0; layer < Weights.size(); ++layer) {
		//W * X + B
		//input layer
		if (layer == 0) {
			tmp = input.transpose();
			tmp2 = Weights[layer].dot(tmp);
			tmp = Biases[layer].transpose();
			innerPotential = tmp2 + tmp;
			//innerPotential = *Weights[layer]->dot(*input.transpose()) + *Biases[layer]->transpose();
		}
		//W * X + B
		else {
			tmp = Weights[layer].dot(Y2[thread][layer - 1]);
			tmp2 = Biases[layer].transpose();
			innerPotential = tmp + tmp2;
			//innerPotential = *Weights[layer]->dot(*Y[layer - 1]) +*Biases[layer]->transpose();
		}
		//Sigma(innerState)
		if (layer < Weights.size() - 1) {
			Y2[thread][layer] = innerPotential.computeOutput(Fsigm);
		}
		else {
			Y2[thread][layer] = innerPotential.softMax();
		}
	}
	return;
}

vector<Matrix> NeuralNetwork::backpropagation(float expectedOutput) {
	vector<float> Dkj;
	vector<Matrix> EderY(Y.size());
	vector<float> sums;
	Matrix tmpDelete;
	Matrix tmpDelete2;

	//derivative = Yj - Dkj ####### output layer
	//This computes derivation of mistake E with respect to innerpotential of output layer
	EderY[EderY.size() - 1] = Y.back().subExpectedOutput(expectedOutput);
	//Thic computes derivation of mistake E with respect to innerpotential of all other layers
	for (int i = Y.size() - 2; i >= 0; --i) {
		tmpDelete = EderY[i + 1].transpose();
		tmpDelete2 = tmpDelete.dot(Weights[i + 1]);
		tmpDelete = tmpDelete2.transpose();
		tmpDelete2 = Y[i].computeOutput(Fderivative);
		EderY[i] = tmpDelete.multiply(tmpDelete2);
		//EderY[i] = EderY[i + 1]->transpose()->dot(*Weights[i + 1])->transpose()->multiply(*Y[i]->computeOutput(Fderivative));
	}
	return EderY;
}

vector<Matrix> NeuralNetwork::backpropagation(float expectedOutput, int thread) {
	vector<float> Dkj;
	vector<Matrix> EderY(Y2[thread].size());
	vector<float> sums;
	Matrix tmpDelete;
	Matrix tmpDelete2;

	//derivative = Yj - Dkj ####### output layer
	//This computes derivation of mistake E with respect to innerpotential of output layer
	EderY[EderY.size() - 1] = Y2[thread].back().subExpectedOutput(expectedOutput);
	//Thic computes derivation of mistake E with respect to innerpotential of all other layers
	for (int i = Y2[thread].size() - 2; i >= 0; --i) {
		tmpDelete = EderY[i + 1].transpose();
		tmpDelete2 = tmpDelete.dot(Weights[i + 1]);
		tmpDelete = tmpDelete2.transpose();
		/***********It should be faster, but it's not WTF********/
		//tmpDelete = EderY[i + 1].transposeDotTranspose(Weights[i + 1]);
		tmpDelete2 = Y2[thread][i].computeOutput(Fderivative);
		EderY[i] = tmpDelete.multiply(tmpDelete2);
		//EderY[i] = EderY[i + 1]->transpose()->dot(*Weights[i + 1])->transpose()->multiply(*Y[i]->computeOutput(Fderivative));
	}
	return EderY;
}

void NeuralNetwork::trainNetwork() {
	//1. forward pass
	//2. backward pass (backpropagation)
	//3. compute Der_Ek/Der_Wji
	//4. sum
	//[Layer][To][From] weights WITH INPUT LAYER!!!!!!!

	vector<Matrix> Eji;
	vector<Matrix> dE_dY;
	vector<Matrix> dE_dY_sum;
	//vector<Matrix*> previousEji;
	//vector<Matrix*> previousdE_dY_sum;
	Matrix tmp;
	Matrix tmp2;
	Matrix tmp3;
	unsigned batchSize = 128;
	float stepSize = 0.001f, stepSize0 = stepSize;

	// Number of cycles = for training the neural network 
	// For testing memory - change to 10000+

	for (unsigned cycles = 0; cycles < 100; ++cycles) {
		for (unsigned k = 0; k < batchSize; ++k) {
			//thread safe random 
			std::random_device rd;
			std::uniform_int_distribution<int> dist(0, data.size()-1);
			int dataSet = dist(rd);
			forwardPropagation(data[dataSet]);
			dE_dY = backpropagation(labels[dataSet]);
			for (unsigned layer = 0; layer < Y.size(); ++layer) {
				if (k == 0) {
					if (layer == 0) {
						Eji.push_back(dE_dY[layer].dot(data[dataSet]));
						dE_dY_sum.push_back(dE_dY[layer]);
					}
					else {
						tmp = Y[layer - 1].transpose();
						Eji.push_back(dE_dY[layer].dot(tmp));
						//Eji.push_back(dE_dY[layer]->dot(*Y[layer - 1]->transpose()));
						dE_dY_sum.push_back(dE_dY[layer]);
					}
				}
				else {
					if (layer == 0) {
						tmp = dE_dY[layer].dot(data[dataSet]);
						tmp2 = Eji[layer];
						Eji[layer] = Eji[layer] + tmp;
						//Eji[layer] = *Eji[layer] + (*dE_dY[layer]->dot(data[dataSet]));
						tmp = dE_dY_sum[layer];
						dE_dY_sum[layer] = dE_dY_sum[layer] + dE_dY[layer];
					}
					else {
						tmp = Y[layer - 1].transpose();
						tmp2 = dE_dY[layer].dot(tmp);
						tmp = Eji[layer];
						Eji[layer] = Eji[layer] + tmp2;
						tmp = dE_dY_sum[layer];
						//Eji[layer] = *Eji[layer] + (*dE_dY[layer]->dot(*Y[layer - 1]->transpose()));
						dE_dY_sum[layer] = dE_dY_sum[layer] + dE_dY[layer];
					}
				}
			}
		}
		for (unsigned layer = 0; layer < Weights.size(); ++layer) {
			tmp = Weights[layer];
			if (cycles) {
				tmp2 = Eji[layer].multiply(stepSize / batchSize);
				Weights[layer] = Weights[layer] - tmp2;
				tmp = Biases[layer];
				tmp2 = dE_dY_sum[layer].multiply(stepSize / batchSize);
				tmp3 = tmp2.transpose();
				Biases[layer] = Biases[layer] - tmp3;

				/*Biases[layer] = *tmp2 - *previousdE_dY_sum[layer]->multiply(0.3)->multiply(stepSize / batchSize);
				delete tmp, previousdE_dY_sum[layer];
				previousdE_dY_sum[layer] = dE_dY_sum[layer]->transpose();*/
				//Weights[layer] = *Weights[layer] - *(Eji[layer]->multiply(stepSize)->multiply(1.0 / batchSize));// - *previousEji[layer]->multiply(0.1));
			}
			else {
				/*tmp2 = Eji[layer]->multiply(stepSize / batchSize);
				Weights[layer] = *Weights[layer] - *tmp2;
				delete tmp2, tmp;
				previousEji.push_back(Eji[layer]);
				//Weights[layer] = *Weights[layer] - *(Eji[layer]->multiply(stepSize)->multiply(1.0 / batchSize));*/

				tmp2 = Eji[layer].multiply(stepSize / batchSize);
				Weights[layer] = Weights[layer] - tmp2;	
				tmp = Biases[layer];
				tmp2 = dE_dY_sum[layer].multiply(stepSize / batchSize);
				tmp3 = tmp2.transpose();
				Biases[layer] = Biases[layer] - tmp3;

				/*tmp = Biases[layer];
				tmp2 = dE_dY_sum[layer]->multiply(stepSize / batchSize);
				tmp3 = tmp2->transpose();
				Biases[layer] = *Biases[layer] - *tmp3;
				delete tmp, tmp2, tmp3;
				previousdE_dY_sum.push_back(dE_dY_sum[layer]->transpose());*/
			}
			//Biases[layer] = *Biases[layer] - *(dE_dY_sum[layer]->multiply(stepSize / batchSize))->transpose();
			//Weights[layer] = *Weights[layer] - *(Eji[layer]->multiply(stepSize / batchSize));
		}
		Eji.clear();
		dE_dY_sum.clear();
		predict();
		stepSize = stepSize0;
	}
	//predict();
	return;
}

void NeuralNetwork::trainNetworkThreads() {
	//1. forward pass
	//2. backward pass (backpropagation)
	//3. compute Der_Ek/Der_Wji
	//4. sum
	//[Layer][To][From] weights WITH INPUT LAYER!!!!!!!

	vector<Matrix> Eji;
	vector<Matrix> EjiThreads[THREAD_NUM];
	vector<Matrix> dE_dY_sum;
	vector<Matrix> dE_dY_sumThreads[THREAD_NUM];
	//vector<Matrix*> previousEji;
	//vector<Matrix*> previousdE_dY_sum;
	int batchSize = 100;
	float stepSize = 0.01f, stepSize0 = stepSize;
	omp_lock_t layerLockEji[3], layerLockdE[3];
	for (unsigned i = 0; i < Y2[0].size(); ++i) {
		omp_init_lock(&layerLockEji[i]);
		omp_init_lock(&layerLockdE[i]);
	}
	// Number of cycles = for training the neural network 
	// For testing memory - change to 10000+
	time_t cycleTime = std::time(nullptr);
	for (unsigned cycles = 0; cycles < 10000; ++cycles) {
		cout << cycles << endl;
		for (unsigned layer = 0; layer < Y2[0].size(); ++layer) {
			if (layer == 0) {
				Eji.push_back(Matrix(Y2[0][layer].rows, 784));
			}
			else {
				Eji.push_back(Matrix(Y2[0][layer].rows, Y2[0][layer-1].rows));
			}
			dE_dY_sum.push_back(Matrix(Y2[0][layer].rows, 1));
		}
		for (unsigned i = 0; i < THREAD_NUM; ++i) {
			EjiThreads[i] = Eji;
			dE_dY_sumThreads[i] = dE_dY_sum;
		}
	#pragma omp parallel for num_threads(THREAD_NUM)
		for (int k = 0; k < batchSize; ++k) {
			Matrix tmp;
			Matrix tmp2;
			Matrix tmp3;
			vector<Matrix> dE_dY;
			//thread safe random 
			//std::random_device rd;
			//std::uniform_int_distribution<int> dist(0, data.size()-1);
			//int dataSet = dist(rd);
			int dataSet = (cycles * batchSize + k) % DATA_SIZE;
			forwardPropagation(data[dataSet], omp_get_thread_num());
			dE_dY = backpropagation(labels[dataSet], omp_get_thread_num());
			for (unsigned layer = 0; layer < Y2[omp_get_thread_num()].size(); ++layer) {
				if (layer == 0) {
					tmp = dE_dY[layer].dot(data[dataSet]);
					/*omp_set_lock(&layerLockEji[layer]);
					Eji[layer] = Eji[layer] + tmp;
					omp_unset_lock(&layerLockEji[layer]);*/
					EjiThreads[omp_get_thread_num()][layer] = EjiThreads[omp_get_thread_num()][layer] + tmp;
					//Eji[layer] = *Eji[layer] + (*dE_dY[layer]->dot(data[dataSet]));
					/*omp_set_lock(&layerLockdE[layer]);
					dE_dY_sum[layer] = dE_dY_sum[layer] + dE_dY[layer];
					omp_unset_lock(&layerLockdE[layer]);*/
					dE_dY_sumThreads[omp_get_thread_num()][layer] = dE_dY_sumThreads[omp_get_thread_num()][layer] + dE_dY[layer];
				}
				else {
					tmp = Y[layer - 1].transpose();
					tmp2 = dE_dY[layer].dot(tmp);
					/*omp_set_lock(&layerLockEji[layer]);
					Eji[layer] = Eji[layer] + tmp2;
					omp_unset_lock(&layerLockEji[layer]);
					//Eji[layer] = *Eji[layer] + (*dE_dY[layer]->dot(*Y[layer - 1]->transpose()));
					omp_set_lock(&layerLockdE[layer]);
					dE_dY_sum[layer] = dE_dY_sum[layer] + dE_dY[layer];
					omp_unset_lock(&layerLockdE[layer]);*/
					EjiThreads[omp_get_thread_num()][layer] = EjiThreads[omp_get_thread_num()][layer] + tmp2;
					dE_dY_sumThreads[omp_get_thread_num()][layer] = dE_dY_sumThreads[omp_get_thread_num()][layer] + dE_dY[layer];
				}
			}
		}
		for (int i = 0; i < THREAD_NUM; ++i) {
			for (int layer = 0; layer < Y2[0].size(); ++layer) {
				Eji[layer] = Eji[layer] + EjiThreads[omp_get_thread_num()][layer];
				dE_dY_sum[layer] = dE_dY_sum[layer] + dE_dY_sumThreads[omp_get_thread_num()][layer];
			}
		}
	#pragma omp parallel for num_threads(THREAD_NUM)
		for (int layer = 0; layer < Weights.size(); ++layer) {
			Matrix tmp;
			Matrix tmp2;
			Matrix tmp3;
			if (cycles) {
				tmp2 = Eji[layer].multiply(stepSize / batchSize);
				Weights[layer] = Weights[layer] - tmp2;
				tmp2 = dE_dY_sum[layer].multiply(stepSize / batchSize);
				tmp3 = tmp2.transpose();
				Biases[layer] = Biases[layer] - tmp3;

				/*Biases[layer] = *tmp2 - *previousdE_dY_sum[layer]->multiply(0.3)->multiply(stepSize / batchSize);
				delete tmp, previousdE_dY_sum[layer];
				previousdE_dY_sum[layer] = dE_dY_sum[layer]->transpose();*/
				//Weights[layer] = *Weights[layer] - *(Eji[layer]->multiply(stepSize)->multiply(1.0 / batchSize));// - *previousEji[layer]->multiply(0.1));
			}
			else {
				/*tmp2 = Eji[layer]->multiply(stepSize / batchSize);
				Weights[layer] = *Weights[layer] - *tmp2;
				delete tmp2, tmp;
				previousEji.push_back(Eji[layer]);
				//Weights[layer] = *Weights[layer] - *(Eji[layer]->multiply(stepSize)->multiply(1.0 / batchSize));*/

				tmp2 = Eji[layer].multiply(stepSize / batchSize);
				Weights[layer] = Weights[layer] - tmp2;
				tmp2 = dE_dY_sum[layer].multiply(stepSize / batchSize);
				tmp3 = tmp2.transpose();
				Biases[layer] = Biases[layer] - tmp3;

				/*tmp = Biases[layer];
				tmp2 = dE_dY_sum[layer]->multiply(stepSize / batchSize);
				tmp3 = tmp2->transpose();
				Biases[layer] = *Biases[layer] - *tmp3;
				delete tmp, tmp2, tmp3;
				previousdE_dY_sum.push_back(dE_dY_sum[layer]->transpose());*/
			}
			//Biases[layer] = *Biases[layer] - *(dE_dY_sum[layer]->multiply(stepSize / batchSize))->transpose();
			//Weights[layer] = *Weights[layer] - *(Eji[layer]->multiply(stepSize / batchSize));
		}
		Eji.clear();
		dE_dY_sum.clear();
		int pred = cycles % 600;
		if (pred == 599) {
			stepSize = 0.001;
			predict();
		}
		if (pred == 10) {
			time_t newTime = std::time(nullptr);
			cout << newTime - cycleTime << endl;
			cycleTime = newTime;
		}
	}
	//predict();
	return;
}

void NeuralNetwork::predict() {
	unsigned sameLabels = 0;
omp_lock_t writelock;
omp_init_lock(&writelock);
#pragma omp parallel for num_threads(THREAD_NUM)
	for (int i = 0; i < PREDICT_SIZE; ++i) {
		unsigned label;
		forwardPropagation(dataForCompare[i], omp_get_thread_num());
		label = argMaxThreads(omp_get_thread_num());
		if (label == labelsForCompare[i]) {
			omp_set_lock(&writelock);
			++sameLabels;
			omp_unset_lock(&writelock);
		}
		//cout << omp_get_thread_num() << endl;
	}
	cout << "Succesfully predicted labels: " << (float)sameLabels / dataForCompare.size() << endl;
}

int main() {
	vector<int> layers{128, 50, 10};
	NeuralNetwork obj(file1, file2, layers);
	/*vector<int> layers{20, 2};
	NeuralNetwork obj(XOR_DATA, XOR_LABEL, layers);*/
	cout << "Before:" << endl;
	obj.predict();
	//obj.trainNetwork();
	obj.trainNetworkThreads();
	//obj.predict();
}