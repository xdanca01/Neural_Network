#include "NeuralNetwork.hpp"
#include "Matrix.h"

#define Fsigm ReLU
#define Fderivative ReLuDerivative

using namespace std;

//string file1 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_vectors.csv";
//string file2 = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_train_labels.csv";
//string XOR_DATA = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\XOR_DATA.txt";
//string XOR_LABEL = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\XOR_LABEL.txt";
//string resultData = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_test_vectors.csv";
//string resultLabels = "C:\\MUNI\\PV021_Neuronove_site\\Projekt\\pv021_project\\data\\fashion_mnist_test_labels.csv";

//string file1 = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\fashion_mnist_train_vectors.csv";
//string file2 = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\fashion_mnist_train_labels.csv";
//string XOR_DATA = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\XOR_DATA.txt";
//string XOR_LABEL = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\XOR_LABEL.txt";
//string resultData = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\fashion_mnist_test_vectors.csv";
//string resultLabels = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\data\\fashion_mnist_test_labels.csv";
//string trainPredictions = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\train_predictions.csv"; // NEW - OUTPUT
//string testPredictions = "C:\\Users\\H514045\\MUNI\\PV021\\pv021_project\\test_predictions.csv"; // NEW - OUTPUT

// Aisa
string file1 = "/home/xmahutov/workspace/pv021_project/data/fashion_mnist_train_vectors.csv";
string file2 = "/home/xmahutov/workspace/pv021_project/data/fashion_mnist_train_labels.csv";
string XOR_DATA = "/home/xmahutov/workspace/pv021_project/data/XOR_DATA.txt";
string XOR_LABEL = "/home/xmahutov/workspace/pv021_project/data/.txt";
string resultData = "/home/xmahutov/workspace/pv021_project/data/fashion_mnist_test_vectors.csv";
string resultLabels = "/home/xmahutov/workspace/pv021_project/data/fashion_mnist_test_labels.csv";
string trainPredictions = "/home/xmahutov/workspace/pv021_project/train_predictions.csv";
string testPredictions = "/home/xmahutov/workspace/pv021_project/test_predictions.csv";

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
	//Count of layers
	int Layers = hiddenNeuronsInLayer.size();
	weightVec = new vector<float>(INPUTS);
	Weights.reserve(Layers);
	Matrix M;

	//Input layer outcoming weights
	for (int j = 0; j < hiddenNeuronsInLayer[0]; ++j) {
		for (int i = 0; i < INPUTS; ++i) {
			//RAND_MAX is max number that rand can return so the randomNumber is <-0.05,0.05>
			(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5f) / 10;
		}
		M.addRow(*weightVec);
	}
	Weights.push_back(M);

	//Each hidden layer outcoming weights 
	for (int layer = 0; layer < Layers - 1; ++layer) {
		// M = Matrix::Matrix();
		M = Matrix();
		weightVec = new vector<float>(hiddenNeuronsInLayer[layer]);
		for (int j = 0; j < hiddenNeuronsInLayer[layer + 1]; ++j) {
			for (int i = 0; i < hiddenNeuronsInLayer[layer]; ++i) {
				//RAND_MAX is max number that rand can return so the randomNumber is <-0.05,0.05>
				(*weightVec)[i] = ((float)rand() / RAND_MAX - 0.5f) / 10;
			}
			M.addRow(*weightVec);
		}
		Weights.push_back(M);
	}

	//Set biases
	for (int layer = 0; layer < Layers; ++layer) {
		biasesVec = new vector<float>(hiddenNeuronsInLayer[layer]);
		for (int i = 0; i < hiddenNeuronsInLayer[layer]; ++i) {
			(*biasesVec)[i] = ((float)rand() / RAND_MAX - 0.5f) / 10;
		}
		Biases.push_back(Matrix(*biasesVec));
	}


	//Set output values to 0
	for (int layer = 0; layer < Layers; ++layer) {
		Y.push_back(Matrix(hiddenNeuronsInLayer[layer], 1));
	}
	for (int thread = 0; thread < THREAD_NUM; ++thread) {
		Y2.push_back(Y);
	}
}

void NeuralNetwork::readData(string filename, bool compare) {

	int sizeToRead = 0;
	if (compare == false) {
		data = vector<vector<float>>(DATA_SIZE, vector<float>(INPUTS, 0));
		sizeToRead = DATA_SIZE;
	}
	else {
		dataForCompare = vector<vector<float>>(PREDICT_SIZE, vector<float>(INPUTS, 0));
		sizeToRead = PREDICT_SIZE;
	}
	//open file for read-only
	fstream file(filename, ios::in);
	if (!file.is_open()) {
		cout << "File " << filename << " couldn't be open." << endl;
		return;
	}

	vector<string> fileData;
	string line;
	int index = 0;
	// Read all data in single thread
	while (getline(file, line)) {
		fileData.emplace_back(line);
		if ((index + 1 == DATA_SIZE && compare == false) || (index + 1 == PREDICT_SIZE && compare == true)) {
			break;
		}
		++index;
	}

#pragma omp parallel for num_threads(THREAD_NUM)
	for (int index = 0; index < (int)fileData.size(); ++index) {
		string word;
		int cnt = 0;

		for (int i = 0; i < (int)(fileData[index]).size(); ++i) { // fileData[index] == line from file
			//delimiter
			if ((fileData[index])[i] == ',') {
				if (compare == false) {
					this->data[index][cnt] = stof(word);
				}
				else {
					this->dataForCompare[index][cnt] = stof(word);
				}
				++cnt;
				word.clear();
			}
			else {
				word.push_back((fileData[index])[i]);
			}
		}
		if (compare == false) {
			this->data[index][cnt] = stof(word);
		}
		else {
			this->dataForCompare[index][cnt] = stof(word);
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

	if (compare == false) {
		labels.reserve(DATA_SIZE);
	}
	else {
		labelsForCompare.reserve(PREDICT_SIZE);
	}

	for (int i = 0; getline(file, line); ++i) {
		// reading predictions (as float)
		if (compare == false) {
			this->labels.push_back(stof(line));
		}
		else {
			this->labelsForCompare.push_back(stof(line));
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
		}
		//W * X + B
		else {
			tmp = Weights[layer].dot(Y[layer - 1]);
			tmp2 = Biases[layer].transpose();
			innerPotential = tmp + tmp2;
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
		}
		//W * X + B
		else {
			tmp = Weights[layer].dot(Y2[thread][layer - 1]);
			tmp2 = Biases[layer].transpose();
			innerPotential = tmp + tmp2;
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
		tmpDelete2 = Y2[thread][i].computeOutput(Fderivative);
		EderY[i] = tmpDelete.multiply(tmpDelete2);
	}
	return EderY;
}

void NeuralNetwork::trainNetworkThreads() {
	//1. forward pass
	//2. backward pass (backpropagation)
	//3. compute Der_Ek/Der_Wji
	//4. sum
	//[Layer][To][From] weights WITH INPUT LAYER!!!!!!!

	//int labelIndex = 0;
	vector<Matrix> Eji;
	vector<Matrix> EjiThreads[THREAD_NUM];
	vector<Matrix> dE_dY_sum;
	vector<Matrix> dE_dY_sumThreads[THREAD_NUM];
	int batchSize = 100;
	float stepSize = 0.001f;
	float stepSize0 = stepSize;
	omp_lock_t layerLockEji[3], layerLockdE[3];
	for (unsigned i = 0; i < Y2[0].size(); ++i) {
		omp_init_lock(&layerLockEji[i]);
		omp_init_lock(&layerLockdE[i]);
	}
	// Number of cycles = for training the neural network 
	// For testing memory - change to 10000+
	time_t cycleTime = std::time(nullptr);
	for (unsigned cycles = 0; cycles < 10000; ++cycles) {
		for (unsigned layer = 0; layer < Y2[0].size(); ++layer) {
			if (layer == 0) {
				Eji.push_back(Matrix(Y2[0][layer].rows, 784));
			}
			else {
				Eji.push_back(Matrix(Y2[0][layer].rows, Y2[0][layer - 1].rows));
			}
			dE_dY_sum.push_back(Matrix(Y2[0][layer].rows, 1));
		}
		for (unsigned i = 0; i < THREAD_NUM; ++i) {
			EjiThreads[i] = Eji;
			dE_dY_sumThreads[i] = dE_dY_sum;
		}
//#pragma omp parallel for num_threads(THREAD_NUM) // REMOVE THIS FOR AISA: >88% (but slow)
		for (int k = 0; k < batchSize; ++k) {
			Matrix tmp;
			Matrix tmp2;
			Matrix tmp3;
			vector<Matrix> dE_dY;
			int dataSet = (cycles * batchSize + k) % DATA_SIZE;
			forwardPropagation(data[dataSet], omp_get_thread_num());
			dE_dY = backpropagation(labels[dataSet], omp_get_thread_num());
			for (unsigned layer = 0; layer < Y2[omp_get_thread_num()].size(); ++layer) {
				if (layer == 0) {
					tmp = dE_dY[layer].dot(data[dataSet]);
					EjiThreads[omp_get_thread_num()][layer] = EjiThreads[omp_get_thread_num()][layer] + tmp;
					dE_dY_sumThreads[omp_get_thread_num()][layer] = dE_dY_sumThreads[omp_get_thread_num()][layer] + dE_dY[layer];
				}
				else {
					tmp = Y[layer - 1].transpose();
					tmp2 = dE_dY[layer].dot(tmp);
					EjiThreads[omp_get_thread_num()][layer] = EjiThreads[omp_get_thread_num()][layer] + tmp2;
					dE_dY_sumThreads[omp_get_thread_num()][layer] = dE_dY_sumThreads[omp_get_thread_num()][layer] + dE_dY[layer];
				}
			}
		}
		for (int i = 0; i < THREAD_NUM; ++i) {
			for (int layer = 0; layer < (int)Y2[0].size(); ++layer) {
				Eji[layer] = Eji[layer] + EjiThreads[omp_get_thread_num()][layer];
				dE_dY_sum[layer] = dE_dY_sum[layer] + dE_dY_sumThreads[omp_get_thread_num()][layer];
			}
		}
#pragma omp parallel for num_threads(THREAD_NUM)
		for (int layer = 0; layer < (int)Weights.size(); ++layer) {
			Matrix tmp;
			Matrix tmp2;
			Matrix tmp3;
			if (cycles) {
				tmp2 = Eji[layer].multiply(stepSize / batchSize);
				Weights[layer] = Weights[layer] - tmp2;
				tmp2 = dE_dY_sum[layer].multiply(stepSize / batchSize);
				tmp3 = tmp2.transpose();
				Biases[layer] = Biases[layer] - tmp3;
			}
			else {
				tmp2 = Eji[layer].multiply(stepSize / batchSize);
				Weights[layer] = Weights[layer] - tmp2;
				tmp2 = dE_dY_sum[layer].multiply(stepSize / batchSize);
				tmp3 = tmp2.transpose();
				Biases[layer] = Biases[layer] - tmp3;
			}
		}
		Eji.clear();
		dE_dY_sum.clear();
		int pred = cycles % 600;
		if (pred == 599) {
			time_t newTime = std::time(nullptr);
			cout << newTime - cycleTime << endl;
			predict();
			//writeLabel(trainPredictions, labelIndex); // just test
			//labelIndex += PREDICT_SIZE;
			cycleTime = std::time(nullptr);
		}	
	}

	//outputToFile(testPredictions, this->labels); // just test
	writeLabelToFile(testPredictions, this->labels, DATA_SIZE); // NEW 
	writeLabelToFile(trainPredictions, labelsForCompare, PREDICT_SIZE); // NEW
	//predict();
	return;
}

//void NeuralNetwork::outputToFile(string filename, vector<float> writeData) {
//	ofstream file;
//	file.open(filename, ofstream::trunc); // ::trunc = deletes the file when opened
//
//	for (int i = 0; i + 1 <= DATA_SIZE; ++i) {
//		file << writeData[i] << endl;
//		cout << "this->labels[" << i << "]: " << this->labels[i] << endl;
//	}
//	file.close();
//}

//void NeuralNetwork::writeLabel(string filename, int labelIndex) {
//	fstream file(filename, ios::out); 
//
//	for (int i = labelIndex; i < labelIndex + PREDICT_SIZE; i++) {
//		file << labelsForCompare[i] << endl;
//		cout << "lablesForCompare[" << i << "]: " << labelsForCompare[i] << endl;
//	}
//	file.close();
//}

// Write data to output files
void NeuralNetwork::writeLabelToFile(string filename, vector<float> writeData, int dataSize) {
	fstream file;
	file.open(filename, ios::out);

	if (!file.is_open()) {
		cout << "File " << filename << " couldn't be open." << endl;
		return;
	}

	for (int i = 0; i + 1 <= dataSize; ++i) {
		file << writeData[i] << endl;
	}
	file.close();
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
	}
	cout << "Succesfully predicted labels: " << (float)sameLabels / (float)dataForCompare.size() << endl;
}

int main() {
	vector<int> layers{ 190, 55, 10 };
	NeuralNetwork obj(file1, file2, layers);
	/*vector<int> layers{20, 2};
	NeuralNetwork obj(XOR_DATA, XOR_LABEL, layers);*/

	cout << "Before:" << endl;
	obj.predict();
	obj.trainNetworkThreads();
}