#include "Matrix.h"

Matrix::Matrix(std::vector<std::vector<float>> &input) {
	M = new std::vector<std::vector<float>>(input);
	rows = input.size();
	cols = input[0].size();
}

Matrix::Matrix(std::vector<float>& input) {
	M = new std::vector<std::vector<float>>(1, input);
	rows = 1;
	cols = input.size();
}

void Matrix::addRow(std::vector<float>& vec) {
	M->push_back(vec);
	rows += 1;
	if (cols == 0) {
		cols = vec.size();
	}
}

Matrix::Matrix(unsigned R, unsigned C) {
	M = new std::vector<std::vector<float>>(R, std::vector<float>(C, 0.0f));
	rows = R;
	cols = C;
}

Matrix::Matrix() {
	M = new std::vector<std::vector<float>>();
	rows = 0;
	cols = 0;
}

Matrix* Matrix::dot(Matrix &M2) {
	checkDimensions(M2.rows);
	Matrix* output = new Matrix(rows, M2.cols);
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			for (unsigned k = 0; k < M2.cols; ++k) {
				output->M->at(i)[k] += this->at(i, j) * M2.at(j, k);
			}
		}
	}
	return output;
}

Matrix* Matrix::dot(std::vector<float> vec) {
	checkDimensions(1);
	Matrix* output = new Matrix(rows, vec.size());
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < vec.size(); ++j) {
			output->M->at(i)[j] += this->at(i, 0) * vec[j];
		}
	}
	return output;
}

Matrix* Matrix::operator *(Matrix &M2) {
	if (rows != M2.rows || cols != M2.cols)
		throw 789;
	Matrix *output = new Matrix(rows, cols);
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->at(i, j) * M2.at(i, j);
		}
	}
	return output;
}

Matrix* Matrix::operator *(std::vector<float> vec) {
	if (rows != 1 || cols != vec.size())
		throw 789;
	Matrix* output = new Matrix(rows, cols);
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->at(i, j) * vec[j];
		}
	}
	return output;
}

Matrix* Matrix::multiply(Matrix& M2) {
	if (rows != M2.rows || cols != M2.cols)
		throw 789;
	Matrix* output = new Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->at(i, j) * M2.at(i, j);
		}
	}
	return output;
}

Matrix* Matrix::multiply(float numb) {
	Matrix* output = new Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->at(i, j) * numb;
		}
	}
	return output;
}

Matrix* Matrix::multiply(std::vector<float>& vec) {
	if (rows != 1 || cols != vec.size())
		throw 789;
	Matrix* output = new Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->at(i, j) * vec[j];
		}
	}
	return output;
}

Matrix* Matrix::operator +(Matrix &M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	Matrix* output = new Matrix(rows, M2.cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->at(i, j) + M2.at(i, j);
		}
	}
	return output;
}

Matrix* Matrix::operator -(float D) {
	Matrix* output = new Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->at(i, j) - D;
		}
	}
	return output;
}

Matrix* Matrix::subExpectedOutput(float expected) {
	Matrix* output = new Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			//if row == expected output, then subtract -1 from the value network computed
			//else subtract -0
			output->M->at(i)[j] = i == expected ? this->M->at(i)[j] - 1.0f : this->M->at(i)[j];
		}
	}
	return output;
}

void Matrix::checkDimensions(unsigned bRows) {
	if(cols != bRows)
		throw 787;
}

void Matrix::checkDimensionsPlus(unsigned bRows, unsigned bCols) {
	if (cols != bCols || rows != bRows)
		throw 795;
}

float Matrix::at(unsigned row, unsigned col) {
	//out of range
	if (row >= rows || col >= cols) throw 708;
	return (*M)[row][col];
}

Matrix* Matrix::transpose() {
	Matrix* output = new Matrix(cols, rows);
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			output->M->at(j)[i] = (*M)[i][j];
		}
	}
	return output;
}

Matrix* Matrix::computeOutput(float (*func)(float&)){
	Matrix* output = new Matrix(rows, cols);
	//Down
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			output->M->at(i)[j] = func((*M)[i][j]);
		}
	}
	return output;
}

Matrix* Matrix::softMax() {
	float sum = 0.0f;
	float numb = 0.0f;
	Matrix* output = new Matrix(rows, cols);
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			numb = exp((*M)[i][j]);
			output->M->at(i)[j] = numb;
			sum += numb;
		}
	}
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			output->M->at(i)[j] /= sum;
		}
	}
	return output;
}

float Matrix::sum() {
	float sum = 0;
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			sum += (*M)[i][j];
		}
	}
	return sum;
}



Matrix* Matrix::operator -(Matrix& M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	Matrix* output = new Matrix(rows, M2.cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output->M->at(i)[j] = this->M->at(i)[j] - M2.at(i, j);
		}
	}
	return output;
}



