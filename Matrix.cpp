#include "Matrix.h"

Matrix::Matrix(std::vector<std::vector<float>>& input) {
	M = std::vector<std::vector<float>>(input);
	rows = input.size();
	cols = input[0].size();
}

Matrix::Matrix(std::vector<float>& input) {
	M = std::vector<std::vector<float>>(1, input);
	rows = 1;
	cols = input.size();
}

void Matrix::addRow(std::vector<float> vec) {
	M.push_back(vec);
	rows += 1;
	if (cols == 0) {
		cols = vec.size();
	}
}

Matrix::Matrix(unsigned R, unsigned C) {
	M = std::vector<std::vector<float>>(R, std::vector<float>(C, 0.0f));
	rows = R;
	cols = C;
}

Matrix::Matrix() {
	M = std::vector<std::vector<float>>();
	rows = 0;
	cols = 0;
}

Matrix Matrix::dot(Matrix& M2) {
	checkDimensions(M2.rows);
	Matrix output = Matrix::Matrix(rows, M2.cols);

	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			for (unsigned k = 0; k < M2.cols; ++k) {
				output.M[i][k] += this->at(i, j) * M2.at(j, k); // this->at(i,j) = this.M[i][j] 
			}
		}
	}
	return output;
}

Matrix Matrix::dot(std::vector<float> vec) {
	checkDimensions(1);
	Matrix output = Matrix::Matrix(rows, vec.size());
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < vec.size(); ++j) {
			output.M[i][j] += this->at(i, 0) * vec[j];
		}
	}
	return output;
}

Matrix Matrix::operator *(Matrix& M2) {
	if (rows != M2.rows || cols != M2.cols)
		throw 789;
	Matrix output = Matrix::Matrix(rows, cols);
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			output.M[i][j] = this->at(i, j) * M2.at(i, j);
		}
	}
	return output;
}

Matrix Matrix::multiply(Matrix& M2) {
	if (rows != M2.rows || cols != M2.cols)
		throw 789;

	Matrix output = Matrix::Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output.M[i][j] = this->at(i, j) * M2.at(i, j);
		}
	}
	return output;
}

Matrix Matrix::multiply(float numb) {
	Matrix output = Matrix::Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output.M[i][j] = this->at(i, j) * numb;
		}
	}
	return output;
}

Matrix Matrix::operator +(Matrix& M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	Matrix output = Matrix::Matrix(rows, M2.cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output.M[i][j] = this->at(i, j) + M2.at(i, j);
		}
	}
	return output;
}

Matrix Matrix::operator -(float D) {
	Matrix output = Matrix::Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output.M[i][j] = this->at(i, j) - D;
		}
	}
	return output;
}

Matrix Matrix::subExpectedOutput(float expected) {
	if (expected + 1 > rows)
		throw 797;

	Matrix output = Matrix::Matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			//if row == expected output, then subtract -1 from the value network computed
			//else subtract -0
			output.M[i][j] = i == expected ? this->M[i][j] - 1.0f : this->M[i][j];
		}
	}
	return output;
}

void Matrix::checkDimensions(unsigned bRows) {
	if (cols != bRows)
		throw 787;
}

void Matrix::checkDimensionsPlus(unsigned bRows, unsigned bCols) {
	if (cols != bCols || rows != bRows)
		throw 795;
}

float Matrix::at(unsigned row, unsigned col) {
	//out of range
	if (row >= rows || col >= cols) throw 708;
	return this->M[row][col];
}

Matrix Matrix::transpose() {
	Matrix output = Matrix::Matrix(cols, rows);
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			output.M[j][i] = M[i][j];
		}
	}
	return output;
}

Matrix Matrix::computeOutput(float (func)(float&)) {
	Matrix output = Matrix::Matrix(rows, cols);
	//Down
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			output.M[i][j] = func(M[i][j]);
		}
	}
	return output;
}

Matrix Matrix::softMax() {
	float sum = 0.0f;
	float numb = 0.0f;
	Matrix output = Matrix::Matrix(rows, cols);
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			numb = exp(M[i][j]);
			output.M[i][j] = numb;
			sum += numb;
		}
	}
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			output.M[i][j] /= sum;
		}
	}
	return output;
}

float Matrix::sum() {
	float sum = 0;
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			sum += M[i][j];
		}
	}
	return sum;
}

Matrix Matrix::operator -(Matrix& M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	Matrix output = Matrix::Matrix(rows, M2.cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			output.M[i][j] = this->M[i][j] - M2.at(i, j);
		}
	}
	return output;
}

Matrix Matrix::oneHot(int label, int classes) {
	Matrix output = Matrix::Matrix(classes, 1);
	output.M[label][0] = 1.0;
	return output;
}

Matrix Matrix::test(int expected) {
	Matrix output = Matrix::Matrix(rows, 1);
	for (int i = 0; i < rows; ++i) {
		//d = 1
		if (i == expected) {
			output.M[i][0] = 1.0 / this->at(i, 0);
		}
		//d = 0
		else {
			output.M[i][0] = 1.0 / (1.0 - this->at(i, 0));
		}
	}
	return output;
}

Matrix Matrix::jacobian() {
	Matrix output = Matrix::Matrix(rows, rows);
	float s1 = 0.0, s2 = 0.0;
	for (unsigned row = 0; row < rows; ++row) {
		for (unsigned col = 0; col < rows; ++col) {
			s1 = this->at(row, 0);
			s2 = this->at(col, 0);
			if (col == row) {
				output.M[row][col] = s1 * (1 - s2);
			}
			else {
				output.M[row][col] = -s1 * s2;
			}
		}
	}
	return output;
}