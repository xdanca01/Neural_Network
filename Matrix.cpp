#include "Matrix.h"

Matrix::Matrix(const std::vector<std::vector<float>>& input) : M(input), rows(input.size()), cols(input[0].size()) {}

Matrix::Matrix(std::vector<std::vector<float>>&& input) : M(std::move(input)), rows(input.size()), cols(input[0].size()) {}

Matrix::Matrix(const std::vector<float>& input) : M(1, input), rows(1), cols(input.size()) {}

Matrix::Matrix(std::vector<float>&& input) : M(1, std::move(input)), rows(1), cols(input.size()) {}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
	if (this != &other) {
		M = std::move(other.M);
		rows = other.rows;
		cols = other.cols;
		other.rows = 0;
		other.cols = 0;
	}
	return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
	: M(std::move(other.M)), rows(other.rows), cols(other.cols) {
	other.rows = 0;
	other.cols = 0;
}
Matrix::Matrix(const Matrix& other) noexcept : M(other.M), rows(other.rows), cols(other.cols) {}


Matrix& Matrix::operator=(const Matrix& other) noexcept {
	if (this != &other) {
		M = other.M;
		rows = other.rows;
		cols = other.cols;
	}
	return *this;
}



void Matrix::addRow(std::vector<float>&& vec) {
	if (cols == 0) {
		cols = vec.size();
	}
	M.emplace_back(std::move(vec));
	rows += 1;
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
	Matrix output = Matrix(rows, M2.cols);

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

Matrix Matrix::dot(std::vector<float>& vec) {
	checkDimensions(1);
	Matrix output = Matrix(rows, vec.size());
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < vec.size(); ++j) {
			output.M[i][j] += this->at(i, 0) * vec[j];
		}
	}
	return output;
}

Matrix& Matrix::operator *(Matrix& M2) {
	if (rows != M2.rows || cols != M2.cols)
		throw 789;
	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] *= M2.at(i, j);
		}
	}
	return *this;
}

Matrix& Matrix::operator *(float numb) {
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] *= numb;
		}
	}
	return *this;
}

Matrix& Matrix::multiply(Matrix& M2) {
	if (rows != M2.rows || cols != M2.cols)
		throw 789;
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] *= M2.at(i, j);
		}
	}
	return *this;
}

Matrix& Matrix::multiply(float numb) {
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] *= numb;
		}
	}
	return *this;
}

Matrix& Matrix::operator +(Matrix& M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] += M2.at(i, j);
		}
	}
	return *this;
}

Matrix& Matrix::operator +=(Matrix& M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] += M2.at(i, j);
		}
	}
	return *this;
}

Matrix& Matrix::operator -(float D) {
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] -= D;
		}
	}
	return *this;
}

Matrix& Matrix::operator -=(float D) {
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] -= D;
		}
	}
	return *this;
}

Matrix Matrix::subExpectedOutput(float expected) {
	if (expected + 1 > rows)
		throw 797;

	Matrix output = Matrix(rows, cols);
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
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
	Matrix output = Matrix(cols, rows);
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			output.M[j][i] = M[i][j];
		}
	}
	return output;
}

Matrix Matrix::computeOutput(float (func)(float&)) {
	Matrix output = Matrix(rows, cols);
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
	Matrix output = Matrix(rows, cols);
	for (unsigned i = 0; i < rows; ++i) {
		//Right
		for (unsigned j = 0; j < cols; ++j) {
			numb = std::exp(M[i][j]);
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

Matrix& Matrix::operator -(Matrix& M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] -= M2.at(i, j);
		}
	}
	return *this;
}

Matrix& Matrix::operator -=(Matrix& M2) {
	checkDimensionsPlus(M2.rows, M2.cols);
	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			this->M[i][j] -= M2.at(i, j);
		}
	}
	return *this;
}


Matrix Matrix::oneHot(int label, int classes) {
	Matrix output = Matrix(classes, 1);
	output.M[label][0] = 1.0;
	return output;
}

Matrix Matrix::test(int expected) {
	Matrix output = Matrix(rows, 1);
	for (unsigned i = 0; i < rows; ++i) {
		//d = 1
		if ((int) i == expected) {
			output.M[i][0] = 1.0f / this->at(i, 0);
		}
		//d = 0
		else {
			output.M[i][0] = 1.0f / (1.0f - this->at(i, 0));
		}
	}
	return output;
}

Matrix Matrix::jacobian() {
	Matrix output = Matrix(rows, rows);
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

unsigned Matrix::argMax() {
	unsigned biggest = 0;
	for (unsigned i = 1; i < rows; ++i) {
		if (M[biggest][0] < M[i][0]) {
			biggest = i;
		}
	}
	return biggest;
}

Matrix Matrix::transposeDotTranspose(Matrix& M2) {
	//checkDimensions(M2.rows);
	Matrix output = Matrix(M2.cols, cols);

	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			for (unsigned k = 0; k < M2.cols; ++k) {
				output.M[k][j] += this->at(i, j) * M2.at(i, k);
			}
		}
	}

	return output;
}

Matrix Matrix::transposeDot(Matrix& M2) {
	//checkDimensions(M2.rows);
	Matrix output = Matrix(cols, M2.cols);

	//Down M1
	for (unsigned i = 0; i < rows; ++i) {
		//Right M1
		for (unsigned j = 0; j < cols; ++j) {
			for (unsigned k = 0; k < M2.cols; ++k) {
				output.M[j][k] += this->at(i, j) * M2.at(i, k);
			}
		}
	}

	return output;
}