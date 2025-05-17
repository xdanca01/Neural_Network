#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

class Matrix
{
private:
	std::vector<std::vector<float> > M;
public:
	unsigned rows;
	unsigned cols;
	Matrix(const std::vector<std::vector<float>>& input);
	Matrix(std::vector<std::vector<float>>&& input);
	Matrix(const std::vector<float>& input);
	Matrix(std::vector<float>&& input);
	Matrix(Matrix&& other) noexcept;
	Matrix(const Matrix& other) noexcept;
	Matrix();
	Matrix(unsigned R, unsigned C);
	Matrix& operator=(Matrix&& other) noexcept;
	Matrix& operator=(const Matrix& other) noexcept;
	Matrix dot(Matrix& M2);
	Matrix dot(std::vector<float>& vec);
	Matrix& operator *(Matrix& M2);
	Matrix& operator *(float D);
	Matrix& multiply(Matrix& M2);
	Matrix& multiply(float numb);
	Matrix& operator +(Matrix& M2);
	Matrix& operator +=(Matrix& M2);
	Matrix& operator -(float D);
	Matrix& operator -=(float D);
	Matrix& operator -=(Matrix& M2);
	Matrix& operator -(Matrix& M2);
	Matrix transpose();
	void addRow(std::vector<float>&& vec);
	Matrix transposeDotTranspose(Matrix& M2);
	Matrix transposeDot(Matrix& M2);

	//Exception if dimensions are not same
	void checkDimensions(unsigned bRows);
	void checkDimensionsPlus(unsigned bRows, unsigned bCols);
	float at(unsigned row, unsigned col);
	Matrix computeOutput(float (func)(float&));
	Matrix softMax();
	Matrix subExpectedOutput(float expected);
	Matrix oneHot(int label, int classes);
	Matrix test(int expected);
	Matrix jacobian();
	float sum();
	unsigned argMax();
};