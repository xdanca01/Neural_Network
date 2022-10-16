#pragma once

#include <vector>

class Matrix
{
private:
	std::vector<std::vector<float>>* M;
public:
	unsigned rows;
	unsigned cols;
	Matrix(std::vector<std::vector<float>> &input);
	Matrix(std::vector<float>& input);
	Matrix();
	Matrix(unsigned R, unsigned C);
	Matrix* dot(Matrix &M2);
	Matrix* operator *(Matrix &M2);
	Matrix* operator *(std::vector<float> vec);
	Matrix* operator +(Matrix &M2);
	Matrix* transpose();
	void addRow(std::vector<float>& vec);
	//Exception if dimensions are not same
	void checkDimensions(unsigned bRows);
	void checkDimensionsPlus(unsigned bRows, unsigned bCols);
	float at(unsigned row, unsigned col);
	Matrix* computeOutput(float (*func)(float&));
	Matrix* softMax();
};

