#ifndef __ACTIVATE_FUNCTIONS_H__
#define __ACTIVATE_FUNCTIONS_H__
#include "base_operation.h"

using Eigen::MatrixXd;
using namespace std;

float64 sigmoid(float64 x);
float64 step(float64 x);
float64 relu(float64 x);
Tensor<float64,1> softmax(Tensor<float64,1>* x);
//Tensor<float64,2> softmax(Tensor<float64,2>* x);
void softmax(Tensor<float64,2>* x);
double softmaxx(double x);
#endif