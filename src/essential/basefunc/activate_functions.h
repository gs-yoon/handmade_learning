#ifndef __ACTIVATE_FUNCTIONS_H__
#define __ACTIVATE_FUNCTIONS_H__
#include "base_operation.h"
#include "tensor.h"


using namespace std;

float sigmoid(float x);
float step(float x);
float relu(float x);
Tensor<float> softmax(Tensor<float>* x);
#endif