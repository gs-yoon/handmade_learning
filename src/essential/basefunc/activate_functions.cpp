#ifndef __ACTIVATE_FUNCTIONS_H__
#define __ACTIVATE_FUNCTIONS_H__
#include "base_operation.h"

using Eigen::MatrixXd;
using Eigen::Tensor;
using namespace std;

inline float64 sigmoid(float64 x)
{
    return 1 / (1 + exp(-x));
}
inline float64 step(float64 x)
{
    return  ( x > 0 ) ? 1 : 0;
}
inline float64 relu(float64 x)
{
    return (x > 0 ) ? x : 0;
}

double softmaxx(double x)
{
  return 3;
}

Tensor<float64,1> softmax(Tensor<float64,1>* x)
{
  double c = toScalarVal(x->maximum());
  Tensor<float64,1> exp_calc_x = (*x - c).exp();
  
  return exp_calc_x / toScalarVal(exp_calc_x.sum());
}

Tensor<float64,2> softmax(Tensor<float64,2>* x)
{
    cout<<"sft 2"<<endl;
    Eigen::array<int, 1> dims;
    Tensor<float64,1> c = (x->maximum(dims));
    cout << c<<endl;
    Tensor<float64,2> exp_calc_x = (*x - c).exp();
    
    return exp_calc_x / exp_calc_x.sum(dims);
}

#endif