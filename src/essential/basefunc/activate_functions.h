#ifndef __ACTIVATE_FUNCTIONS_H__
#define __ACTIVATE_FUNCTIONS_H__
#include "base_operation.h"

using Eigen::MatrixXd;
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

float64 softmax(MatrixXd* x)
{
    //MatrixXd C(x->rows, x->cols);
    //C.setOnes();
    //C = C * x->maxCoeff();
    //return exp(x - C) / mexp(x - C).sum ;
    cout << mexp(x) << endl;;
}
#endif