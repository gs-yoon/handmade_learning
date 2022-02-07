#ifndef __BASE_OPERATION_H__
#define __BASE_OPERATION_H__

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>
#include "type_definition.h"

using Eigen::MatrixXd;
using Eigen::Tensor;
typedef Eigen::Tensor<float64,0> Scalar;

Eigen::MatrixXd mexp(Eigen::MatrixXd* x);

Eigen::MatrixXd mlog(Eigen::MatrixXd* x);

Eigen::MatrixXd mpow(Eigen::MatrixXd* x, float32 p);

Eigen::Tensor<int,3> mexp(Eigen::Tensor<int,3>* x);

inline double toScalarVal(Scalar x)
{
    return x.data()[0];
}

#endif