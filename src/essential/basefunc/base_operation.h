#ifndef __BASE_OPERATION_H__
#define __BASE_OPERATION_H__

#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include "type_definition.h"

using Eigen::MatrixXd;

Eigen::MatrixXd mexp(Eigen::MatrixXd* x);

Eigen::MatrixXd mlog(Eigen::MatrixXd* x);

Eigen::MatrixXd mpow(Eigen::MatrixXd* x, float32 p);

#endif