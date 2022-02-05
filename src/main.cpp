#include <iostream>
#include <Eigen/Dense>
#include <opencv4/core/mat.hpp>
#include "activate_functions.h"

using Eigen::MatrixXd;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m.array().exp() << std::endl;
  std::cout << m << std::endl;

  softmax(&m);
}