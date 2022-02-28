#include <iostream>

//#include "activate_functions.h"
//#include <Eigen/Dense>
//using Eigen::MatrixXd;
#include "essential/tensor/tensor.h"

Tensor<float> f(Tensor<float> tt){
  std::cout <<"ff"<<&tt<<std::endl;
  return tt;
  //std::cout << &t <<std::endl;
}

int main()
{
  Tensor<float> m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = 4;

  std::cout << m << std::endl;

  Tensor<float> rst;

  rst = m.matMul(m);
  std::cout << rst << std::endl;

  return 0;
}