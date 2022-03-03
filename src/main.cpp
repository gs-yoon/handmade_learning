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
  Tensor<float> m(1,2);
  m(0,0) = 3;
  m(0,1) = 1;

  std::cout << m << std::endl;
  Tensor<float> k(2,2,2,2,2);
  k(0,0,0,0,0) = 1;
  k(0,0,0,1,0) = 2;
  k(0,0,0,0,1) = 3;
  k(0,0,0,1,1) = 4;
  k(0,0,1,0,0) = 5;
  k(0,0,1,1,0) = 6;
  k(0,0,1,0,1) = 7;
  k(0,0,1,1,1) = 8;
  k(0,1,0,0,0) = 1;
  k(0,1,0,1,0) = 2;
  k(0,1,0,0,1) = 3;
  k(0,1,0,1,1) = 4;
  k(0,1,1,0,0) = 5;
  k(0,1,1,1,0) = 6;
  k(0,1,1,0,1) = 7;
  k(0,1,1,1,1) = 8;

  k(1,0,0,0,0) = 1;
  k(1,0,0,1,0) = 2;
  k(1,0,0,0,1) = 3;
  k(1,0,0,1,1) = 4;
  k(1,0,1,0,0) = 5;
  k(1,0,1,1,0) = 6;
  k(1,0,1,0,1) = 7;
  k(1,0,1,1,1) = 8;
  k(1,1,0,0,0) = 1;
  k(1,1,0,1,0) = 2;
  k(1,1,0,0,1) = 3;
  k(1,1,0,1,1) = 4;
  k(1,1,1,0,0) = 5;
  k(1,1,1,1,0) = 6;
  k(1,1,1,0,1) = 7;
  k(1,1,1,1,1) = 8;

  std::cout << k << std::endl;

  Tensor<float> ret;
  ret = k + m;
  ret = k - m;
  ret = k / m;
  ret = k * m;

  std::cout << ret << std::endl;

  return 0;
}