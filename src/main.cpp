#include <iostream>

//#include "activate_functions.h"
//#include <Eigen/Dense>
//using Eigen::MatrixXd;
#include "essential/tensor/tensor.h"
#include "essential/basis/functions.h"

Tensor f(Tensor tt){
  std::cout <<"ff"<<&tt<<std::endl;
  return tt;
  //std::cout << &t <<std::endl;
}

int main()
{
  Tensor m(1,32);
  for (int i =0 ; i< 32; i++)
    m(0,i) = i;

  std::cout << m << std::endl;
  Tensor k(2,2,2,2,2);
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

  std::cout << "transpose" << std::endl;
  Tensor ret = m.reshape(2,2,2,2,2);
  std::cout << ret << std::endl;
  ret.printShape();

  return 0;
}