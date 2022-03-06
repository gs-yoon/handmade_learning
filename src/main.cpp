#include <iostream>

//#include "activate_functions.h"
//#include <Eigen/Dense>
//using Eigen::MatrixXd;
#include "essential/tensor/tensor.h"
#include "essential/basis/functions.h"
#include "essential/basis/layers.h"

Tensor f(Tensor tt){
  std::cout <<"ff"<<&tt<<std::endl;
  return tt;
  //std::cout << &t <<std::endl;
}

int main()
{
  int size = 8;
  Tensor m(4,4);
  for (int i =0 ; i< 16; i++)
    m(i/4,i%4) = i;

  std::cout << m << std::endl;
  Tensor k;
  k.makeTensor(2,2);
  k(0,0) = 1;
  k(1,0) = 2;
  k(0,1) = 3;
  k(1,1) = 4;

  std::cout << "k = "<< k << std::endl;

  Tensor ret;
  ret = softmax(m);
  std::cout << ret << std::endl;
  std::cout << m.getShape(0) << std::endl;
  std::cout << m.getShape(1) << std::endl;

  return 0;
}