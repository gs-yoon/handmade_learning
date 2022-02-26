#include <iostream>

//#include "activate_functions.h"
//#include <Eigen/Dense>
//using Eigen::MatrixXd;
#include "tensor.cpp"

int main()
{
  /*
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m.array().exp() << std::endl;
  std::cout << m << std::endl;

  Eigen::Tensor<float64, 2> x(5,5);
  x.setRandom();
  cout << "xx" << endl;
  cout << x << endl;
  softmax( &x );
*/
  Tensor<float> tensor(2,2);

  return 0;
}