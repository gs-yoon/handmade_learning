#include <iostream>
#include "essential/model/two_layer_net.h"
#include "essential/utils/minst_read.h"


int main()
{
  Tensor img;
  ReadMNIST(10000,784,img);

  img.printShape();
  //cout << img;
  TwoLayerNet model(784, 50, 10);
  
  Tensor x;
  x = img.extract(5,-1,-1);
  cout << x;
  cout << model.predict(x);

  int iters_num = 10000;


  return 0;
}