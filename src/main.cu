#include <iostream>
#include "essential/model/two_layer_net.h"
#include "essential/model/two_layer_net_copy.h"
#include "essential/model/simple_net.h"
#include "essential/utils/minst_read.h"
#include <time.h>

int main()
{

  //file.read((char*)&fweight,sizeof(fweight));



  Tensor img[10000];
  Tensor label[10000];
  ReadMNIST(img, "t10k-images.idx3-ubyte");
  ReadMNISTLabel(label, "t10k-labels.idx1-ubyte");
  TwoLayerNet model(784, 50, 10);
  
  Tensor* x = img;
  Tensor* t = label;
  Tensor grad;
  int batch_size = 1;
  float lr = 0.1;
  int iters_num = 10000;
  float loss_list[10000] = {0};
  
  int epochs =0;
  float loss_val=0;
  int acc_cnt =0;
  for (int epoch = 0; epoch < epochs ; epoch++)
  {
    for (int i =0; i< iters_num ; i++)
    {
      x[i] = x[i]/255;
      loss_val += model.gradient(x[i],t[i]);
      //cout<< "acc = "<<model.acc<<endl;
      //cout<< "? = "<<model.affine_2.dW_<<endl;
      //cout << model.affine_2.dW_;
      //model.numerical_gradient(x[i],t[i],model.affine_1.W_,model.affine_1.dW_);
      //model.numerical_gradient(x[i],t[i],model.affine_1.b_,model.affine_1.db_);
      //model.numerical_gradient(x[i],t[i],model.affine_2.W_,model.affine_2.dW_);
      //loss_val = model.numerical_gradient(x[i],t[i],model.affine_2.b_,model.affine_2.db_);
      model.upadte(lr);

      //loss_list[i] = model.loss(x[i],t[i]);
      if(i % 1000 ==0)
      {
        printf("loss = %f \n",loss_val/(float)1000);
        loss_val =0;
      }
    }
    loss_val =0;
    //cout <<"epoch = "<< epoch << " Accuracy:" << (float)acc_cnt / (float)iters_num  <<endl;
    //acc_cnt = 0;
    cout << "epoch = "<< epoch << " loss = "<< loss_val<<endl;
  }

  TwoLayerNetCopy model_copy;
  VALUETYPE loss_v=0;
  //model_copy.loadWeight();
  Tensor y;
  int accuracy_cnt =0;

  time_t start, end; 
  double result;
  int ep = 2;
  lr = 0.001;
  iters_num = 10000;
  for (int epoch =0; epoch<ep; epoch++)
  {
    start = time(NULL); // 수행 시간 측정 시작
    for (int i =0; i< iters_num ; i++)
    {
      if(epoch == 0)
        x[i] = x[i] / 255.0;
      y = model_copy.predict(x[i]);
      //loss_v += model_copy.loss(y,t[i]);
      loss_v += model_copy.gradient(y,t[i]);
      model_copy.upadte(lr);
      int p = y.argmax();
      int t_s = t[i].toScalar();
      //printf("p = %d, t = %d, loss = %f \n", p, t_s, loss_v);
      if ((int)p == (int)t_s)
      {
          accuracy_cnt++;
          //printf("true\n");
      }
      else
      {
        //printf("false\n");
      }
      if(i % 1000 ==0)
        {
          printf("loss = %f \n",loss_v/(float)1000);
          loss_v =0;
        }
    }
    end = time(NULL); // 시간 측정 끝
    result = (double)(end - start);
    cout << "time = " << result<<endl;;
    loss_v =0;
    cout <<"epoch = "<< epoch << " Accuracy:" << (float)accuracy_cnt / (float)iters_num  <<endl;
    accuracy_cnt = 0;
  }

  /*
  SimpleNet network;
  network.init_network();
  Tensor xx(2);
  xx(0) = 1.0; xx(1)= 0.5;
  Tensor yy;
  yy = network.forward(xx);
  cout<<yy;
*/
  return 0;
}