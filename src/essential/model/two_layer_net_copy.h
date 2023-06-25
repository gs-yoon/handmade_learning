#ifndef __TWOLAYERNETCOPY_H__
#define __TWOLAYERNETCOPY_H__

#include "layers.h"
#include <fstream>
using namespace std;
class TwoLayerNetCopy
{
private:
    Tensor y_;
    Tensor t_;

public:

    Affine affine_1;
    Relu sigmoid1;
    Affine affine_2;
    Relu sigmoid2;
    Affine affine_3;
    SoftmaxWithLoss lastLayer;

    unsigned long acc =0;

    ~TwoLayerNetCopy(){}

    TwoLayerNetCopy()
    {
        t_.createTensor(10);
        affine_1.init(784, 52);
        affine_2.init(52, 100);
        affine_3.init(100, 10);
    }

    void loadWeight()
    {
        ifstream file ("weight.txt",ios_base::in);
        float fweight;
        //w1
        for(int row =0 ; row < 784 ; row++)
        {
            for(int col =0 ; col < 50 ; col++)
            {
                file >> fweight;
                affine_1.W_(row,col) = fweight;
            }
        }

        //w2
        for(int row =0 ; row < 50 ; row++)
        {
            for(int col =0 ; col < 100 ; col++)
            {
                file >> fweight;
                affine_2.W_(row,col) = fweight;
            }
        }

        //w3
        for(int row =0 ; row < 100 ; row++)
        {
            for(int col =0 ; col < 10 ; col++)
            {
                file >> fweight;
                affine_3.W_(row,col) = fweight;
            }
        }

        //b1
        for(int col =0 ; col < 50 ; col++)
        {
            file >> fweight;
            affine_1.b_(col) = fweight;
        }

        //b2
        for(int col =0 ; col < 100 ; col++)
        {
            file >> fweight;
            affine_2.b_(col) = fweight;
        }

        //b3
        for(int col =0 ; col < 10 ; col++)
        {
            file >> fweight;
            affine_3.b_(col) = fweight;
        }
    }

    Tensor predict(Tensor& x)
    {
        y_ = affine_1.forward(x);
        y_ = sigmoid1.forward(y_);
        y_ = affine_2.forward(y_);
        y_ = sigmoid2.forward(y_);
        y_ = affine_3.forward(y_);

        return y_;
    }
    Tensor predict_raw(Tensor& x_in)
    {
        Tensor x,a1,z1,a2,z2,a3,y;
        x = x_in.reshape(1,784);
        a1 = x.dotMul(affine_1.W_) + affine_1.b_;
        z1 = sigmoid(a1);
        a2 = z1.dotMul(affine_2.W_) + affine_2.b_;
        z2 = sigmoid(a2);
        a3 = z2.dotMul(affine_3.W_) + affine_3.b_;
        y = softmax(a3);
        return y;
    }
    VALUETYPE loss(Tensor& x, Tensor& t)
    {
        setLabels(t);
        lastLayer.setLabel(t_);

        VALUETYPE result =lastLayer.forward(x,t_).toScalar(); 
        //std::cout << y<<std::endl;
        int tdx = t.toScalar();
        t_(tdx) = 0;
        return result;
    }

    void setLabels(Tensor& t)
    {

        if (t.getSize() == 1)
        {
            int tdx = t.toScalar();
            t_(tdx) = 1;
        }
        else
        {
            t_ = t;
        }
    }
    Tensor accuracy(Tensor& x,Tensor& t)
    {
        /*
        y_ - np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        accuarcy = np.sum(y == t) / float(x.shape[0])

        return accuarcy*/
        return 0;
    }
    VALUETYPE loss_W(Tensor& x, Tensor& t)
    {
        Tensor y;
        y = predict(x);

        return loss(y, t);
    }

    VALUETYPE numerical_gradient(Tensor& x, Tensor& t, Tensor& var, Tensor& grad)
    {
        int* shape = grad.getRawShape();
        float h = 1e-4; // 0.0001
        VALUETYPE loss_tmp=0;
        for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        {
                            VALUETYPE tmp_val = var(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            var(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = tmp_val + h;
                            VALUETYPE fxh1 = loss(x, t); // f(x+h)
                            
                            var(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = tmp_val - h ;
                            VALUETYPE fxh2 = loss(x, t); // f(x-h)
                            loss_tmp= fxh2;
                            grad(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = (fxh1 - fxh2) / (2*h);
                            
                            var(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = tmp_val;
                        }
        return loss_tmp;
    }
    VALUETYPE gradient(Tensor& x, Tensor& t)
    {
        VALUETYPE loss_val = loss(x,t);

        Tensor dout;
        dout = 1;
        dout = lastLayer.backward(dout);
        dout = affine_3.backward(dout);
        dout = sigmoid2.backward(dout);
        dout = affine_2.backward(dout);
        dout = sigmoid1.backward(dout);
        dout = affine_1.backward(dout);

        return loss_val;
    }

    void upadte(float lr)
    {
        updateWithGradient(affine_1.W_, affine_1.dW_, lr);
        updateWithGradient(affine_1.b_, affine_1.db_, lr);
        updateWithGradient(affine_2.W_, affine_2.dW_, lr);
        updateWithGradient(affine_2.b_, affine_2.db_, lr);
        updateWithGradient(affine_3.W_, affine_3.dW_, lr);
        updateWithGradient(affine_3.b_, affine_3.db_, lr);
        //affine_1.W_ = affine_1.W_ - (lr*affine_1.dW_);
        //affine_1.b_ = affine_1.b_ - (lr*affine_1.db_);
        //affine_2.W_ = affine_2.W_ - (lr*affine_2.dW_);
        //affine_2.b_ = affine_2.b_ - (lr*affine_2.db_);
        //affine_3.W_ = affine_3.W_ - (lr*affine_3.dW_);
        //affine_3.b_ = affine_3.b_ - (lr*affine_3.db_);
        //std::cout<<affine_3.dW_;
    }
};

#endif