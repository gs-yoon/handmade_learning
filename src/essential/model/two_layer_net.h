#ifndef __TWOLAYERNET_H__
#define __TWOLAYERNET_H__

#include "layers.h"

class TwoLayerNet
{
private:
    Tensor y_;
    Tensor t_;

public:

    Affine affine_1;
    Relu Relu_1;
    Affine affine_2;
    SoftmaxWithLoss lastLayer;

    unsigned long acc =0;

    TwoLayerNet(){}
    ~TwoLayerNet(){}

    TwoLayerNet(int input_size, int hidden_size, int output_size)
    {
        t_.createTensor(10);
        affine_1.init(input_size, hidden_size);
        affine_2.init(hidden_size, output_size);
    }

    Tensor predict(Tensor& x)
    {
        y_ = affine_1.forward(x);
        y_ = Relu_1.forward(y_);
        y_ = affine_2.forward(y_);

        return y_;
    }

    VALUETYPE loss(Tensor& x, Tensor& t)
    {
        setLabels(t);
        lastLayer.setLabel(t_);
        Tensor y;
        y = predict(x);

        if (y.argmax() == t.toScalar())
        {
            acc++;
        }

        VALUETYPE result =lastLayer.forward(y,t_).toScalar(); 
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
        //std::cout << dout<<std::endl;
        dout = lastLayer.backward(dout);
       // std::cout << "lastlayer "<<std::endl;
       // std::cout << dout<<std::endl;
        dout = affine_2.backward(dout);
      //  std::cout << dout<<std::endl;
        dout = Relu_1.backward(dout);
       // std::cout << dout<<std::endl;
        dout = affine_1.backward(dout);
        //std::cout << dout<<std::endl;


        return loss_val;
    }

    void upadte(float lr)
    {
        affine_1.W_ = affine_1.W_ - (lr*affine_1.dW_);
        affine_1.b_ = affine_1.b_ - (lr*affine_1.db_);
        affine_2.W_ = affine_2.W_ - (lr*affine_2.dW_);
        affine_2.b_ = affine_2.b_ - (lr*affine_2.db_);
    }
};

#endif