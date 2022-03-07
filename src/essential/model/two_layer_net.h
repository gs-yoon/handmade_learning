#ifndef __TWOLAYERNET_H__
#define __TWOLAYERNET_H__

#include "layers.h"

class TwoLayerNet
{
private:
    Tensor y_;
    Tensor t_;

    Affine affine_1;
    Relu Relu_1;
    Affine affine_2;
    SoftmaxWithLoss lastLayer;

public:
    TwoLayerNet(){}
    ~TwoLayerNet(){}

    TwoLayerNet(int input_size, int hidden_size, int output_size)
    {
        affine_1.init(input_size, hidden_size);
        affine_2.init(hidden_size, output_size);
    }

    Tensor predict(Tensor& x)
    {
        x = affine_1.forward(x);
        x.printShape();
        x = Relu_1.forward(x);
        x.printShape();
        x = affine_2.forward(x);
        x.printShape();

        y_ = x;
        return y_;
    }

    VALUETYPE loss(Tensor& x, Tensor& t)
    {
        lastLayer.setLabel(t);
        return lastLayer.forward(y_).toScalar();
    }

    void setLabels(Tensor& t)
    {
        t_ = t;
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
    VALUETYPE loss_W(Tensor& x)
    {
        Tensor y;
        y = predict(x);

        return loss(y, t_);
    }

    Tensor numericalGrad(Tensor& x, Tensor& t)
    {

        Tensor grads;
        //grads = numerical_gradient(loss_W, Affine.w);
        return grads;
    }
    Tensor gradient(Tensor& x, Tensor& t)
    {
        Tensor tmp;
        tmp = predict(x);
        VALUETYPE loss_val = loss(x,t);

        Tensor dout;
        dout = 1;
        dout = lastLayer.backward(dout);

        dout = affine_2.backward(dout);
        dout = Relu_1.backward(dout);
        dout = affine_1.backward(dout);

        Tensor dW1;
        Tensor db1;
        Tensor dW2;
        Tensor db2;

        dW1 = affine_1.dW_;
        db1 = affine_1.db_;
        dW2 = affine_2.dW_;
        db2 = affine_2.db_;

        return dW1;
    }
};

#endif