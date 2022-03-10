
#ifndef __FUNTIONS_H__
#define __FUNTIONS_H__

#include "tensor.h"

Tensor identity_function(Tensor& x)
{
    return x;
}

inline double step_function_atomic(double x)
{
    return (x > 0);
}

Tensor step_function(Tensor& x)
{
    Tensor result(x.getRawShape());
    result = x.baseOp(step_function_atomic);
    return result;
}

Tensor sigmoid(Tensor& x)
{
    return 1 / (1 + exp( (x*(-1)) ));
}

Tensor sigmoid_grad(Tensor& x)
{
    return (1.0 - sigmoid(x)) * sigmoid(x);
}

inline double relu_atomic(double x)
{
    return x > 0 ? x:0;
}

Tensor relu(Tensor& x)
{
    return x.baseOp(relu_atomic);
}

inline double relu_grad_atomic(double x)
{
    return x <= 0? true : false;
}

Tensor relu_grad(Tensor& x)
{
    Tensor result(x.getRawShape());
    result = x.baseOp(relu_grad_atomic);
    return result;
}
    
Tensor softmax(Tensor& x)
{
    Tensor x_c;
    x_c = x - x.max(0);
    //std::cout << x;
    return x_c.exp() / x_c.exp().sum();
}


VALUETYPE sum_squared_error(Tensor& y , Tensor& t) //??
{
    return 0.5 * ((y-t).pow(2)).sum();
}


Tensor cross_entropy_error(Tensor& y, Tensor& t)
{
    int t_size = t.getSize();
    int y_size = y.getSize();
    if (y.rank() ==  1)
        t = t.reshape(1, t_size);
        
    // one-hot-vector
    //if (t_size == y_size)
    //    t = t.argmax(axis=1);
             
    int batch_size = 1;// y.getShape(0);
    //return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size;
    Tensor result;
    result = ( sum(t * (y + (1e-9)).log())*(-1) ) / (float)batch_size;
    return result;
}

Tensor softmax_loss(Tensor& X, Tensor& t)
{
    Tensor y;
    y = softmax(X);
    return cross_entropy_error(y, t);
}

#endif