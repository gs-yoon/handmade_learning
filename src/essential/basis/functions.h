
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
    Tensor result(x.getShape());
    result = x.baseOp(step_function_atomic);
    return result;
}

Tensor sigmoid(Tensor& x)
{
    return 1 / (1 + exp( (0-x) ));
}

Tensor sigmoid_grad(Tensor& x)
{
    return (1.0 - sigmoid(x)) * sigmoid(x);
}

Tensor relu(Tensor& x)
{
    return x.maximum(0); //??
}

inline double relu_grad_atomic(double x)
{
    return (x >= 0);
}

Tensor relu_grad(Tensor& x)
{
    Tensor result(x.getShape());
    result = x.baseOp(relu_grad_atomic);
    return result;
}
    
/*
Tensor softmax(Tensor& x)
{
    x = x - np.max(x, axis=-1, keepdims=True);
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True);
}


Tensor sum_squared_error(y, Tensor& t)
{
    return 0.5 * np.sum((y-t)**2);
}


Tensor cross_entropy_error(y, Tensor& t)
{
    if y.ndim ==  1
        t = t.reshape(1, t.size);
        y = y.reshape(1, y.size);
        
    // one-hot-vector
    if t.size == y.size
        t = t.argmax(axis=1);
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size;
}


Tensor softmax_loss(X, Tensor& t)
{
    y = softmax(X);
    return cross_entropy_error(y, t);
}
*/