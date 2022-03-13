
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

Tensor convolution(const Tensor&x, const Tensor& w, int filters, int kernel_size, int stride)
{
    Tensor result;
    int* w_shape = w.getRawShape();
    int* x_shape = x.getRawShape();
    int row = x_shape[3];
    int col = x_shape[4];
    int kernel_half = kernel_size/2;

    result.createTensor(filters,row,col); // how to 3d images? (color images)

    for(int ri = kernel_half ; ri < row - kernel_half; ri += stride)
    {
        for(int ci = kernel_half ; ci < col - kernel_half; ci += stride)
        {
//                for(int d1_idx = 0 ; d1_idx < w_shape[0]; d1_idx++)
//                    for( int d2_idx = 0; d2_idx < w_shape[1] ;d2_idx++)
            int ri_k = ri - kernel_half;
            int ci_k = ci - kernel_half;
            for( int d3_idx = 0; d3_idx < w_shape[2] ;d3_idx++) //filter num
            {
                for( int d4_idx = 0; d4_idx < w_shape[3] ; d4_idx++)
                {
                    if (ri_k +d4_idx < row)
                    {
                        for( int d5_idx = 0; d5_idx < w_shape[4] ;d5_idx++)
                        {
                            if (ci_k + d5_idx < col)
                                result(d3_idx, ri_k, ci_k) = x(d3_idx, ri_k + d4_idx, ci_k + d5_idx ) * w(d3_idx,d4_idx,d5_idx);
                        }
                    }
                }
            }
        }
    }

    return result;
}


#endif