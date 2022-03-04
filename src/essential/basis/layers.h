#include "functions.h"

Tensor _numerical_gradient_1d(VALUETYPE (*f)(Tensor), Tensor& x)
{
    float h = 1e-4; // 0.0001
    
    Tensor grad(x.getRawShape());
    int x_size = x.getSize();

    for (int idx = 0 ; idx < x_size ; idx++)
    {
        VALUETYPE tmp_val = x(idx);
        x(idx) = tmp_val + h;
        VALUETYPE fxh1 = f(x); // f(x+h)
        
        x(idx) = tmp_val - h ;
        VALUETYPE fxh2 = f(x); // f(x-h)

        grad(idx) = (fxh1 - fxh2) / (2*h);
        
        x(idx) = tmp_val;
    }    
    return grad;
}

/*
Tensor numerical_gradient_2d(VALUETYPE (*f)(Tensor), Tensor& X)
{
    if (X.rank() == 1)
        return _numerical_gradient_1d(f, X)
    else:
        Tensor grad(x.getRawShape());

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad
}

Tensor numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad
*/