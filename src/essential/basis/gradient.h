#include "functions.h"

Tensor _numerical_gradient_1d(VALUETYPE (*f)(Tensor), Tensor x)
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

Tensor numerical_gradient_2d(VALUETYPE (*f)(Tensor), Tensor X)
{
    if (X.rank() == 1)
    {
        return _numerical_gradient_1d(f, X);
    }
    else
    {
        Tensor grad(X.getRawShape());

        for (int idx =0 ; idx ; idx++)
        {
            Tensor temp;
            temp = _numerical_gradient_1d(f, X.extract(idx,-1));
            for (int jdx = 0; jdx ; jdx++)
            {
                grad(idx,jdx) = temp(jdx);
            }
        }
        return grad;
    }
}
Tensor numerical_gradient(VALUETYPE (*f)(Tensor), Tensor& x)
{
    float h = 1e-4; // 0.0001
    Tensor grad(x.getRawShape());
    int* shape = x.getRawShape();

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                    {
                        VALUETYPE tmp_val = x(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                        x(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = tmp_val + h;
                        VALUETYPE fxh1 = f(x); // f(x+h)
                        
                        x(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = tmp_val - h ;
                        VALUETYPE fxh2 = f(x); // f(x-h)
                        grad(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = (fxh1 - fxh2) / (2*h);
                        
                        x(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = tmp_val;
                    }
    return grad;
}