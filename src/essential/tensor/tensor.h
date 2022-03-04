#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "tensor_core.h"

Tensor log(const Tensor& x)
{
    return x.baseOp(log);
}

Tensor log10(const Tensor& x)
{
    return x.baseOp(log10);
}

Tensor exp(const Tensor& x)
{
    return x.baseOp(exp);
}

Tensor pow(const Tensor& x, double p)
{
    return x.baseOp(p,pow);
}
Tensor sum(const Tensor& x)
{
    return x.sum();
}
Tensor sum(const Tensor& x, int dim)
{
    return x.sum(dim);    
}
Tensor max(const Tensor& x)
{
    return x.max();
}
Tensor max(const Tensor& x, int dim)
{
    return x.max(dim);
}
Tensor min(const Tensor& x)
{
    return x.min();
}
Tensor min(const Tensor& x, int dim)
{
    return x.min(dim);
}
/*
Tensor maximum(const Tensor& x)
{
    return x.maximum(x);
}
Tensor maximum(const Tensor& x, int dim)
{
    return x.maximum(dim);
}
Tensor minimum(const Tensor& x)
{
    return x.minimum();
}
Tensor minimum(const Tensor& x, int dim)
{
    return x.minimum(dim);
}
*/

/*************
/* operator */
/*************/
Tensor operator+(int i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i + t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}
Tensor operator-(int i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i + t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}
Tensor operator*(int i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i * t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}
Tensor operator/(int i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        if(t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) != 0)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i / t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}

Tensor operator+(double i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i + t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}
Tensor operator-(double i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i + t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}
Tensor operator*(double i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i * t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}
Tensor operator/(double i, const Tensor& t)
{
    Tensor result;
    int shape[5] = {0};
    
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    result.makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]);

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        if(t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) != 0)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i / t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}



//template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    int t_rank = t.rank();
    int shape[5] = {0};
    for (int i =0 ; i< 5; i ++)
        shape[i] = t.getRawShape(i);

    for(int d1_idx =0; d1_idx < shape[0]; d1_idx++)
    {
        os << "{";
        for(int d2_idx =0; d2_idx < shape[1]; d2_idx++)
        {
            os << "[";
            for(int d3_idx =0; d3_idx < shape[2]; d3_idx++)
            {
                os << "{";
                for(int d4_idx = 0 ; d4_idx < shape[3] ; d4_idx++ )
                {
                    if (d4_idx >0)
                        os << "   ";
                    os << "[ ";
                    for(int d5_idx = 0 ; d5_idx < shape[4] ; d5_idx++ )
                    {
                        if (d5_idx >0)
                            os << " ";
                        os << t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) << ", ";
                    }
                    os << "]";
                    if( (t_rank > 1) && (d4_idx != shape[3] -1) )
                        os << "\n";
                }
                os << "}";
                if( (t_rank > 2) && (d3_idx != shape[2] -1) )
                    os << "\n";
            }
            os << "]";
            if( (t_rank > 3)&& (d2_idx != shape[1] -1) )
                os << "\n";
        }
        os << "}";
        if( (t_rank > 4)&& (d1_idx != shape[0] -1) )
            os << "\n";
    }
    return os;
}

#endif