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

Tensor abs(const Tensor& x)
{
    return x.baseOp(abs);
}

Tensor pow(const Tensor& x, double p)
{
    return x.baseOp(p,pow);
}
VALUETYPE sum(const Tensor& x)
{
    return x.sum();
}
Tensor sum(const Tensor& x, int dim)
{
    return x.sum(dim);    
}
VALUETYPE max(const Tensor& x)
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

void updateWithGradient(Tensor& w, Tensor& dw, float lr )
{
    int* shape = w.getRawShape();
    int size = w.getSize();
#if SIMDENABLE
    VALUETYPE* w_root = w.getData();
    VALUETYPE* dw_root = dw.getData();
    __m128 lr_duplicate;
    lr_duplicate = _mm_set1_ps(lr); // very expensive! 
    int i =0;
    for (int l =0 ; l < size / 4 ; l++)
    {
        __m128 w_128;
        __m128 dw_128;
        __m128 mulres; 
        __m128 res; 
        ALIGN float res_float[4] = {0};
        w_128 = _mm_load_ps(w_root + i);
        dw_128 = _mm_load_ps(dw_root + i);
        mulres = _mm_mul_ps(dw_128, lr_duplicate); // substantial calculations.
        res = _mm_sub_ps(w_128,mulres);
        _mm_store_ps(w_root + i, res); // expensive ! relocate the result data from __m128 into float*. 
        i +=4;
        if( i >= size )
            break;
    }
    for ( int i = size / 4 * 4 ; i < size ; i ++)
    {
        w(i) -= lr*dw(i);
    }

#else
    for (int i =0 ; i < size ; i++)
        w(i) -= lr*dw(i);
#endif
}

Tensor padding(Tensor& x, int pad_size )
{
    Tensor result;
    int* x_shape = x.getRawShape();
    
    result.createTensor(x_shape[0],x_shape[1],x_shape[2],x_shape[3] + pad_size*2 ,x_shape[4] + pad_size * 2);
    for(int d1_idx = 0 ; d1_idx < x_shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < x_shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < x_shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < x_shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < x_shape[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx+pad_size,d5_idx+pad_size) = x(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
    return result;
}

/*
void allocInGpu()
{
    //if (gpu_valid_ == 1)
    //    breakTensorInGpu();

    gpu_valid_ = 1;
    cudaMalloc(&shape_gpu_, DEFAULTMAXDIM*sizeof(int));

    int d1 = shape_[0];
    int d2 = shape_[1];
    int d3 = shape_[2];
    int d4 = shape_[3];
    int d5 = shape_[4];

    cudaMalloc(&root_gpu_, d1*sizeof(VALUETYPE));

    for(int idx = 0 ; idx < d1 ; idx ++)
    {
        cudaMalloc(&root_gpu_[idx], d2*sizeof(VALUETYPE));
    }
    
    for(int ri = 0 ; ri < d1 ; ri++)
    {
        for( int ci = 0; ci < d2 ; ci++)
        {
            cudaMalloc(&root_gpu_[ri][ci], d3*sizeof(VALUETYPE));
        }
    }

    for(int ri = 0 ; ri < d1 ; ri++)
    {
        for( int ci = 0; ci < d2 ; ci++)
        {
            for( int di = 0; di < d3 ; di++)
            {
                cudaMalloc(&root_gpu_[ri][ci][di], d4*sizeof(VALUETYPE));
            }
        }
    }

    for(int ri = 0 ; ri < d1 ; ri++)
    {
        for( int ci = 0; ci < d2 ; ci++)
        {
            for( int di = 0; di < d3 ; di++)
            {
                for(int d4i =0 ;d4i < d4 ; d4i++)
                    cudaMalloc(&root_gpu_[ri][ci][di][d4i], d5*sizeof(VALUETYPE));
            }
        }
    }
}


//template<typename T>
void breakTensorInGpu()
{
    gpu_valid_ = 0;

    for(int d1_idx = 0 ; d1_idx < shape_[0] ; d1_idx++)
    {
        for( int d2_idx = 0; d2_idx < shape_[1] ; d2_idx++)
        {
            for( int d3_idx = 0; d3_idx < shape_[2] ; d3_idx++)
            {
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                {
                    if (root_gpu_[d1_idx][d2_idx][d3_idx][d4_idx] != nullptr)
                    {
                        cudaFree(root_gpu_[d1_idx][d2_idx][d3_idx][d4_idx]);
                        root_gpu_[d1_idx][d2_idx][d3_idx][d4_idx] = nullptr;
                    }

                }
                if (root_gpu_[d1_idx][d2_idx][d3_idx] != nullptr)
                {
                    cudaFree(root_gpu_[d1_idx][d2_idx][d3_idx]);
                    root_gpu_[d1_idx][d2_idx][d3_idx] = nullptr;
                }
            }
            if (root_gpu_[d1_idx][d2_idx] != nullptr)
            {
                cudaFree(root_gpu_[d1_idx][d2_idx]);
                root_gpu_[d1_idx][d2_idx] = nullptr;
            }
        }
        if (root_gpu_[d1_idx] != nullptr)
        {
            cudaFree(root_gpu_[d1_idx]);
            root_gpu_[d1_idx] = nullptr;
        }
    }
    if (root_gpu_ != nullptr)
        {
            cudaFree(root_gpu_);
            root_gpu_ = nullptr;
        }

    if (shape_ != nullptr)
    
    {
        delete[] shape_gpu_;
        shape_gpu_ = nullptr;
    }
}

void copyHostToGpu()
{
    allocInGpu();
    cudaMalloc3D()
    cudaMemcpy(shape_gpu_, root_, DEFAULTMAXDIM*sizeof(int), cudaMemcpyHostToDevice);
    for(int d1_idx = 0 ; d1_idx < shape_[0] ; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ; d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ; d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    checkCuda(cudaMemcpy(root_gpu_[d1_idx][d2_idx][d3_idx][d4_idx], root_[d1_idx][d2_idx][d3_idx][d4_idx], shape_[4]*sizeof(VALUETYPE),cudaMemcpyHostToDevice));

}
void copyGpuToHost()
{
    for(int d1_idx = 0 ; d1_idx < shape_[0] ; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ; d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ; d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    checkCuda(cudaMemcpy(root_[d1_idx][d2_idx][d3_idx][d4_idx], root_gpu_[d1_idx][d2_idx][d3_idx][d4_idx], shape_[4]*sizeof(VALUETYPE),cudaMemcpyDeviceToHost));

    breakTensorInGpu();
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
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i - t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
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
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = i - t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
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
    os << "\n";
    return os;
}

#endif