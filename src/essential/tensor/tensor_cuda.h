#ifndef __TENSOR_CUDA_H__
#define __TENSOR_CUDA_H__

#include "tensor_core.h"
class TensorForGPU
{
private:
    int shape_0_1_2_3_4_ = 0; //total size
    int shape_1_2_3_4_ = 0; 
    int shape_2_3_4_ = 0; 
    int shape_3_4_ = 0; 
    int shape_4_ = 0; 

public:
    VALUETYPE* root_;
    int* shape_;
    int size_ = 0;
    int rank_ = 0;

    __device__ void init()
    {
        shape_4_ = shape_[4];
        shape_3_4_= shape_[3] * shape_4_;
        shape_2_3_4_ = shape_[2] * shape_3_4_;
        shape_1_2_3_4_ = shape_[1] * shape_2_3_4_;
        shape_0_1_2_3_4_ = shape_[0] * shape_1_2_3_4_;
        size_ = shape_0_1_2_3_4_;
    }

    __device__ VALUETYPE& root()const{ return *root_; }
    __device__ VALUETYPE& root(int i)const{ return *(root_ + i); }
    __device__ VALUETYPE& root(int i, int j)const{ return *(root_ + (shape_4_ *i) + j); }
    __device__ VALUETYPE& root(int i, int j, int k)const{ return *(root_ + (shape_3_4_ * i) + (shape_4_ *j) + k); }
    __device__ VALUETYPE& root(int i, int j, int k, int l)const{ return *(root_ + (shape_2_3_4_ * i) + (shape_3_4_ * j) + (shape_4_ *k) + l);}
    __device__ VALUETYPE& root(int i, int j, int k, int l, int m)const{ return *(root_ + (shape_1_2_3_4_*i) +  (shape_2_3_4_ * j) + (shape_3_4_ * k) + (shape_4_ *l) + m); }
};

Tensor dot(const Tensor&in_a , const Tensor&in_b);

void dotInGpu(const Tensor& in_a, const Tensor& in_b, const Tensor& in_result, TensorForGPU dev_a, TensorForGPU dev_b, TensorForGPU dev_result);

__global__ void dotRowKernel(TensorForGPU a, TensorForGPU b, TensorForGPU result);
__global__ void dotColKernel(TensorForGPU a, TensorForGPU b, TensorForGPU result);
__global__ void matMulKernel(TensorForGPU a, TensorForGPU b, TensorForGPU result);

Tensor dot(const Tensor&in_a , const Tensor&in_b)
{
    Tensor result;
    int result_shape[5];
    std::copy(in_a.getRawShape(),in_a.getRawShape()+5, result_shape);

    if ( (in_a.rank() == 1) && (in_b.rank()==1))
    {
        if ( in_a.getRawShape(COLIDX) == in_b.getRawShape(COLIDX) )
        {
            result_shape[3]=0;
            result_shape[4]=0;
        }
    }
    else if ( (in_a.rank() == 2) && (in_b.rank()==2))
    {
        if ( in_a.getRawShape(ROWIDX) == in_b.getRawShape(ROWIDX) )
        {
            result_shape[3]=0;
            result_shape[4]=0;
        }
    }
    else
    {
        result_shape[3]=in_a.getRawShape(ROWIDX);
        result_shape[4]=in_b.getRawShape(COLIDX);
    }

    #if CUDAENABLE
    result.createTensor(result_shape);

    TensorForGPU a_gpu;
    TensorForGPU b_gpu;
    TensorForGPU rst_gpu;
    int *a_shape;
    int *b_shape;

    cudaMalloc(&a_gpu.shape_, DEFAULTMAXDIM*sizeof(int));
    cudaMalloc(&b_gpu.shape_, DEFAULTMAXDIM*sizeof(int));
    cudaMalloc(&rst_gpu.shape_, DEFAULTMAXDIM*sizeof(int));
    cudaMemcpy(a_gpu.shape_, in_a.getRawShape(), DEFAULTMAXDIM*sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu.shape_, in_b.getRawShape(), DEFAULTMAXDIM*sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(rst_gpu.shape_, result.getRawShape(), DEFAULTMAXDIM*sizeof(int) , cudaMemcpyHostToDevice);
    
    cudaMalloc(&a_gpu.root_, in_a.getSize()*sizeof(VALUETYPE));
    cudaMalloc(&b_gpu.root_, in_b.getSize()*sizeof(VALUETYPE));
    cudaMemcpy(a_gpu.root_, in_a.rootAddress(), in_a.getSize()*sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu.root_, in_a.rootAddress(), in_b.getSize()*sizeof(int) , cudaMemcpyHostToDevice);
    a_gpu.rank_ = in_a.rank();
    b_gpu.rank_ = in_b.rank();

    cudaMalloc(&rst_gpu.root_, result.getSize()*sizeof(VALUETYPE));
    //dotInGpu(in_a, in_b, result, a_gpu, b_gpu, rst_gpu);
    cudaMemcpy(result.rootAddress(), rst_gpu.root_, result.getSize()*sizeof(int) , cudaMemcpyDeviceToHost);

    cudaFree(a_gpu.shape_);
    cudaFree(a_gpu.root_);
    cudaFree(b_gpu.shape_);
    cudaFree(b_gpu.root_);
    cudaFree(rst_gpu.root_);
    cudaFree(rst_gpu.shape_);

    #else
    result = in_a.dotMul(in_b);
    #endif
    result = in_a.dotMul(in_b);

    return result;
}


void dotInGpu(const Tensor& in_a, const Tensor& in_b, const Tensor& in_result, TensorForGPU dev_a, TensorForGPU dev_b, TensorForGPU dev_result)
{
    VALUETYPE sum = 0;
    size_t threads_per_block = 256;
    size_t number_of_blocks = 32;

    if ( (in_a.rank() == 1) && (in_b.rank()==1))
    {
        if ( in_a.getRawShape(COLIDX) == in_b.getRawShape(COLIDX) )
        {
//            number_of_blocks =
//            threads_per_block = 
            dotRowKernel<<<number_of_blocks, threads_per_block>>>(dev_a,dev_b,dev_result);
            cudaDeviceSynchronize();
            VALUETYPE* data;

            //for(int i =0 ; i< in_a.shape_[COLIDX]; i ++)
            //{
            //    sum += result.root(i);
            //}
            //result.root(0) = sum;
            printf("dot product error. dimension is not mathced\n ");
        }
        else
        {
            printf("dot product error. dimension is not mathced\n ");
        }
    }
    else if ( (in_a.rank() == 2) && (in_b.rank()==2))
    {
        if ( in_a.getRawShape(ROWIDX) == in_b.getRawShape(ROWIDX) )
        {
            dotRowKernel<<<number_of_blocks, threads_per_block>>>(dev_a,dev_b,dev_result);
            cudaDeviceSynchronize();
            //for(int i = 0 ; i< in_a.shape_[ROWIDX]; i ++)
            //{
            //    sum += result.root(i);
            //}
            //result.root(0) = sum;
            printf("dot product error. dimension is not mathced\n ");
        }
        else
        {
            printf("dot product error. dimension is not mathced\n ");
        }
    }
    else
    {
        if ( (in_a.getRawShape(0) == in_b.getRawShape(0) )
                && (in_a.getRawShape(0) == in_b.getRawShape(0))
                && (in_a.getRawShape(0) == in_b.getRawShape(0))
                && (in_a.getRawShape(COLIDX) == in_b.getRawShape(ROWIDX)) )
        {
            matMulKernel<<<number_of_blocks, threads_per_block>>>(dev_a,dev_b,dev_result);
            cudaDeviceSynchronize();
        }
        else
        {
            printf("MatMul Error. Matrix shapes are not matched\n");
            printf("col == %d , row == %d\n ",in_a.getRawShape(COLIDX), in_b.getRawShape(ROWIDX));
        }
    }
}

__global__
void dotRowKernel(TensorForGPU a, TensorForGPU b, TensorForGPU result)
{
    a.init();
    b.init();
    result.init();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i =idx ; i< a.shape_[COLIDX]; i += stride)
        result.root(i)= a.root(i) * b.root(i);

}
__global__
void dotColKernel(TensorForGPU a, TensorForGPU b, TensorForGPU result)
{
    a.init();
    b.init();
    result.init();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i =idx ; i< a.shape_[COLIDX]; i += stride)
        result.root(i)= a.root(i) * b.root(i);
}

__global__
void matMulKernel(TensorForGPU a, TensorForGPU b, TensorForGPU result)
{
    a.init();
    b.init();
    result.init();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int asr_bs =  a.shape_[ROWIDX] * b.shape_[COLIDX];  // in host fucntion
    int as2_asr_bs = a.shape_[2] * asr_bs;              // in host fucntion
    int as1_as2_asr_bs = a.shape_[1] * as2_asr_bs;      // in host fucntion
    int tatal_size = a.shape_[0] * as1_as2_asr_bs;      // in host fucntion

    for (int Ldx = idx ; Ldx < tatal_size; Ldx += stride)
    {

        int d1_idx = (Ldx / as1_as2_asr_bs) % a.shape_[0];
        int d2_idx = (Ldx / as2_asr_bs) % a.shape_[1];
        int d3_idx = (Ldx / asr_bs) % a.shape_[2];
        int i = (Ldx / b.shape_[COLIDX]) % a.shape_[ROWIDX] ;
        int k = Ldx % b.shape_[COLIDX];
        
        //int j = idx % a.shape_[4];
        int sum =0;
        for(int j = 0 ; j < a.shape_[4] ; j ++)
            sum += result.root(d1_idx, d2_idx, d3_idx, i , k) += a.root(d1_idx,d2_idx,d3_idx,i,j) * b.root(d1_idx,d2_idx,d3_idx,j,k);
        result.root(d1_idx, d2_idx, d3_idx, i , k) = sum;
    }
}


#endif