
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
    //dotInGpu(a_gpu, b_gpu, rst_gpu);
    cudaMemcpy(result.rootAddress(), rst_gpu.root_, result.getSize()*sizeof(int) , cudaMemcpyDeviceToHost);

    cudaFree(a_gpu.shape_);
    cudaFree(a_gpu.root_);
    cudaFree(b_gpu.shape_);
    cudaFree(b_gpu.root_);
    cudaFree(rst_gpu.root_);
    cudaFree(rst_gpu.shape_);

    #else
    result = a.dotMul(b);
    #endif
    result = a.dotMul(b);

    return result;
}


void dotInGpu(TensorForGPU a, TensorForGPU b, TensorForGPU result)
{
    VALUETYPE sum = 0;


    size_t threads_per_block = 64;
    size_t number_of_blocks = 8;

    if ( (a.rank_ == 1) && (b.rank_==1))
    {
        if ( a.shape_[COLIDX] == b.shape_[COLIDX] )
        {
//            number_of_blocks =
//            threads_per_block = 
            dotRowKernel<<<number_of_blocks, threads_per_block>>>(a,b,result);
            VALUETYPE* data;

            //for(int i =0 ; i< a.shape_[COLIDX]; i ++)
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
    else if ( (a.rank_ == 2) && (b.rank_==2))
    {
        if ( a.shape_[ROWIDX] == b.shape_[ROWIDX] )
        {
            dotRowKernel<<<number_of_blocks, threads_per_block>>>(a,b,result);
            //for(int i = 0 ; i< a.shape_[ROWIDX]; i ++)
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
        if ( (a.shape_[0] == b.shape_[0]) 
                && (a.shape_[1] == b.shape_[1])
                && (a.shape_[2] == b.shape_[2])
                && (a.shape_[COLIDX] == b.shape_[ROWIDX]) )
        {
            matMulKernel<<<number_of_blocks, threads_per_block>>>(a,b,result);
        }
        else{
            printf("MatMul Error. Matrix shapes are not matched\n");
            printf("col == %d , row == %d\n ",a.shape_[COLIDX], b.shape_[ROWIDX]);
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