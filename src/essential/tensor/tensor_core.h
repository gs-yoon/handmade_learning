#ifndef __TENSOR_CORE_H__
#define __TENSOR_CORE_H__

#include<stdio.h>
#include<math.h>
#include<algorithm>
#include<iostream>
#include <unistd.h> // unix defined library.
#include <immintrin.h> // Intel core SIMD intrinsic instructions. 
#include <pmmintrin.h> // SSE3
#include <malloc.h>
//#include"cuda_utils.h"

#define TENSORDEBUG 0
#define CUDAENABLE 0
#define SIMDENABLE 1
#define DEFAULTMAXDIM 5
#define ROWIDX 3
#define COLIDX 4

#define ALIGN alignas(16) // macro for aligned memory


typedef float SCALARTYPE;
typedef float VALUETYPE;

enum OPERATIONS{
    SUM=0,
    SUB,
    MUL,
    DIV};

enum SHAPEMATCH{
    SHAPE_UNMATCHED =0,
    SHAPE_EQUAL,
    SHAPE_4D_CONST,
    SHAPE_3D_CONST,
    SHAPE_2D_CONST,
    SHAPE_1D_CONST,
    SHAPE_0D_CONST, //scalar
    SHAPE_1D_MATCH, //col match
    SHAPE_2D_MATCH,
    SHAPE_3D_MATCH,
    SHAPE_4D_MATCH
};

long long g_delete_cnt =0;
long long g_make_cnt =0;

//template<typename T = float>
class Tensor
{
private:
    VALUETYPE* root_ = nullptr;

    int* shape_ = nullptr;
    int row_ =0;
    int col_ =0;
    int rank_ = 0;
    int valid_ = 0;
    int shape_0_1_2_3_4_ = 0; //total size
    int shape_1_2_3_4_ = 0; 
    int shape_2_3_4_ = 0; 
    int shape_3_4_ = 0; 
    int shape_4_ = 0; 
    int size_ = 0;

private:
    VALUETYPE& root()const{ return *root_; }
    VALUETYPE& root(int i)const{ return *(root_ + i); }
    VALUETYPE& root(int i, int j)const{ return *(root_ + (shape_4_ *i) + j); }
    VALUETYPE& root(int i, int j, int k)const{ return *(root_ + (shape_3_4_ * i) + (shape_4_ *j) + k); }
    VALUETYPE& root(int i, int j, int k, int l)const{ return *(root_ + (shape_2_3_4_ * i) + (shape_3_4_ * j) + (shape_4_ *k) + l);}
    VALUETYPE& root(int i, int j, int k, int l, int m)const{ return *(root_ + (shape_1_2_3_4_*i) +  (shape_2_3_4_ * j) + (shape_3_4_ * k) + (shape_4_ *l) + m); }

    VALUETYPE* rootAddress()const{ return root_; }
    VALUETYPE* rootAddress(int i, int j, int k, int l, int m)const{ return (root_ + (shape_1_2_3_4_*i) +  (shape_2_3_4_ * j) + (shape_3_4_ * k) + (shape_4_ *l) + m); }


    void breakTensorInGpu();
    void allocInGpu();

    Tensor matMul1D(const Tensor& )const;
    Tensor matMul2D(const Tensor& )const;
    Tensor elementWise(const Tensor&,const Tensor& ,int ,int);
    void breakTensor();
    int checkShape(const Tensor&) const;

public:
    Tensor() { valid_ = 0; }
    Tensor(int d1) { makeTensor(1,1,1,1,d1); }
    Tensor(int d1, int d2) { makeTensor(1,1,1,d1,d2); }
    Tensor(int d1, int d2, int d3) { makeTensor(1,1,d1,d2,d3); }
    Tensor(int d1, int d2, int d3, int d4) { makeTensor(1,d1,d2,d3,d4); }
    Tensor(int d1, int d2, int d3, int d4, int d5) { makeTensor(d1,d2,d3,d4,d5); }
    Tensor(int *shape){ makeTensor(shape[0],shape[1],shape[2],shape[3],shape[4]); }
    Tensor(const Tensor& cp);
    ~Tensor() { if (valid_) breakTensor(); }

    void makeTensor(int d1, int d2, int d3, int d4, int d5);

    void makeTensor(int d1) { makeTensor(1,1,1,1,d1); }
    void makeTensor(int d1, int d2) { makeTensor(1,1,1,d1,d2); }
    void makeTensor(int d1, int d2, int d3) { makeTensor(1,1,d1,d2,d3); }
    void makeTensor(int d1, int d2, int d3, int d4) { makeTensor(1,d1,d2,d3,d4); }
    
    void makeZeros(int d1, int d2, int d3, int d4, int d5);


    void createTensor(int d1) { makeZeros(1,1,1,1,d1); }
    void createTensor(int d1, int d2) { makeZeros(1,1,1,d1,d2); }
    void createTensor(int d1, int d2, int d3) { makeZeros(1,1,d1,d2,d3); }
    void createTensor(int d1, int d2, int d3, int d4) { makeZeros(1,d1,d2,d3,d4); }
    void createTensor(int d1, int d2, int d3, int d4, int d5) { makeZeros(d1,d2,d3,d4,d5); }
    void createTensor(int *shape){ makeZeros(shape[0],shape[1],shape[2],shape[3],shape[4]); }

    int getRawShape(int dim)const;
    int* getRawShape()const;
    int getShape(int idm)const;
    int setVal();
    void setConstant(VALUETYPE);
    int setRandom();
    int getSize()const;
    int rank()const;
    VALUETYPE* getData()const;
    Tensor matMul(const Tensor& )const;
    Tensor dotMul(const Tensor& )const;
    Tensor reshape(int);//const?
    Tensor reshape(int, int);//const?
    Tensor reshape(int, int, int);//const?
    Tensor reshape(int, int, int, int);//const?
    Tensor reshape(int, int, int, int, int);//const?
    Tensor reshape(int*);//const?
    Tensor transpose() const;
    Tensor flatten();//const?

    Tensor extract(int i, int j);
    Tensor extract(int i, int j, int k);
    Tensor extract(int i, int j, int k, int l);
    Tensor extract(int i, int j, int k, int l, int m);

    void copyHostToGpu();
    void copyGpuToHost();

    VALUETYPE toScalar()
    {
        if( rank_ ==0 )
            return root(0);
        else
            printf("Invalid Converting. No Scalar\n");
            return -1;
    }
    //Tensor partialCopy(int i, int j, int k, int l, Tesnor& x);

    Tensor baseOp(double (*fp)(double ) ) const;
    Tensor baseOp(float p, double (*fp)(double, double )) const;
    VALUETYPE sum() const;
    Tensor sum(int dim) const; // TODO: to modify for support 3D 
    int argmax() const;
    VALUETYPE max() const;
    Tensor max(int dim) const;
    VALUETYPE min() const;
    Tensor min(int dim) const;
    Tensor log();//const?
    Tensor log10();//const?
    Tensor exp();//const?
    Tensor pow(float p);//const?

    inline void printShape()const;

    VALUETYPE& operator()(int i)const { return root(i); }
    VALUETYPE& operator()(int i, int j)const { return root(i,j); }
    VALUETYPE& operator()(int i, int j, int k)const { return root(i,j,k); }
    VALUETYPE& operator()(int i, int j, int k, int l)const { return root(i,j,k,l); }
    VALUETYPE& operator()(int i, int j, int k, int l, int m)const { return root(i,j,k,l,m); }


    Tensor operator +(int val) 
    {
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) + val;
        return result;
    }
    Tensor operator +(double val) 
    {
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) + val;
        return result;
    }
    Tensor operator *(int val) 
    {
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) * val;
        return result;
    }
    Tensor operator *(double val) 
    {
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) * val;
        return result;
    }
    Tensor operator -(int val) 
    {
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) - val;
        return result;
    }
    Tensor operator -(double val) 
    {
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) - val;
        return result;
    }
    Tensor operator /(int val) 
    {
        if (val == 0)
        {
            printf("divided by 0\n");
            return -1;
        }
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) / val;
        return result;
    }
    Tensor operator /(double val) 
    {
        if (val == 0)
        {
            printf("divided by 0\n");
            return -1;
        }
        Tensor result;
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                            result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) / val;
        return result;
    }

    Tensor operator +(const Tensor& in) 
    {
        Tensor ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, SUM);
        else
        {
            shape_status = checkShape(in) ;
            if (shape_status >0)
                ret = elementWise(in,*this, shape_status, SUM);
            else
                printf("elementWise error\n");
        }
        return ret;
    }

    Tensor operator *(const  Tensor& in) 
    {
        Tensor ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, MUL);
        else
        {
            shape_status = checkShape(in) ;
            if (shape_status >0)
                ret = elementWise(in,*this, shape_status, MUL);
            else
                printf("elementWise error\n");
        }
        return ret;
    }
    Tensor operator -(const Tensor& in) 
    {
        Tensor ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, SUB);
        else
            printf("element wise error.\n");
        return ret;
    }
    Tensor operator /(const Tensor& in) 
    {
        Tensor ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, DIV);
        else
            printf("element wise error.\n");
        return ret;
    }

    Tensor& operator=(const Tensor& cp);
    Tensor& operator=(const VALUETYPE scalar);
    Tensor& operator=(const int scalar);
    Tensor& operator=(const bool scalar);

    bool operator!=(int t)
    {
        if (rank_ == 0)
        {
            return root(0) != t; 
        }
        else
        {
            printf("no scalar\n");
            return true;
        }
    }
    bool operator!=(bool t)
    {
        if (rank_ == 0)
        {
            return root(0) != t; 
        }
        else
        {
            printf("no scalar\n");
            return true;
        }
    }
    bool operator==(int t)
    {
        if (rank_ == 0)
        {
            return root(0) == t; 
        }
        else
        {
            printf("no scalar\n");
            return true;
        }
    }
    bool operator==(bool t)
    {
        if (rank_ == 0)
        {
            return root(0) == t; 
        }
        else
        {
            printf("no scalar\n");
            return true;
        }
    }
    bool operator<(const Tensor t)
    {
        if (rank_ == 0)
            return root(0) < t.root(0,0,0,0,0); 
        else
            return false;
    }
    bool operator>(const Tensor t)
    {
        if (rank_ == 0)
            return root(0) > t.root(0,0,0,0,0); 
        else
            return false;
    }
};

//copy constructor
//template<typename T>
Tensor::Tensor(const Tensor& cp)
{
    #if TENSORDEBUG
    printf("copy\n");
    #endif 

    if ((root_ == nullptr) && (shape_== nullptr))
    {
        makeTensor(cp.shape_[0],cp.shape_[1],cp.shape_[2],cp.shape_[3],cp.shape_[4]);
    }

    else if ((root_ == nullptr) || (shape_== nullptr))
    {
        #if TENSORDEBUG
        printf("Critical constructor error\n");
        printf("Rebuilding Tensor...\n");
        #endif 
        breakTensor();
        makeTensor(cp.shape_[0],cp.shape_[1],cp.shape_[2],cp.shape_[3],cp.shape_[4]);
    }
    else{
        #if TENSORDEBUG
        printf("Rebuilding Tensor...\n");
        #endif 
        breakTensor();
        makeTensor(cp.shape_[0],cp.shape_[1],cp.shape_[2],cp.shape_[3],cp.shape_[4]);
    }

    rank_ = cp.rank_;
    for( int i =0 ;i <= 5 ; i++)
    {
        shape_[i] = cp.shape_[i];
    }

    std::copy(cp.root_, cp.root_ + size_, root_) ;

}

//assign constructor
//template<typename T>
Tensor& Tensor::operator=(const VALUETYPE scalar)
{
    if (valid_ == true)
    {
        if(rank_ > 0)
        {
            breakTensor();
            printf("warning! a tensor was breaked for assign scalar\n");
        }
    }

    if (valid_ == false)
    {
        makeTensor(1,1,1,1,1);
    }

    root(0) = scalar;
    return *this;
}

//template<typename T>
Tensor& Tensor::operator=(const int scalar)
{
    if (valid_ == true)
    {
        if(rank_ > 0)
        {
            breakTensor();
            printf("warning! a tensor was breaked for assign scalar\n");
        }
    }

    if (valid_ == false)
    {
        makeTensor(1,1,1,1,1);
    }

    root(0) = scalar;
    return *this;
}
//template<typename T>
Tensor& Tensor::operator=(const bool scalar)
{
    if (valid_ == true)
    {
        if(rank_ > 0)
        {
            breakTensor();
            printf("warning! a tensor was breaked for assign scalar\n");
        }
    }

    if (valid_ == false)
    {
        makeTensor(1,1,1,1,1);
    }

    root(0) = scalar;
    return *this;
}
//template<typename T>
Tensor& Tensor::operator=(const Tensor& cp)
{
    bool equal = false;
    #if TENSORDEBUG
    printf("====\n");
    #endif
    if ((root_ == nullptr) && (shape_== nullptr))
    {
        makeTensor(cp.shape_[0],cp.shape_[1],cp.shape_[2],cp.shape_[3],cp.shape_[4]);
    }
    else if ((root_ == nullptr) || (shape_== nullptr))
    {
        #if TENSORDEBUG
        printf("Critical constructor error\n");
        printf("Rebuilding Tensor...\n");
        #endif
        breakTensor();
        makeTensor(cp.shape_[0],cp.shape_[1],cp.shape_[2],cp.shape_[3],cp.shape_[4]);
    }
    else{
        if (checkShape(cp) == SHAPE_EQUAL)
        {
            equal = true;
        }
        else
        {
            #if TENSORDEBUG
            printf("Rebuilding Tensor...\n");
            #endif
            breakTensor();
            makeTensor(cp.shape_[0],cp.shape_[1],cp.shape_[2],cp.shape_[3],cp.shape_[4]); // TODO : segmentation fault, why?
        }
    }

    if(equal == false)
    {
        rank_ = cp.rank_;
        for( int i =0 ;i < 5 ; i++)
        {
            shape_[i] = cp.shape_[i];
        }
    }
    std::copy(cp.root_, cp.root_ + size_, root_) ;

    return *this;
}

//template<typename T>
inline void Tensor::makeTensor(int d1, int d2, int d3, int d4, int d5)
{
    if (valid_ == 1)
        breakTensor();

    valid_ = 1;
    g_make_cnt ++;

    row_ = d4;
    col_ = d5;

    shape_ = new int[DEFAULTMAXDIM];
    shape_[0] = d1;
    shape_[1] = d2;
    shape_[2] = d3;
    shape_[3] = d4;
    shape_[4] = d5;

    shape_4_ = shape_[4];
    shape_3_4_= shape_[3] * shape_4_;
    shape_2_3_4_ = shape_[2] * shape_3_4_;
    shape_1_2_3_4_ = shape_[1] * shape_2_3_4_;
    shape_0_1_2_3_4_ = shape_[0] * shape_1_2_3_4_;
    size_ = shape_0_1_2_3_4_;
    
    //root_ = new ALIGN VALUETYPE[size_];
    root_ = (VALUETYPE*)aligned_alloc((size_t)16*sizeof(char), (size_t)size_*sizeof(VALUETYPE));

    if (d1 >1)
        rank_ = 5;
    else if (d2 >1)
        rank_ = 4;
    else if (d3 >1)
        rank_ = 3;
    else if (d4 >1)
        rank_ = 2;
    else if (d5 >1)
        rank_ = 1;
    else
        rank_ = 0;

}

//template<typename T>
inline void Tensor::makeZeros(int d1, int d2, int d3, int d4, int d5)
{
    if (valid_ == 1)
        breakTensor();

    valid_ = 1;
    g_make_cnt ++;

    row_ = d4;
    col_ = d5;

    shape_ = new int[DEFAULTMAXDIM];
    shape_[0] = d1;
    shape_[1] = d2;
    shape_[2] = d3;
    shape_[3] = d4;
    shape_[4] = d5;

    shape_4_ = shape_[4];
    shape_3_4_= shape_[3] * shape_4_;
    shape_2_3_4_ = shape_[2] * shape_3_4_;
    shape_1_2_3_4_ = shape_[1] * shape_2_3_4_;
    shape_0_1_2_3_4_ = shape_[0] * shape_1_2_3_4_;
    size_ = shape_0_1_2_3_4_;
    
    root_ = new VALUETYPE[size_];
    std::fill(root_, root_ + size_, 0x00);

    if (d1 >1)
        rank_ = 5;
    else if (d2 >1)
        rank_ = 4;
    else if (d3 >1)
        rank_ = 3;
    else if (d4 >1)
        rank_ = 2;
    else if (d5 >1)
        rank_ = 1;
    else
        rank_ = 0;

}

//template<typename T>
void Tensor::breakTensor()
{
    g_delete_cnt ++;
    valid_ = 0;

    if (root_ != nullptr)
        {
            free(root_);
            root_ = nullptr;
        }

    if (shape_ != nullptr)
    {
        delete[] shape_;
        shape_ = nullptr;
    }
}

//template<typename T>
int* Tensor::getRawShape() const
{
    return shape_;
}
//template<typename T>
int Tensor::getRawShape(int dim) const
{
    if (dim < 5)
    {
        return shape_[dim];
    }
    else
    {
        printf("input parameter of getRawShape is out of bound\n");
        return -1;
    }
}

//template<typename T>
int Tensor::getShape(int dim) const
{
    if (rank_ > dim)
    {
        return shape_[dim + (5 - rank_)];
    }
    else
    {
        printf("input parameter of getRawShape is out of bound\n");
        return -1;
    }
}

int Tensor::setVal()
{
    return 0;
}
void Tensor::setConstant(VALUETYPE scalar)
{  
    std::fill(root_, root_ + size_, scalar);
}
int Tensor::setRandom()
{
    return 0;
}
int Tensor::getSize()const
{
    if(shape_ == nullptr)
        printf("getSize error. shape is null\n");
    return shape_[0] * shape_[1] * shape_[2] * shape_[3] * shape_[4];
}

//template<typename T>
int Tensor::rank() const
{
    return rank_;
}
//template<typename T>
VALUETYPE* Tensor::getData() const
{

    return root_;
}

//template<typename T>
Tensor Tensor::matMul1D(const Tensor& in_tensor) const
{
    Tensor result;
    if ( (rank_ == 0 ) && (in_tensor.rank_ > 0) )
    {
        VALUETYPE* ptr = in_tensor.rootAddress(0,0,0,0,0);
        int numOfElements = 1;
        for(int i =0 ; i< in_tensor.rank_ ; i++)
        {
            numOfElements *= in_tensor.shape_[i];
        }

        for(int i=0; i < numOfElements; i++)
        {
            result(i) = *(ptr+i) * (root(0));
        }
    }
    else if ( (in_tensor.rank_ == 0 ) && (rank_ > 0 )  )
    {
        VALUETYPE* ptr = rootAddress(0,0,0,0,0);
        int numOfElements = 1;
        for(int i =0 ; i< rank_ ; i++)
        {
            numOfElements *= shape_[i];
        }

        for(int i=0; i < numOfElements; i++)
        {
            *(ptr+i) = *(ptr+i) * (root(0));
        }
    }
    else
    {
        printf("Dimension erorr. Use matMul2D or matMul3D \n") ;
    }
    return result;
}

//template<typename T>
Tensor Tensor::matMul2D(const Tensor& in_tensor)const
{
    Tensor result(shape_[ROWIDX], in_tensor.shape_[COLIDX]);
    double sum = 0;
    if ( (rank_ == 2) && (in_tensor.rank_ == 2))
    {
        if (shape_[COLIDX] == in_tensor.shape_[ROWIDX])
        {
            for(int i = 0 ; i < shape_[ROWIDX] ; i ++ )
            {
                for(int k =0 ; k < in_tensor.shape_[COLIDX]; k++)
                {
                    sum =0 ;
                    for(int j = 0 ; j < shape_[4] ; j ++)
                    {
                        sum += root(i,j) * in_tensor.root(j,k);
                    }
                    result(i,k) = sum;
                }
            }
        }
        else{
            printf("Matrix shapes are not matched\n");
        }
    }
    else
    {
        printf("Dimension Error. Use matMul1D or matMul3D\n");
    }
    return result;
}

//template<typename T>
Tensor Tensor::matMul(const Tensor& in_tensor)const
{
    Tensor result(shape_[0], shape_[1], shape_[2], shape_[ROWIDX], in_tensor.shape_[COLIDX]);
    double sum = 0;

    #if SIMDENABLE
    Tensor in_T = in_tensor.transpose();
    #endif

    if ( (shape_[0] == in_tensor.shape_[0]) 
            && (shape_[1] == in_tensor.shape_[1])
            && (shape_[2] == in_tensor.shape_[2])
            && (shape_[COLIDX] == in_tensor.shape_[ROWIDX]) )
    {
        for(int d1_idx =0; d1_idx < shape_[0]; d1_idx++)
            for(int d2_idx =0; d2_idx < shape_[1]; d2_idx++)
                for(int d3_idx =0; d3_idx < shape_[2]; d3_idx++)
                    for(int i = 0 ; i < shape_[ROWIDX] ; i ++ )
                    {
                        int residual = 0;
                        for(int k =0 ; k < in_tensor.shape_[COLIDX]; k++)
                        {
                            sum =0 ;
                            #if SIMDENABLE
                            int j =0;
                            if ( shape_[4] / 4 == 0)
                            {
                                for (int l =0 ; l < shape_[4] / 4 ; l++)
                                {
                                    __m128 my_128;
                                    __m128 in_128;
                                    __m128 mulres; 
                                    __m128 res; 
                                    ALIGN float res_float[4] = {0};
                                    VALUETYPE local_sum =0;
                                    printShape();
                                    my_128 = _mm_load_ps((root_ + (shape_4_ *i) + j));
                                    in_T.printShape();
                                    in_128 = _mm_load_ps((in_T.root_ + (in_T.shape_4_ *k) + j));
                                    _mm_dp_ps(my_128, in_128,local_sum);
                                    sum += local_sum;
                                    j +=4;
                                    if( j >= shape_[4] )
                                        break;
                                }
                            }
                            for ( j ; j < shape_[4] ; j ++)
                            {
                                sum += root(i,j) * in_T.root(k,j);
                            }
                            #else
                            for(int j = 0 ; j < shape_[4] ; j ++)
                            {
                                sum += root(i,j) * in_tensor.root(j,k);
                            }
                            #endif
                            result(d1_idx, d2_idx, d3_idx, i , k) = sum;
                        }
                    }
    }
    else{
        printf("MatMul Error. Matrix shapes are not matched\n");
        printf("col == %d , row == %d\n ",shape_[COLIDX], in_tensor.shape_[ROWIDX]);
    }

    return result;
}

//template<typename T>
inline int Tensor::checkShape(const Tensor& in_tensor) const
{
    if( (shape_[0] == in_tensor.shape_[0]) 
        && (shape_[1] == in_tensor.shape_[1]) 
        && (shape_[2] == in_tensor.shape_[2]) 
        && (shape_[3] == in_tensor.shape_[3]) 
        && (shape_[4] == in_tensor.shape_[4]) )
        {
            return SHAPE_EQUAL;
        }
    else if ((shape_[0] == in_tensor.shape_[0]) 
        && (shape_[1] == in_tensor.shape_[1]) 
        && (shape_[2] == in_tensor.shape_[2]) 
        && (shape_[3] == in_tensor.shape_[3])
        && (shape_[4] == 1 ))
        {
            return SHAPE_4D_CONST;
        }
    else if ((shape_[0] == in_tensor.shape_[0]) 
        && (shape_[1] == in_tensor.shape_[1]) 
        && (shape_[2] == in_tensor.shape_[2]) 
        && (shape_[3] == 1 )
        && (shape_[4] == 1 ))
        {
            return SHAPE_3D_CONST;
        }
    else if ((shape_[0] == in_tensor.shape_[0]) 
        && (shape_[1] == in_tensor.shape_[1]) 
        && (shape_[2] == 1) 
        && (shape_[3] == 1 )
        && (shape_[4] == 1 ))
        {
            return SHAPE_2D_CONST;
        }
    else if ((shape_[0] == in_tensor.shape_[0]) 
        && (shape_[1] == 1) 
        && (shape_[2] == 1) 
        && (shape_[3] == 1 )
        && (shape_[4] == 1 ))
        {
            return SHAPE_1D_CONST;
        }
    else if ((shape_[0] == 1) 
        && (shape_[1] == 1) 
        && (shape_[2] == 1) 
        && (shape_[3] == 1 )
        && (shape_[4] == 1 ))
        {
            return SHAPE_0D_CONST;
        }
    else if ((shape_[0] == 1) 
        && (shape_[1] == 1) 
        && (shape_[2] == 1) 
        && (shape_[3] == 1 ) 
        && (shape_[4] == in_tensor.shape_[4]))
        {
            return SHAPE_1D_MATCH;
        }
    else if ((shape_[0] == 1) 
        && (shape_[1] == 1) 
        && (shape_[2] == 1) 
        && (shape_[3] == in_tensor.shape_[3]) 
        && (shape_[4] == in_tensor.shape_[4]))
        {
            return SHAPE_2D_MATCH;
        }
    else if ((shape_[0] == 1) 
        && (shape_[1] == 1) 
        && (shape_[2] == in_tensor.shape_[2]) 
        && (shape_[3] == in_tensor.shape_[3])
        && (shape_[4] == in_tensor.shape_[4]))
        {
            return SHAPE_3D_MATCH;
        }
    else if ((shape_[0] == 1) 
        && (shape_[1] == in_tensor.shape_[1]) 
        && (shape_[2] == in_tensor.shape_[2]) 
        && (shape_[3] == in_tensor.shape_[3]) 
        && (shape_[4] == in_tensor.shape_[4]))
        {
            return SHAPE_4D_MATCH;
        }
    else
    {   
        return SHAPE_UNMATCHED;
    }
}

//template<typename T>
Tensor Tensor::dotMul(const Tensor& in) const
{
    Tensor result;
    int status = true;
    VALUETYPE sum = 0;

    if ( (rank_ == 1) && (in.rank_==1))
    {
        if ( shape_[COLIDX] == in.shape_[COLIDX] )
        {
            #if SIMDENABLE
            int i =0;
            for (int l =0 ; l < shape_[COLIDX] / 4 ; l++)
            {
                __m128 my_128;
                __m128 in_128;
                __m128 mulres; 
                __m128 res; 
                ALIGN float res_float[4] = {0};
                VALUETYPE local_sum =0;
                my_128 = _mm_load_ps(root_ + i);
                in_128 = _mm_load_ps(in.root_ + i);
                _mm_dp_ps(my_128, in_128,local_sum); // substantial calculations.
                sum += local_sum;
                i +=4;
                if( i >= shape_[COLIDX] )
                    break;
            }
            for ( int i = shape_[COLIDX] / 4 * 4 ; i < shape_[COLIDX] ; i ++)
            {
                sum += root(i) * in.root(i);
            }
            #else
            for(int i =0 ; i< shape_[COLIDX]; i++)
            {
                sum += root(i) * in.root(i);
            }
            #endif
            status = true;
            result = sum;
        }
        else
        {
            printf("dot product error. dimension is not mathced\n ");
            status = false;
        }
    }
    else if ( (rank_ == 2) && (in.rank_==2))
    {
        if ( shape_[ROWIDX] == in.shape_[ROWIDX] )
        {
            #if SIMDENABLE
            int i =0;
            for (int l =0 ; l < shape_[ROWIDX] / 4 ; l++)
            {
                __m128 my_128;
                __m128 in_128;
                __m128 mulres; 
                __m128 res; 
                ALIGN float res_float[4] = {0};
                VALUETYPE local_sum =0;
                my_128 = _mm_load_ps((root_ + (shape_4_ *i) + 1));
                in_128 = _mm_load_ps((in.root_ + (in.shape_4_ *i) + 1));
                _mm_dp_ps(my_128, in_128,local_sum); // substantial calculations.
                sum += local_sum;
                i +=4;
                if( i >= shape_[ROWIDX] )
                    break;
            }
            for ( int i = shape_[ROWIDX] / 4 * 4 ; i < shape_[ROWIDX] ; i ++)
            {
                sum += root(i,1) * in.root(i,1);
            }
            #else
            for(int i =0 ; i< shape_[ROWIDX]; i++)
            {
                sum += root(i,1) * in.root(i,1);
            }
            #endif
            status = true;
            result = sum;
        }
        else
        {
            printf("dot product error. dimension is not mathced\n ");
            status = false;
        }
    }
    else
    {
        result = matMul(in);
    }



    return result;
}


//template<typename T>
Tensor Tensor::elementWise(const Tensor& A,const Tensor& B, int shape_status, int op )
{
    Tensor result;
    if (SHAPE_UNMATCHED)
    {
        printf("shape is not matched \n");
    }
    else if (SHAPE_EQUAL <= shape_status && shape_status <= SHAPE_0D_CONST  )
    {
        result.makeTensor(A.shape_[0],A.shape_[1],A.shape_[2],A.shape_[3],A.shape_[4]);
        int in_d1_idx = 0;
        for(int d1_idx = 0 ; d1_idx < A.shape_[0]; d1_idx++)
        {
            int in_d2_idx = 0;
            for( int d2_idx = 0; d2_idx < A.shape_[1] ;d2_idx++)
            {
                int in_d3_idx = 0;
                for( int d3_idx = 0; d3_idx < A.shape_[2] ;d3_idx++)
                {
                    int in_d4_idx = 0;
                    for( int d4_idx = 0; d4_idx < A.shape_[3] ; d4_idx++)
                    {
                        int in_d5_idx = 0;
                        for( int d5_idx = 0; d5_idx < A.shape_[4] ;d5_idx++)
                        {
                            if (op == SUM)
                            {
                                result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) + B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                            }
                            else if (op == SUB)
                            {
                                result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) - B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                            }
                            else if (op == MUL)
                            {
                                result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) * B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                            }
                            if (op == DIV)
                            {
                                if (B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) != 0)
                                    result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) / B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                                else
                                    printf("divide by 0\n");
                            }
                            if (shape_status == SHAPE_EQUAL)
                                in_d5_idx++;
                        }
                        if (shape_status <= SHAPE_4D_CONST)
                            in_d4_idx++;
                    }
                    if (shape_status <= SHAPE_3D_CONST)
                        in_d3_idx++;
                }
                if (shape_status <= SHAPE_2D_CONST)
                    in_d2_idx++;
            }
            if (shape_status <= SHAPE_1D_CONST)
                in_d1_idx++;
        }
    }

    else if (shape_status>= SHAPE_1D_MATCH)
    {
        result.makeTensor(A.shape_[0],A.shape_[1],A.shape_[2],A.shape_[3],A.shape_[4]);
        int in_d1_idx = 0;
        for(int d1_idx = 0 ; d1_idx < A.shape_[0]; d1_idx++)
        {
            int in_d2_idx = 0;
            for( int d2_idx = 0; d2_idx < A.shape_[1] ;d2_idx++)
            {
                int in_d3_idx = 0;
                for( int d3_idx = 0; d3_idx < A.shape_[2] ;d3_idx++)
                {
                    int in_d4_idx = 0;
                    for( int d4_idx = 0; d4_idx < A.shape_[3] ; d4_idx++)
                    {
                        int in_d5_idx = 0;
                        for( int d5_idx = 0; d5_idx < A.shape_[4] ;d5_idx++)
                        {
                            if (op == SUM)
                            {
                                result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) + B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                            }
                            else if (op == SUB)
                            {
                                result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) - B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                            }
                            else if (op == MUL)
                            {
                                result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) * B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                            }
                            if (op == DIV)
                            {
                                if (B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) != 0)
                                    result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = A(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) / B(in_d1_idx,in_d2_idx,in_d3_idx,in_d4_idx,in_d5_idx) ;
                                else
                                    printf("divide by 0\n");
                            }
                            if (shape_status >= SHAPE_1D_MATCH) //1D 2D 3D 4D
                                in_d5_idx++;
                        }
                        if (shape_status >= SHAPE_2D_MATCH) //2D 3D 4D
                            in_d4_idx++;
                    }
                    if (shape_status >= SHAPE_3D_MATCH) // 4D 3D
                        in_d3_idx++;
                }
                if (shape_status >= SHAPE_4D_MATCH) // only 4D ++
                    in_d2_idx++;
            }
                //in_d1_idx++; 
        }
    }
    else{
        printf("Shape mismath for Elemenet Wise\n ");
        result = 0;
    }
    return result;
}

inline Tensor Tensor::baseOp(double (*fp)(double ) ) const
{
    Tensor result;

    result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = fp((double)root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx));

    return result;
}

inline Tensor Tensor::baseOp(float p, double (*fp)(double, double ) ) const
{
    
    Tensor result;
    int shape[5] = {0};
    
    result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        result(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = fp((double)root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx), p);

    return result;
}
//template<typename T>
Tensor Tensor::reshape(int d1)
{
    Tensor result;
    int length = shape_[0] * shape_[1] * shape_[2] *shape_[3] * shape_[4];
    if (d1 == length)
    {
        int new_idx = 0;
        result.makeTensor(1,1,1,1,d1);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            result(new_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            new_idx ++;
                        }
    }
    else
    {
        printf("reshape error. input dimension is not matched! \n");
        result = -1;
    }
    return result;
}
//template<typename T>
Tensor Tensor::reshape(int d1, int d2)
{
    Tensor result;
    int length = shape_[0] * shape_[1] * shape_[2] *shape_[3] * shape_[4];
    if (d1 * d2 == length)
    {
        int new_idx = 0;
        result.makeTensor(1,1,1,d1,d2);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            result(new_idx / d2, new_idx % d2) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            new_idx ++;
                        }
    }
    else
    {
        printf("reshape error. input dimension is not matched! \n");
        result = -1;
    }
    return result;
}
//template<typename T>
Tensor Tensor::reshape(int d1, int d2, int d3)
{
    Tensor result;
    int length = shape_[0] * shape_[1] * shape_[2] *shape_[3] * shape_[4];
    if (d1 * d2 * d3 == length)
    {
        int d2_d3 = d2*d3;
        int new_idx = 0;
        result.makeTensor(1,1,d1,d2,d3);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            printf("[%d][%d][%d]\n",new_idx/d2 , (new_idx / d3) % d2, new_idx % d3);
                            result(new_idx/(d2_d3) , (new_idx / d3) % d2, new_idx % d3) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            new_idx ++;
                        }
    }
    else
    {
        printf("reshape error. input dimension is not matched! \n");
        result = -1;
    }
    return result;
}
//template<typename T>
Tensor Tensor::reshape(int d1, int d2, int d3, int d4)
{
    Tensor result;
    int length = shape_[0] * shape_[1] * shape_[2] *shape_[3] * shape_[4];
    if (d1 * d2 * d3 * d4 == length)
    {
        int d3_d4 = d3*d4;
        int d2_d3_d4 = d2*d3_d4;
        int new_idx = 0;
        result.makeTensor(1,d1,d2,d3,d4);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            result((new_idx/(d2_d3_d4)) , (new_idx/(d3_d4))%d2 , (new_idx / d4) % d3, new_idx % d4) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            new_idx ++;
                        }
    }
    else
    {
        printf("reshape error. input dimension is not matched! \n");
        result = -1;
    }
    return result;
}
//template<typename T>
Tensor Tensor::reshape(int d1, int d2, int d3, int d4, int d5)
{
    Tensor result;
    int length = shape_[0] * shape_[1] * shape_[2] *shape_[3] * shape_[4];
    if (d1 * d2 * d3 * d4 *d5 == length)
    {
        int d4_d5 = d4*d5;
        int d3_d4_d5 = d3*d4_d5;
        int d2_d3_d4_d5 = d2*d3_d4_d5;
        int new_idx = 0;
        result.makeTensor(d1,d2,d3,d4,d5);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            result( new_idx/(d2_d3_d4_d5) , (new_idx/(d3_d4_d5))%d2 , (new_idx/(d4_d5))%d3 , (new_idx / d5) % d4, new_idx % d5) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            new_idx ++;
                        }
    }
    else
    {
        printf("reshape error. input dimension is not matched! \n");
        result = -1;
    }
    return result;
}

//template<typename T>
Tensor Tensor::reshape(int *d)
{
    Tensor result;
    int length = shape_[0] * shape_[1] * shape_[2] *shape_[3] * shape_[4];
    if (d[0] * d[1] * d[2] * d[3] *d[4] == length)
    {
        int d4_d5 = d[3]*d[4];
        int d3_d4_d5 = d[2]*d4_d5;
        int d2_d3_d4_d5 = d[1]*d3_d4_d5;
        int new_idx = 0;
        result.makeTensor(d[0],d[1],d[2],d[3],d[4]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            result( new_idx/(d2_d3_d4_d5) , (new_idx/(d3_d4_d5))%d[1] , (new_idx/(d4_d5))%d[2] , (new_idx / d[4]) % d[3], new_idx % d[4]) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            new_idx ++;
                        }
    }
    else
    {
        printf("reshape error. input dimension is not matched! \n");
        result = -1;
    }
    return result;
}

//template<typename T>
VALUETYPE Tensor::sum() const
{
    VALUETYPE sum = 0;

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                    {
                        sum += root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                    }
    
    return sum;
}
//template<typename T>
Tensor Tensor::sum(int dim) const
{
    VALUETYPE sum = 0;

    Tensor result;

    if (rank_ > 2)
    {
        printf("summation dimension error. result is may not be correct \n");
    }

    if (dim == 0)
    {
        result.makeTensor(shape_[0],shape_[1],shape_[2], 1, shape_[4]);
    }
    else if(dim == 1)
    {
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],1);
    }
    else
    {
        printf("Expected parameter of sum(x) is 0 or 1 \n");
        result = -1;
        return result;
    }

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
    {
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
        {
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
            {
                if (dim == 0)
                {
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                    {
                        sum = 0;
                        for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        {
                            sum += root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                        }
                        result(d1_idx,d2_idx,d3_idx,0,d5_idx) = sum;
                    }
                }
                else if (dim == 1)
                {
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    {
                        sum = 0;
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            sum += root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                        }
                        result(d1_idx,d2_idx,d3_idx,d4_idx,0) = sum;
                    }
                }
                else
                {
                    printf("Expected parameter of sum(x) is 0 or 1 \n");
                }

            }
        }
    }
    
    return result;
}

int Tensor::argmax()const
{
    VALUETYPE temp_max = INT32_MIN;
    int midx = 0;

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                    {
                        if ( temp_max < root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx))
                        {
                            temp_max = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            midx = d5_idx;
                        }
                    }    
    return midx;
}
//template<typename T>
VALUETYPE Tensor::max()const
{
    VALUETYPE temp_max = INT32_MIN;

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                    {
                        if ( temp_max < root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx))
                        {
                            temp_max = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                        }
                    }
    
    return temp_max;
}
//template<typename T>
Tensor Tensor::max(int dim)const
{
    VALUETYPE temp_max = INT32_MIN;

    Tensor result;
    if (dim == 0)
    {
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],1);
    }
    else if(dim == 1)
    {
        result.makeTensor(shape_[0],shape_[1],shape_[2], 1, shape_[4]);
    }
    else
    {
        printf("Expected parameter of max(x) is 0 or 1 \n");
        result = -1;
        return result;
    }

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
    {
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
        {
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
            {
                if (dim == 0)
                {
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    {
                        temp_max = INT32_MIN;
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            if ( temp_max < root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx))
                            {
                                temp_max = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            }
                        }
                        result(d1_idx,d2_idx,d3_idx,d4_idx,0) = temp_max;
                    }
                }
                else if (dim == 1)
                {
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                    {
                        temp_max = INT32_MIN;
                        for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        {
                            if ( temp_max < root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx))
                            {
                                temp_max = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            }
                        }
                        result(d1_idx,d2_idx,d3_idx,0,d5_idx) = temp_max;
                    }
                }
                else
                {
                    printf("Expected parameter of max(x) is 0 or 1 \n");
                }

            }
        }
    }
    
    return result;
}
//template<typename T>
VALUETYPE Tensor::min()const
{
    VALUETYPE temp_min = INT32_MAX;

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                    {
                        if ( temp_min > root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx))
                        {
                            temp_min = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                        }
                    }
    
    return temp_min;
}
//template<typename T>
Tensor Tensor::min(int dim)const
{
    float temp_min = INT32_MAX;

    Tensor result;
    if (dim == 0)
    {
        result.makeTensor(shape_[0],shape_[1],shape_[2],shape_[3],1);
    }
    else if(dim == 1)
    {
        result.makeTensor(shape_[0],shape_[1],shape_[2], 1, shape_[4]);
    }
    else
    {
        printf("Expected parameter of max(x) is 0 or 1 \n");
        result = -1;
        return result;
    }

    for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
    {
        for( int d2_idx = 0; d2_idx < shape_[1] ;d2_idx++)
        {
            for( int d3_idx = 0; d3_idx < shape_[2] ;d3_idx++)
            {
                if (dim == 0)
                {
                    for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    {
                        temp_min = INT32_MIN;
                        for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                        {
                            if ( temp_min > root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx))
                            {
                                temp_min = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            }
                        }
                        result(d1_idx,d2_idx,d3_idx,d4_idx,0) = temp_min;
                    }
                }
                else if (dim == 1)
                {
                    for( int d5_idx = 0; d5_idx < shape_[4] ;d5_idx++)
                    {
                        temp_min = INT32_MAX;
                        for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                        {
                            if ( temp_min > root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx))
                            {
                                temp_min = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);
                            }
                        }
                        result(d1_idx,d2_idx,d3_idx,0,d5_idx) = temp_min;
                    }
                }
                else
                {
                    printf("Expected parameter of max(x) is 0 or 1 \n");
                }

            }
        }
    }
    
    return result;
}
//template<typename T>
Tensor Tensor::transpose() const
{
    Tensor result;
    if (rank_ == 0)
    {
        result = root(0);
    }
    else if (rank_ == 1)
    {
        result.makeTensor(1,1,1,shape_[4],1);
            for( int d5_idx = 0; d5_idx < shape_[4]; d5_idx++)
                result(d5_idx,0) = root(d5_idx);
    }
    else if (rank_==2)
    {
        result.makeTensor(1,1,1,shape_[4],shape_[3]);
        for( int d4_idx = 0; d4_idx < shape_[3]; d4_idx++)
            for( int d5_idx = 0; d5_idx < shape_[4]; d5_idx++)
                result(d5_idx,d4_idx) = root(d4_idx,d5_idx);
    }
    else if (rank_==3)
    {
        result.makeTensor(1,1,shape_[4],shape_[3],shape_[2]);
        for( int d3_idx = 0; d3_idx < shape_[2]; d3_idx++)
            for( int d4_idx = 0; d4_idx < shape_[3]; d4_idx++)
                for( int d5_idx = 0; d5_idx < shape_[4]; d5_idx++)
                    result(d5_idx,d4_idx,d3_idx) = root(d3_idx,d4_idx,d5_idx);
    }
    else if (rank_==4)
    {
        result.makeTensor(1,shape_[4],shape_[3],shape_[2],shape_[1]);
        for( int d2_idx = 0; d2_idx < shape_[1]; d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2]; d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3]; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4]; d5_idx++)
                        result(d5_idx,d4_idx,d3_idx,d2_idx) = root(d2_idx,d3_idx,d4_idx,d5_idx);
    }
    else if (rank_==5)
    {
        result.makeTensor(shape_[4],shape_[3],shape_[2],shape_[1],shape_[0]);
        for(int d1_idx = 0 ; d1_idx < shape_[0]; d1_idx++)
            for( int d2_idx = 0; d2_idx < shape_[1]; d2_idx++)
                for( int d3_idx = 0; d3_idx < shape_[2]; d3_idx++)
                    for( int d4_idx = 0; d4_idx < shape_[3]; d4_idx++)
                        for( int d5_idx = 0; d5_idx < shape_[4]; d5_idx++)
                            result(d5_idx,d4_idx,d3_idx,d2_idx,d1_idx) = root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx);

    }
    return result;
}

Tensor Tensor::flatten()
{
    return reshape(getSize());
}

Tensor Tensor::extract(int i, int j) 
{
    Tensor result;
    if (j == -1)
    {
        result.makeTensor(1,1,1,1,shape_[4]);
        std::copy(rootAddress(0,0,0,i,0),rootAddress(0,0,0,i,0) + shape_[4], result.rootAddress(0,0,0,0,0) );
    }
    return result;
}

Tensor Tensor::extract(int i, int j, int k)
{
    Tensor result;
    if (k == -1)
    {
        if (j == -1)
        {
            result.makeTensor(1,1,1,shape_[3],shape_[4]);
            std::copy(rootAddress(0,0,i,0,0),rootAddress(0,0,i,0,0) + shape_3_4_, result.rootAddress(0,0,0,0,0) );

        }
        else
        {
            result.makeTensor(1,1,1,1,shape_[4]);
            std::copy(rootAddress(0,0,i,j,0),rootAddress(0,0,i,j,0) + shape_4_, result.rootAddress(0,0,0,0,0) );
        }
    }
    else
    {
        printf("extract error\n");
    }
    return result;
}
Tensor Tensor::extract(int i, int j, int k, int l)
{
    Tensor result;
    if (l == -1)
    {
        if (k == -1)
        {
            if (j == -1)
            {
                result.makeTensor(1,1,shape_[2],shape_[3],shape_[4]);
                std::copy(rootAddress(0,i,0,0,0),rootAddress(0,i,0,0,0) + shape_2_3_4_, result.rootAddress(0,0,0,0,0) );
            }
            else
            {
                result.makeTensor(1,1,1,shape_[3],shape_[4]);
                std::copy(rootAddress(0,i,j,0,0),rootAddress(0,i,j,0,0) + shape_3_4_, result.rootAddress(0,0,0,0,0) );
            }
        }
        else
        {
            result.makeTensor(1,1,1,shape_[3],shape_[4]);
            std::copy(rootAddress(0,i,j,k,0),rootAddress(0,i,j,k,0) + shape_4_, result.rootAddress(0,0,0,0,0) );
        }
        
    }
    return result;
}
Tensor Tensor::extract(int i, int j, int k, int l, int m) 
{
    Tensor result;
    if(m == -1)
    {
        if (l == -1)
        {
            if (k == -1)
            {
                if (j == -1)
                {
                    result.makeTensor(1,shape_[1],shape_[2],shape_[3],shape_[4]);
                    std::copy(rootAddress(i,0,0,0,0),rootAddress(i,0,0,0,0) + shape_1_2_3_4_, result.rootAddress(0,0,0,0,0) );
                }
                else
                {
                    result.makeTensor(1,1,shape_[2],shape_[3],shape_[4]);
                    std::copy(rootAddress(i,j,0,0,0),rootAddress(i,j,0,0,0) + shape_2_3_4_, result.rootAddress(0,0,0,0,0) );
                }
            }
            else
            {
                result.makeTensor(1,1,shape_[2],shape_[3],shape_[4]);
                std::copy(rootAddress(i,j,k,0,0),rootAddress(i,j,k,0,0) + shape_3_4_, result.rootAddress(0,0,0,0,0) );
            }
        }
        else
        {
            result.makeTensor(1,1,shape_[2],shape_[3],shape_[4]);
            std::copy(rootAddress(i,j,k,l,0),rootAddress(i,j,k,l,0) + shape_4_, result.rootAddress(0,0,0,0,0) );
        }
    }
    else{
        printf("extract error\n");
    }

    return result;
}

//template<typename T>
Tensor Tensor::log()
{
    return baseOp(std::log);
}
//template<typename T>
Tensor Tensor::log10()
{
    return baseOp(std::log10);
}
//template<typename T>
Tensor Tensor::exp()
{
    return baseOp(std::exp);
}
//template<typename T>
Tensor Tensor::pow(float p)
{
    return baseOp(p, std::pow);
}

inline void Tensor::printShape()const
{
    if ( rank_ == 0)
    {
        printf("(scalar)\n");
    }
    else if ( rank_ == 1)
    {
        printf("(%d)\n", shape_[4]);
    }
    else if ( rank_ == 2)
    {
        printf("(%d,%d)\n", shape_[3],shape_[4]);
    }
    else if ( rank_ == 3)
    {
        printf("(%d,%d,%d)\n", shape_[2],shape_[3],shape_[4]);
    }
    else if ( rank_ == 4)
    {
        printf("(%d,%d,%d,%d)\n", shape_[1],shape_[2],shape_[3],shape_[4]);
    }
    else if ( rank_ == 5)
    {
        printf("(%d,%d,%d,%d,%d)\n", shape_[0],shape_[1],shape_[2],shape_[3],shape_[4]);
    }
    else
    {
        printf("printShape error. Not Supported Error\n");
    }
}


#endif