#ifndef __TENSOR_H__
#define __TENSOR_H__

#include<stdio.h>
#include<iostream>

#define TENSORDEBUG 1
#define DEFAULTMAXDIM 5
#define ROWIDX 3
#define COLIDX 4

long long g_delete_cnt =0;
long long g_make_cnt =0;

struct DimensionType
{
    /* data */
    int* dims;
    int rank;
};

template<typename T = float>
class Tensor
{
private:
    T***** root_ = nullptr;

    int* dims_ = nullptr;
    int row_ =0;
    int col_ =0;
    int rank_ = 0;

private:

    Tensor<T> matMul1D(Tensor<T>& );
    Tensor<T> matMul2D(Tensor<T>& );
    Tensor<T> matMul3D(Tensor<T>& );
    void makeTensor(int d1, int d2, int d3, int d4, int d5);
    void breakTensor();

    T& root(){ return root_[0][0][0][0][0]; }
    T& root(int i){ return root_[0][0][0][0][i]; }
    T& root(int i, int j){ return root_[0][0][0][i][j]; }
    T& root(int i, int j, int k){ return root_[0][0][i][j][k]; }
    T& root(int i, int j, int k, int l){ return root_[0][i][j][k][l]; }
    T& root(int i, int j, int k, int l, int m){ return root_[i][j][k][l][m]; }

public:
    Tensor();
    Tensor(int d1);
    Tensor(int d1, int d2);
    Tensor(int d1, int d2, int d3);
    Tensor(int d1, int d2, int d3, int d4);
    Tensor(int d1, int d2, int d3, int d4, int d5);
    Tensor(const Tensor<T>& cp);
    Tensor(DimensionType);
    ~Tensor();
    int getDims(int dim);
    int setVal();
    int setConstant();
    int setRandom();
    int rank();
    T* getData();
    Tensor<T> matMul(Tensor<T>& );
    Tensor<T> dotMul(Tensor<T>& );
    Tensor<T> reshape(int);
    Tensor<T> reshape(int, int);
    Tensor<T> reshape(int, int, int);
    Tensor<T> reshape(int, int, int, int);
    Tensor<T> reshape(DimensionType d);
    T maximum();
    Tensor<T> maximum(int dim);
    T minimum();
    Tensor<T> minimum(int dim);
    Tensor<T> transpose();
    Tensor<T> log();
    Tensor<T> log10();
    Tensor<T> exp();
    Tensor<T> pow(int p);

    T& operator()(int i) { return root(i); }
    T& operator()(int i, int j) { return root(i,j); }
    T& operator()(int i, int j, int k) { return root(i,j,k); }
    T& operator()(int i, int j, int k, int l) { return root(i,j,k,l); }
    T& operator()(int i, int j, int k, int l, int m) { return root(i,j,k,l,m); }

    Tensor<T>& operator=(const Tensor<T>& cp);

    //operator +
    //operator *
    //operator -
    //operator /
    //operator !=
    //operator ==
    //operator <
    //operator >
};

template<typename T>
Tensor<T>::Tensor()
{
    makeTensor(0,0,0,0,0);
}
template<typename T>
Tensor<T>::Tensor(int d1)
{
    makeTensor(1,1,1,1,d1);
}
template<typename T>
Tensor<T>::Tensor(int d1, int d2)
{
    makeTensor(1,1,1,d1,d2);
}
template<typename T>
Tensor<T>::Tensor(int d1, int d2, int d3)
{
    makeTensor(1,1,d1,d2,d3);
}
template<typename T>
Tensor<T>::Tensor(int d1, int d2, int d3, int d4)
{
    makeTensor(1,d1,d2,d3,d4);
}
template<typename T>
Tensor<T>::Tensor(int d1, int d2, int d3, int d4, int d5)
{
    makeTensor(d1,d2,d3,d4,d5);
}

//copy constructor
template<typename T>
Tensor<T>::Tensor(const Tensor<T>& cp)
{
    #if TENSORDEBUG
    printf("\ncopy\n");
    #endif 

    if ((root_ == nullptr) && (dims_== nullptr))
    {
        makeTensor(cp.dims_[0],cp.dims_[1],cp.dims_[2],cp.dims_[3],cp.dims_[4]);
    }

    else if ((root_ == nullptr) || (dims_== nullptr))
    {
        #if TENSORDEBUG
        printf("Critical constructor error\n");
        printf("Rebuilding Tensor...\n");
        #endif 
        breakTensor();
        makeTensor(cp.dims_[0],cp.dims_[1],cp.dims_[2],cp.dims_[3],cp.dims_[4]);
    }
    else{
        #if TENSORDEBUG
        printf("Rebuilding Tensor...\n");
        #endif 
        breakTensor();
        makeTensor(cp.dims_[0],cp.dims_[1],cp.dims_[2],cp.dims_[3],cp.dims_[4]);
    }

    rank_ = cp.rank_;
    for( int i =0 ;i <= 5 ; i++)
    {
        dims_[i] = cp.dims_[i];
    }

    for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
            for( int d3_idx = 0; d3_idx < dims_[2] ; d3_idx++)
                for( int d4_idx = 0; d4_idx < dims_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < dims_[4] ; d5_idx++)
                        root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = cp.root_[d1_idx][d2_idx][d3_idx][d4_idx][d5_idx] ;
}

//assing constructor
template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& cp)
{
    #if TENSORDEBUG
    printf("\n====\n");
    #endif
    if ((root_ == nullptr) && (dims_== nullptr))
    {
        makeTensor(cp.dims_[0],cp.dims_[1],cp.dims_[2],cp.dims_[3],cp.dims_[4]);
    }

    else if ((root_ == nullptr) || (dims_== nullptr))
    {
        #if TENSORDEBUG
        printf("Critical constructor error\n");
        printf("Rebuilding Tensor...\n");
        #endif
        breakTensor();
        makeTensor(cp.dims_[0],cp.dims_[1],cp.dims_[2],cp.dims_[3],cp.dims_[4]);
    }
    else{
        #if TENSORDEBUG
        printf("Rebuilding Tensor...\n");
        #endif
        breakTensor();
        makeTensor(cp.dims_[0],cp.dims_[1],cp.dims_[2],cp.dims_[3],cp.dims_[4]);
    }

    rank_ = cp.rank_;
    for( int i =0 ;i < 5 ; i++)
    {
        dims_[i] = cp.dims_[i];
    }
    printf("\n ");

    for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
            for( int d3_idx = 0; d3_idx < dims_[2] ; d3_idx++)
                for( int d4_idx = 0; d4_idx < dims_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < dims_[4] ; d5_idx++)
                        root_[d1_idx][d2_idx][d3_idx][d4_idx][d5_idx] = cp.root_[d1_idx][d2_idx][d3_idx][d4_idx][d5_idx] ;

    return *this;
}

template<typename T>
inline void Tensor<T>::makeTensor(int d1, int d2, int d3, int d4, int d5)
{
    g_make_cnt ++;
    root_ = new T****[d1];
    dims_ = new int[DEFAULTMAXDIM];

    for(int idx = 0 ; idx < d1 ; idx ++)
    {
        root_[idx] = new T***[d2];
    }
    
    for(int ri = 0 ; ri < d1 ; ri++)
    {
        for( int ci = 0; ci < d2 ; ci++)
        {
            root_[ri][ci] = new T**[d3];
        }
    }

    for(int ri = 0 ; ri < d1 ; ri++)
    {
        for( int ci = 0; ci < d2 ; ci++)
        {
            for( int di = 0; di < d3 ; di++)
            {
                root_[ri][ci][di] = new T*[d4];
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
                    root_[ri][ci][di][d4i] = new T[d5];
            }
        }
    }
    dims_[0] = d1;
    dims_[1] = d2;
    dims_[2] = d3;
    dims_[3] = d4;
    dims_[4] = d5;
    row_ = d4;
    col_ = d5;
    if (d5 <= 1)
        rank_ = 0;
    else if (d4 <= 1)
        rank_ = 1;
    else if (d3 <= 1)
        rank_ = 2;
    else if (d2 <= 1)
        rank_ = 3;
    else if (d1 <= 1)
        rank_ = 4;
    else
        rank_ = 5;
}

template<typename T>
Tensor<T>::~Tensor()
{
    breakTensor();
}

template<typename T>
void Tensor<T>::breakTensor()
{
    g_delete_cnt ++;
    for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
    {
        for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
        {
            for( int d3_idx = 0; d3_idx < dims_[2] ; d3_idx++)
            {
                for( int d4_idx = 0; d4_idx < dims_[3] ; d4_idx++)
                {
                    if (root_[d1_idx][d2_idx][d3_idx][d4_idx] != nullptr)
                    {
                        delete[] root_[d1_idx][d2_idx][d3_idx][d4_idx];
                        root_[d1_idx][d2_idx][d3_idx][d4_idx] = nullptr;
                    }

                }
                if (root_[d1_idx][d2_idx][d3_idx] != nullptr)
                {
                    delete[] root_[d1_idx][d2_idx][d3_idx];
                    root_[d1_idx][d2_idx][d3_idx] = nullptr;
                }
            }
            if (root_[d1_idx][d2_idx] != nullptr)
            {
                delete[] root_[d1_idx][d2_idx];
                root_[d1_idx][d2_idx] = nullptr;
            }
        }
        if (root_[d1_idx] != nullptr)
        {
            delete[] root_[d1_idx];
            root_[d1_idx] = nullptr;
        }
    }
    if (root_ != nullptr)
        {
            delete[] root_;
            root_ = nullptr;
        }

    if (dims_ != nullptr)
    {
        delete[] dims_;
        dims_ = nullptr;
    }
}

template<typename T>
int Tensor<T>::getDims(int dim)
{
    if (dim <= 5)
    {
        return dims_[dim];
    }
    else
    {
        printf("input parameter of getDims is out of bound\n");
        return -1;
    }
}
template<typename T>
int Tensor<T>::rank()
{
    return rank_;
}
template<typename T>
T* Tensor<T>::getData()
{
    if (rank_ == 1) return root_[0][0][0][0];
    else if (rank_ == 2) return root_[0][0][0];
    else if (rank_ == 3) return root_[0][0];
    else if (rank_ == 4) return root_[0];
    else if (rank_ == 5) return root_;
}
template<typename T>
Tensor<T> Tensor<T>::matMul(Tensor<T>& in_tensor)
{
    Tensor<T> result;

    if ( (rank_ ==0 ) || (in_tensor.rank_ ==0) )
        return matMul1D(in_tensor);
    else if ((rank_ ==2 ) && (in_tensor.rank_ == 2))
        return matMul2D(in_tensor);
    else if ((rank_ >=3 ) && (in_tensor.rank_ >= 3))
        return matMul3D(in_tensor);
    else
        printf("matmul is not defined for the dimension");
    //return result;
}


template<typename T>
Tensor<T> Tensor<T>::matMul1D(Tensor<T>& in_tensor)
{
    if ( (rank_ == 0 ) && (in_tensor.rank_ > 0) )
    {
        T* ptr = in_tensor.root_[0][0][0][0];
        int numOfElements = 1;
        for(int i =0 ; i< in_tensor.rank_ ; i++)
        {
            numOfElements *= in_tensor.dims_[i];
        }

        for(int i=0; i < numOfElements; i++)
        {
            *(ptr+i) = *(ptr+i) * (root(0));
        }
    }
    else if ( (in_tensor.rank_ == 0 ) && (rank_ > 0 )  )
    {
        T* ptr = root_[0][0][0][0];
        int numOfElements = 1;
        for(int i =0 ; i< rank_ ; i++)
        {
            numOfElements *= dims_[i];
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
}

template<typename T>
Tensor<T> Tensor<T>::matMul2D(Tensor<T>& in_tensor)
{
    Tensor<T> result(dims_[ROWIDX], in_tensor.dims_[COLIDX]);
    double sum = 0;
    if ( (rank_ == 2) && (in_tensor.rank_ == 2))
    {
        if (dims_[COLIDX] == in_tensor.dims_[ROWIDX])
        {
            for(int i = 0 ; i < dims_[ROWIDX] ; i ++ )
            {
                for(int k =0 ; k < in_tensor.dims_[COLIDX]; k++)
                {
                    sum =0 ;
                    for(int j = 0 ; j < dims_[4] ; j ++)
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

template<typename T>
Tensor<T> Tensor<T>::matMul3D(Tensor<T>& in_tensor)
{
    Tensor<T> result(dims_[0], dims_[1], dims_[2], dims_[ROWIDX], in_tensor.dims_[COLIDX]);
    double sum = 0;
    if ((rank_ >=3) && (in_tensor.rank_>=3))
    {
        if ( (dims_[0] == in_tensor.dims_[0]) 
                && (dims_[1] == in_tensor.dims_[1])
                && (dims_[2] == in_tensor.dims_[2])
                && (dims_[COLIDX] == in_tensor.dims_[ROWIDX]) )
        {
            for(int d1_idx =0; d1_idx < dims_[0]; d1_idx++)
                for(int d2_idx =0; d2_idx < dims_[1]; d2_idx++)
                    for(int d3_idx =0; d3_idx < dims_[2]; d3_idx++)
                        for(int i = 0 ; i < dims_[ROWIDX] ; i ++ )
                        {
                            for(int k =0 ; k < in_tensor.dims_[COLIDX]; k++)
                            {
                                sum =0 ;
                                for(int j = 0 ; j < dims_[4] ; j ++)
                                {
                                    sum += root(i,j) * in_tensor.root(j,k);
                                }
                                result(d1_idx, d2_idx, d3_idx, i , k) = sum;
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

template<typename T>
Tensor<T> Tensor<T>::dotMul(Tensor<T>& in_tensor)
{
    
}
template<typename T>
Tensor<T> Tensor<T>::reshape(int)
{}
template<typename T>
Tensor<T> Tensor<T>::reshape(int, int)
{}
template<typename T>
Tensor<T> Tensor<T>::reshape(int, int, int)
{}
template<typename T>
Tensor<T> Tensor<T>::reshape(int, int, int, int)
{}
template<typename T>
Tensor<T> Tensor<T>::reshape(DimensionType)
{}
template<typename T>
T Tensor<T>::maximum()
{}
template<typename T>
Tensor<T> Tensor<T>::maximum(int dim)
{}
template<typename T>
T Tensor<T>::minimum()
{}
template<typename T>
Tensor<T> Tensor<T>::minimum(int dim)
{}
template<typename T>
Tensor<T> Tensor<T>::transpose()
{}
template<typename T>
Tensor<T> Tensor<T>::log()
{}
template<typename T>
Tensor<T> Tensor<T>::log10()
{}
template<typename T>
Tensor<T> Tensor<T>::exp()
{}
template<typename T>
Tensor<T> Tensor<T>::pow(int p)
{}

//matrix product

//dot product

//add

//subtract

//reshape

//log

//exp, pow

template <typename T>
std::ostream& operator<<(std::ostream& os, Tensor<T>& t)
{
    int t_rank = t.rank();
    int dims[5] = {0};
    dims[0] = t.getDims(0);
    dims[1] = t.getDims(1);
    dims[2] = t.getDims(2);
    dims[3] = t.getDims(3);
    dims[4] = t.getDims(4);

    for(int d1_idx =0; d1_idx < dims[0]; d1_idx++)
    {
        os << "[";
        for(int d2_idx =0; d2_idx < dims[1]; d2_idx++)
        {
            os << "[";
            for(int d3_idx =0; d3_idx < dims[2]; d3_idx++)
            {
                os << "[";
                for(int d4_idx = 0 ; d4_idx < dims[3] ; d4_idx++ )
                {
                    if (d4_idx >0)
                        os << "   ";
                    os << "[ ";
                    for(int d5_idx = 0 ; d5_idx < dims[4] ; d5_idx++ )
                    {
                        if (d5_idx >0)
                            os << " ";
                        os << t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) << " ";
                    }
                    os << "]";
                    if( (t_rank > 1)&& (d4_idx != dims[3] -1) )
                        os << "\n";
                }
                os << "]";
                if( (t_rank > 2)&& (d3_idx != dims[2] -1) )
                    os << "\n";
            }
            os << "]";
            if( (t_rank > 3)&& (d2_idx != dims[1] -1) )
                os << "\n";
        }
        os << "]";
        if( (t_rank > 4)&& (d1_idx != dims[0] -1) )
            os << "\n";
    }
    return os;
}

#endif