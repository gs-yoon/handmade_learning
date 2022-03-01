#ifndef __TENSOR_H__
#define __TENSOR_H__

#include<stdio.h>
#include<iostream>

#define TENSORDEBUG 0
#define DEFAULTMAXDIM 5
#define ROWIDX 3
#define COLIDX 4

typedef float SCALARTYPE;

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

struct DimensionType
{
    /* data */
    int* shape;
    int rank;
};

template<typename T = float>
class Tensor
{
private:
    T***** root_ = nullptr;

    int* shape_ = nullptr;
    int row_ =0;
    int col_ =0;
    int rank_ = 0;
    int valid_ = 0;

private:
    Tensor<T> matMul1D(Tensor<T>& );
    Tensor<T> matMul2D(Tensor<T>& );
    Tensor<T> matMul3D(Tensor<T>& );
    Tensor<T> elementWise( Tensor<T>&, Tensor<T>& ,int ,int);
    void makeTensor(int d1, int d2, int d3, int d4, int d5);
    void breakTensor();
    int checkShape(const Tensor<T>&);

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
    int getShape(int dim);
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

    Tensor<T> operator +(Tensor<T>& in) 
    {
        Tensor<T> ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, SUM);
        shape_status = checkShape(in) ;
        if (shape_status >0)
            ret = elementWise(in,*this, shape_status, SUM);
        return ret;
    }

    Tensor<T> operator *(Tensor<T>& in) 
    {
        Tensor<T> ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, MUL);
        shape_status = checkShape(in) ;
        if (shape_status >0)
            ret = elementWise(in,*this, shape_status, MUL);
        return ret;
    }
    Tensor<T> operator -(Tensor<T>& in) 
    {
        Tensor<T> ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, SUB);
        shape_status = checkShape(in) ;
        if (shape_status >0)
            ret = elementWise(in,*this, shape_status, SUB);
        return ret;
    }
    Tensor<T> operator /(Tensor<T>& in) 
    {
        Tensor<T> ret;
        int shape_status = in.checkShape(*this);
        if (shape_status >0)
            ret = elementWise(*this, in, shape_status, DIV);
        shape_status = checkShape(in) ;
        if (shape_status >0)
            ret = elementWise(in,*this, shape_status, DIV);
        return ret;
    }

    Tensor<T>& operator=(const Tensor<T>& cp);
    Tensor<T>& operator=(const SCALARTYPE scalar);
    Tensor<T>& operator=(const int scalar);
    Tensor<T>& operator=(const bool scalar);

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
    bool operator<(const Tensor<T> t)
    {
        if (rank_ == 0)
            return root(0) < t.root(0); 
        else
            return false;
    }
    bool operator>(const Tensor<T> t)
    {
        if (rank_ == 0)
            return root(0) > t.root(0); 
        else
            return false;
    }


};

template<typename T>
Tensor<T>::Tensor()
{
    valid_ = 0;
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

    for(int d1_idx = 0 ; d1_idx < shape_[0] ; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ; d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ; d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ; d5_idx++)
                        root(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = cp.root_[d1_idx][d2_idx][d3_idx][d4_idx][d5_idx] ;
}

//assign constructor
template<typename T>
Tensor<T>& Tensor<T>::operator=(const SCALARTYPE scalar)
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
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const int scalar)
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
}
template<typename T>
Tensor<T>& Tensor<T>::operator=(const bool scalar)
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
}
template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& cp)
{
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
        #if TENSORDEBUG
        printf("Rebuilding Tensor...\n");
        #endif
        breakTensor();
        makeTensor(cp.shape_[0],cp.shape_[1],cp.shape_[2],cp.shape_[3],cp.shape_[4]);
    }

    rank_ = cp.rank_;
    for( int i =0 ;i < 5 ; i++)
    {
        shape_[i] = cp.shape_[i];
    }
    printf("\n ");

    for(int d1_idx = 0 ; d1_idx < shape_[0] ; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape_[1] ; d2_idx++)
            for( int d3_idx = 0; d3_idx < shape_[2] ; d3_idx++)
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape_[4] ; d5_idx++)
                        root_[d1_idx][d2_idx][d3_idx][d4_idx][d5_idx] = cp.root_[d1_idx][d2_idx][d3_idx][d4_idx][d5_idx] ;

    return *this;
}

template<typename T>
inline void Tensor<T>::makeTensor(int d1, int d2, int d3, int d4, int d5)
{
    if (valid_ == 1)
        breakTensor();

    valid_ = 1;
    g_make_cnt ++;
    root_ = new T****[d1];
    shape_ = new int[DEFAULTMAXDIM];

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
    shape_[0] = d1;
    shape_[1] = d2;
    shape_[2] = d3;
    shape_[3] = d4;
    shape_[4] = d5;
    row_ = d4;
    col_ = d5;

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

template<typename T>
Tensor<T>::~Tensor()
{
    breakTensor();
}

template<typename T>
void Tensor<T>::breakTensor()
{
    g_delete_cnt ++;
    valid_ = 0;

    for(int d1_idx = 0 ; d1_idx < shape_[0] ; d1_idx++)
    {
        for( int d2_idx = 0; d2_idx < shape_[1] ; d2_idx++)
        {
            for( int d3_idx = 0; d3_idx < shape_[2] ; d3_idx++)
            {
                for( int d4_idx = 0; d4_idx < shape_[3] ; d4_idx++)
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

    if (shape_ != nullptr)
    {
        delete[] shape_;
        shape_ = nullptr;
    }
}

template<typename T>
int Tensor<T>::getShape(int dim)
{
    if (dim <= 5)
    {
        return shape_[dim];
    }
    else
    {
        printf("input parameter of getShape is out of bound\n");
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
    Tensor<T> result;
    if ( (rank_ == 0 ) && (in_tensor.rank_ > 0) )
    {
        T* ptr = in_tensor.root_[0][0][0][0];
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
        T* ptr = root_[0][0][0][0];
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
}

template<typename T>
Tensor<T> Tensor<T>::matMul2D(Tensor<T>& in_tensor)
{
    Tensor<T> result(shape_[ROWIDX], in_tensor.shape_[COLIDX]);
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

template<typename T>
Tensor<T> Tensor<T>::matMul3D(Tensor<T>& in_tensor)
{
    Tensor<T> result(shape_[0], shape_[1], shape_[2], shape_[ROWIDX], in_tensor.shape_[COLIDX]);
    double sum = 0;
    if ((rank_ >=3) && (in_tensor.rank_>=3))
    {
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
                            for(int k =0 ; k < in_tensor.shape_[COLIDX]; k++)
                            {
                                sum =0 ;
                                for(int j = 0 ; j < shape_[4] ; j ++)
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
inline int Tensor<T>::checkShape(const Tensor<T>& in_tensor)
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

template<typename T>
Tensor<T> Tensor<T>::dotMul(Tensor<T>& in)
{
    Tensor<T> result(1);
    int status = true;
    SCALARTYPE sum = 0;

    if ( (rank_ == 1) && (in.rank_==1))
    {
        if ( shape_[COLIDX] == in.shape_[COLIDX] )
        {
            for(int i =0 ; i< shape_[COLIDX]; i++)
            {
                sum += root(i) * in.root(i);
            }
            status = true;
        }
        else
        {
            status = false;
        }
    }
    else if ( (rank_ == 2) && (in.rank_==2))
    {
        if ( shape_[ROWIDX] == in.shape_[ROWIDX] )
        {
            for(int i =0 ; i< shape_[ROWIDX]; i++)
            {
                sum += root(i,1) * in.root(i,1);
            }
            status = true;
        }
        else
        {
            status = false;
        }
    }
    else
    {
        status = false;
    }


    if (status == true)
    {
        result = sum;
    }
    else
    {
        result = status;
    }

    return result;
}


template<typename T>
Tensor<T> Tensor<T>::elementWise( Tensor<T>& A,  Tensor<T>& B, int shape_status, int op )
{
    Tensor<T> result;
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
    int shape[5] = {0};
    shape[0] = t.getShape(0);
    shape[1] = t.getShape(1);
    shape[2] = t.getShape(2);
    shape[3] = t.getShape(3);
    shape[4] = t.getShape(4);

    for(int d1_idx =0; d1_idx < shape[0]; d1_idx++)
    {
        os << "[";
        for(int d2_idx =0; d2_idx < shape[1]; d2_idx++)
        {
            os << "[";
            for(int d3_idx =0; d3_idx < shape[2]; d3_idx++)
            {
                os << "[";
                for(int d4_idx = 0 ; d4_idx < shape[3] ; d4_idx++ )
                {
                    if (d4_idx >0)
                        os << "   ";
                    os << "[ ";
                    for(int d5_idx = 0 ; d5_idx < shape[4] ; d5_idx++ )
                    {
                        if (d5_idx >0)
                            os << " ";
                        os << t(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) << " ";
                    }
                    os << "]";
                    if( (t_rank > 1) && (d4_idx != shape[3] -1) )
                        os << "\n";
                }
                os << "]";
                if( (t_rank > 2) && (d3_idx != shape[2] -1) )
                    os << "\n";
            }
            os << "]";
            if( (t_rank > 3)&& (d2_idx != shape[1] -1) )
                os << "\n";
        }
        os << "]";
        if( (t_rank > 4)&& (d1_idx != shape[0] -1) )
            os << "\n";
    }
    return os;
}

#endif