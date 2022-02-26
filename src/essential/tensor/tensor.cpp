#include<stdio.h>

#define DEFAULTMAXDIM 4

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
    T* root_;
    int* dims_;
    int rank_;

private:
    Tensor<T> matMul1D(Tensor<T>& );
    Tensor<T> matMul2D(Tensor<T>& );
    Tensor<T> matMul3D(Tensor<T>& );

public:
    Tensor();
    Tensor(int d1);
    Tensor(int d1, int d2);
    Tensor(int d1, int d2, int d3);
    Tensor(int d1, int d2, int d3, int d4);
    Tensor(Tensor<T>& cp);
    Tensor(DimensionType);
    ~Tensor();
    int getDims(int dim);
    int setVal();
    int setConstant();
    int setRandom();
    int rank();
    T* getData(int);
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
    rank_ = 0;
    root_ = new int[1];
    dims_ = new int[DEFAULTMAXDIM];
    dims_[0] = 0;
}
template<typename T>
Tensor<T>::Tensor(int d1)
{
    rank_ = 1;
    root_ = new T[d1];
    dims_ = new int[DEFAULTMAXDIM];
    dims_[0] = d1;
}
template<typename T>
Tensor<T>::Tensor(int d1, int d2)
{
    rank_ = 1;
    root_ = new T[d1];
    dims_ = new int[DEFAULTMAXDIM];
    dims_[0] = d1;

    for(int idx = 0 ; idx < d1 ; idx ++)
    {
        root_[idx] = new T[d2];
    }
    dims_[1] = d2;
    rank_ = 2;
}
template<typename T>
Tensor<T>::Tensor(int d1, int d2, int d3)
{
    Tensor(d1, d2);
    for(int ri = 0 ; ri < d1 ; ri++)
    {
        for( int ci = 0; ci < d2 ; ci++)
        {
            root_[ri][ci] = new T[d3];
        }
    }
    dims_[2] = d3;
    rank_ = 3;
}
template<typename T>
Tensor<T>::Tensor(int d1, int d2, int d3, int d4)
{
    Tensor(d1, d2, d3);
    for(int ri = 0 ; ri < d1 ; ri++)
    {
        for( int ci = 0; ci < d2 ; ci++)
        {
            for( int di = 0; di < d3 ; di++)
            {
                root_[ri][ci][di] = new T[d4];
            }
        }
    }
    dims_[3] = d4;
    rank_ = 4;
}
template<typename T>
Tensor<T>::Tensor(DimensionType dims)
{
    dims_ = new int[dims.rank];
    for (int i =0 ; i < dims.rank ; i ++)
    {
        dims_[i] = dims.dims[i];
    }

    //Not Supported
}
template<typename T>
Tensor<T>::Tensor(Tensor& cp)
{
    rank_ = cp.rank_;
    for( int i =0 ;i < rank_ ; i++)
    {
        dims_[i] = cp.dims_[i];
    }

    if(rank_ == 4)
    {
        for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        {
            for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
            {
                for( int d3_idx = 0; d3_idx < dims_[2] ; d3_idx++)
                {
                    for( int d4_idx = 0; d4_idx < dims_[3] ; d4_idx++)
                    {
                        root_[d1_idx][d2_idx][d3_idx][d4_idx] = cp.root[d1_idx][d2_idx][d3_idx][d4_idx];
                    }
                }
            }
        }
    }
    
    else if(rank_ == 3)
    {
        for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        {
            for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
            {
                for( int d3_idx = 0; d3_idx < dims_[2] ; d3_idx++)
                {
                    root_[d1_idx][d2_idx][d3_idx] = cp.root[d1_idx][d2_idx][d3_idx];
                }
            }
        }
    }
    else if(rank_ == 2)
    {
        for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        {
            for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
            {
                root_[d1_idx][d2_idx] = cp.root[d1_idx][d2_idx];
            }
        }
    }
    else if(rank_ == 1)
    {
        for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        {
            root_[d1_idx] = cp.root[d1_idx];
        }
    }
    else if(rank_ == 0)
    {
        root_[0] = cp.root[0];

    }
    else
    {
        printf("high-multidimension grater than 4d is not supproted\n");
    }
}
template<typename T>
Tensor<T>::~Tensor()
{
    if (root_ != nullptr)
        delete[] root_;
    if (dims_ != nullptr)
        delete[] dims_;

    if(rank_ == 4)
    {
        for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        {
            for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
            {
                for( int d3_idx = 0; d3_idx < dims_[2] ; d3_idx++)
                {
                    delete[] root_[d1_idx][d2_idx][d3_idx];
                }
            }
        }
    }

    else if(rank_ == 3)
    {
        for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        {
            for( int d2_idx = 0; d2_idx < dims_[1] ; d2_idx++)
            {
                delete[] root_[d1_idx][d2_idx];
            }
        }
    }
    
    else if(rank_ == 2)
    {
        for(int d1_idx = 0 ; d1_idx < dims_[0] ; d1_idx++)
        {
            delete[] root_[d1_idx];
        }
    }
    else if(rank_ == 1 || rank_ == 0)
    {
        delete[] root_;
    }
    else
    {
        printf("not supporeted\n");
    }
}
template<typename T>
int Tensor<T>::getDims(int dim)
{
    if (dim < rank_)
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
T* Tensor<T>::getData(int)
{
    return root_;
}
template<typename T>
Tensor<T> Tensor<T>::matMul(Tensor<T>& in_tensor)
{
    //in_tensor.root_
}


template<typename T>
Tensor<T> Tensor<T>::matMul1D(Tensor<T>& in_tensor)
{
    if ( (rank_ == 0 ) && (in_tensor.rank_ > 0) )
    {
        T* ptr = in_tensor.root_;
        int numOfElements = 1;
        for(int i =0 ; i< in_tensor.rank_ ; i++)
        {
            numOfElements *= in_tensor.dims_[i];
        }

        for(int i=0; i < numOfElements; i++)
        {
            *(ptr+i) = *(ptr+i) * (*root_);
        }
    }
    else if ( (in_tensor.rank_ == 0 ) && (rank_ > 0 )  )
    {
        T* ptr = root_;
        int numOfElements = 1;
        for(int i =0 ; i< rank_ ; i++)
        {
            numOfElements *= dims_[i];
        }

        for(int i=0; i < numOfElements; i++)
        {
            *(ptr+i) = *(ptr+i) * (*root_);
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
    Tensor<T> result(dims_[0], in_tensor.dims_[1]);
    double sum = 0;
    if ( (rank_ == 2) && (in_tensor.rank_ == 2))
    {
        for(int i = 0 ; i < dims_[0] ; i ++ )
        {
            for(int k =0 ; k < in_tensor.dims_[1]; k++)
            {
                sum =0 ;
                for(int j = 0 ; j < dims_[1] ; j ++)
                {
                    sum += root_[i][j] * in_tensor.root_[j][k];
                }
               result[i][k] = sum;
            }
        }
    }
    else
    {
        printf("Dimension Error. Use matMul1D or matMul3D\n");
    }
}

template<typename T>
Tensor<T> Tensor<T>::matMul3D(Tensor<T>& in_tensor)
{
    //in_tensor.root_
}

template<typename T>
Tensor<T> Tensor<T>::dotMul(Tensor<T>& )
{}
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