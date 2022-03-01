#include"base_operation.h"
/*
inline Eigen::MatrixXd mBaseOp2d(Eigen::MatrixXd* x, double (*fp)(double ) )
{
    uint32 row = x->rows();
    uint32 col = x->cols();
    
    Eigen::MatrixXd m(row,col);

    for (int c =0 ; c < col; c++)
    {
        for (int r =0 ; r < row; r++)
        {
            m(r,c) = fp(x->coeff(r,c));
        }
    }
    return m;
}

inline Eigen::MatrixXd mBaseOp2d(Eigen::MatrixXd* x, float32 p, double (*fp)(double, double ) )
{
    uint32 row = x->rows();
    uint32 col = x->cols();
    
    Eigen::MatrixXd m(row,col);

    for (int c =0 ; c < col; c++)
    {
        for (int r =0 ; r < row; r++)
        {
            m(r,c) = fp(x->coeff(r,c), p);
        }
    }
    return m;
}


Eigen::MatrixXd mexp(Eigen::MatrixXd* x)
{
    return mBaseOp2d(x, exp);
}

Eigen::MatrixXd mlog(Eigen::MatrixXd* x)
{
    return mBaseOp2d(x, log);
}

Eigen::MatrixXd mpow(Eigen::MatrixXd* x, float32 p)
{
    return mBaseOp2d(x,p,pow);
}
*/