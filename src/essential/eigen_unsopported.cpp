#include"eigen_unsopported.h"

Eigen::MatrixXd mBaseOp(Eigen::MatrixXd* x, double (*fp)(double ) )
{
    uint8 row = x->rows();
    uint8 col = x->cols();
    
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


Eigen::MatrixXd mexp(Eigen::MatrixXd* x)
{
    return mBaseOp(x, exp);
}

Eigen::MatrixXd mlog(Eigen::MatrixXd* x)
{
    return mBaseOp(x, log);
}

Eigen::MatrixXd mpow(Eigen::MatrixXd* x)
{
    int a = pow(2, 2);
}