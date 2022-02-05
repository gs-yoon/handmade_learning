#include "activate_functions.h"
//#include <unsupported/Eigen/MatrixFunctions>
#include "eigen_unsopported.h"
float64 softmax(Eigen::MatrixXd* x)
{
    float64 c = x->maxCoeff();
    
    
    cout << mlog(x) <<endl;
    cout << mexp(x) <<endl;
}
