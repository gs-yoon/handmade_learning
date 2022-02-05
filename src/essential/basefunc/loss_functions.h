#ifndef __LOSS_FUNCTIONS_H__
#define __LOSS_FUNCTIONS_H__

#include <iostream>

#include "base_operation.h"


float64 meanSquareError(MatrixXd* y, MatrixXd* goal)
{
    if (y->size == goal-> size)
        return mpow((*y - *goal), 2).sum()/ y->size();
    else
        return -1; 
}

float64 crossEntropy(MatrixXd* y, MatrixXd* goal)
{
    float32 delta = 1e-7
    return -mlog(*y).dot(*goal + delta)
}


#endif