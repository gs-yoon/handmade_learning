#ifndef __ININTIALIZER_H__
#define __ININTIALIZER_H__

#include "tensor.h"

double gaussianRandom(void) {
  double v1, v2, s;

  do {
    v1 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
    v2 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
    s = v1 * v1 + v2 * v2;
  } while (s >= 1 || s == 0);

  s = sqrt( (-2 * log(s)) / s );

  return v1 * s;
}

void gaussianRandomInit(Tensor& w)
{
    int *shape = w.getRawShape();

    for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
        for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
            for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                    for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                        w(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = gaussianRandom()/10.0;//(VALUETYPE)((double) (rand()%100 )/ 1000);
}

#endif