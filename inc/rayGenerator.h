/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RAY_GENERATOR_H
#define EQ_MIVT_RAY_GENERATOR_H

#include "typedef.h"
#include "cuda_runtime.h"

namespace eqMivt
{
void generateRays_CUDA(float * rays, float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, cudaStream_t stream);
}
#endif
