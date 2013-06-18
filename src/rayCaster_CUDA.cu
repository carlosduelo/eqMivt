#include "rayCaster_CUDA.h"

#include "mortonCodeUtil.h"
#include "cuda_help.h"

#include <cutil_math.h>

#include <iostream>
#include <fstream>

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

namespace eqMivt
{
inline __device__ float3 _cuda_BoxToCoordinates(int3 pos, double * xGrid, double * yGrid, double * zGrid, int3 realDim)
{
	float3 r;
	r.x = pos.x >= realDim.x ? xGrid[realDim.x-1] : xGrid[pos.x];
	r.y = pos.y >= realDim.y ? yGrid[realDim.y-1] : yGrid[pos.y];
	r.z = pos.z >= realDim.z ? zGrid[realDim.z-1] : zGrid[pos.z];

	return r;
}

inline __device__ bool _cuda_RayAABB(float3 origin, float3 dir,  float * tnear, float * tfar, float3 minBox, float3 maxBox)
{
	bool hit = true;

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1 / dir.x;
	if (divx >= 0)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1 / dir.y;
	if (divy >= 0)
	{
		tymin = (minBox.y - origin.y)*divy;
		tymax = (maxBox.y - origin.y)*divy;
	}
	else
	{
		tymin = (maxBox.y - origin.y)*divy;
		tymax = (minBox.y - origin.y)*divy;
	}

	if ( (tmin > tymax) || (tymin > tmax) )
	{
		hit = false;
	}

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1 / dir.z;
	if (divz >= 0)
	{
		tzmin = (minBox.z - origin.z)*divz;
		tzmax = (maxBox.z - origin.z)*divz;
	}
	else
	{
		tzmin = (maxBox.z - origin.z)*divz;
		tzmax = (minBox.z - origin.z)*divz;
	}

	if ( (tmin > tzmax) || (tzmin > tmax) )
	{
		hit = false;
	}
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin<0.0)
	 	*tnear=0.0;
	else
		*tnear=tmin;
	*tfar=tmax;

	return *tnear >= *tfar ? false : hit;

}

inline __device__ bool _cuda_RayAABB(float3 origin, float3 dir,  float * tnear, float * tfar, int3 minBox, int3 maxBox)
{
	bool hit = true;

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1 / dir.x;
	if (divx >= 0)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1 / dir.y;
	if (divy >= 0)
	{
		tymin = (minBox.y - origin.y)*divy;
		tymax = (maxBox.y - origin.y)*divy;
	}
	else
	{
		tymin = (maxBox.y - origin.y)*divy;
		tymax = (minBox.y - origin.y)*divy;
	}

	if ( (tmin > tymax) || (tymin > tmax) )
	{
		hit = false;
	}

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1 / dir.z;
	if (divz >= 0)
	{
		tzmin = (minBox.z - origin.z)*divz;
		tzmax = (maxBox.z - origin.z)*divz;
	}
	else
	{
		tzmin = (maxBox.z - origin.z)*divz;
		tzmax = (minBox.z - origin.z)*divz;
	}

	if ( (tmin > tzmax) || (tzmin > tmax) )
	{
		hit = false;
	}
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin<0.0)
	 	*tnear=0.0;
	else
		*tnear=tmin;
	*tfar=tmax;

	return *tnear >= *tfar ? false : hit;

}

inline __device__ float getElement(int x, int y, int z, float * data, int3 dim)
{
	return data[posToIndex(x,y,z,dim.x)]; 
}

__device__ float getElementInterpolate(float3 pos, float * data, int3 minBox, int3 dim)
{
	float3 posR;
	float3 pi = make_float3(modff(pos.x,&posR.x), modff(pos.y,&posR.y), modff(pos.z,&posR.z));

	int x0 = posR.x - minBox.x;
	int y0 = posR.y - minBox.y;
	int z0 = posR.z - minBox.z;
	int x1 = x0 + 1;
	int y1 = y0 + 1;
	int z1 = z0 + 1;

	float c00 = getElement(x0,y0,z0,data,dim) * (1.0f - pi.x) + getElement(x1,y0,z0,data,dim) * pi.x;
	float c01 = getElement(x0,y0,z1,data,dim) * (1.0f - pi.x) + getElement(x1,y0,z1,data,dim) * pi.x;
	float c10 = getElement(x0,y1,z0,data,dim) * (1.0f - pi.x) + getElement(x1,y1,z0,data,dim) * pi.x;
	float c11 = getElement(x0,y1,z1,data,dim) * (1.0f - pi.x) + getElement(x1,y1,z1,data,dim) * pi.x;

	float c0 = c00 * (1.0f - pi.y) + c10 * pi.y;
	float c1 = c01 * (1.0f - pi.y) + c11 * pi.y;

	return c0 * (1.0f - pi.z) + c1 * pi.z;
#if 0

	float p000 = getElement(x0,y0,z1,data,dim);
	float p001 = getElement(x0,y1,z1,data,dim);
	float p010 = getElement(x0,y0,z0,data,dim);
	float p011 = getElement(x0,y1,z0,data,dim);
	float p100 = getElement(x1,y0,z1,data,dim);
	float p101 = getElement(x1,y1,z1,data,dim);
	float p110 = getElement(x1,y0,z0,data,dim);
	float p111 = getElement(x1,y1,z0,data,dim);

//	float3 pi = make_float3(modff(posR.x), modff(posR.y-(float)y0, posR.z-(float)z0);

	return  p000 * (1.0-pi.x) * (1.0-pi.y) * (1.0-pi.z) +       \
		p001 * (1.0-pi.x) * (1.0-pi.y) * pi.z       +       \
		p010 * (1.0-pi.x) * pi.y       * (1.0-pi.z) +       \
		p011 * (1.0-pi.x) * pi.y       * pi.z       +       \
		p100 * pi.x       * (1.0-pi.y) * (1.0-pi.z) +       \
		p101 * pi.x       * (1.0-pi.y) * pi.z       +       \
		p110 * pi.x       * pi.y       * (1.0-pi.z) +       \
		p111 * pi.x       * pi.y       * pi.z;
#endif
}

inline __device__ float3 getNormal(float3 pos, float * data, int3 minBox, int3 maxBox)
{
	return normalize(make_float3(	
				(getElementInterpolate(make_float3(pos.x-1.0f,pos.y,pos.z),data,minBox,maxBox) - getElementInterpolate(make_float3(pos.x+1.0f,pos.y,pos.z),data,minBox,maxBox))        /2.0f,
				(getElementInterpolate(make_float3(pos.x,pos.y-1.0f,pos.z),data,minBox,maxBox) - getElementInterpolate(make_float3(pos.x,pos.y+1.0f,pos.z),data,minBox,maxBox))        /2.0f,
			        (getElementInterpolate(make_float3(pos.x,pos.y,pos.z-1.0f),data,minBox,maxBox) - getElementInterpolate(make_float3(pos.x,pos.y,pos.z+1.0f),data,minBox,maxBox))        /2.0f));
}			

__global__ void cuda_rayCaster(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, float iso, visibleCube_t * cube, int * indexCube, int3 dimCube, int3 cubeInc, int levelO, int levelC, int nLevel, float maxHeight, double * xGrid, double * yGrid, double * zGrid, int3 realDim, float * screen)
{
	unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numRays)
	{
		tid = indexCube[tid];

		if (cube[tid].state == NOCUBE)
		{
			screen[tid*3] = 1.0f; 
			screen[tid*3+1] = 1.0f; 
			screen[tid*3+2] = 1.0f; 
			cube[tid].state = PAINTED;
			return;
		}
		else if (cube[tid].state == CACHED)
		{
			float tnear;
			float tfar;
			// To do test intersection real cube position
			int3 minBox = getMinBoxIndex2(cube[tid].id, levelO, nLevel);
			int dim = powf(2,nLevel-levelO);
			int3 maxBox = minBox + make_int3(dim,dim,dim);

			int i = tid % pvpW;
			int j = tid / pvpW;

			float3 ray = LB - origin;
			ray += (j*h)*up + (i*w)*right;
			ray = normalize(ray);

			if  (_cuda_RayAABB(origin, ray,  &tnear, &tfar, minBox, maxBox))
			{
				bool hit = false;
				float3 Xnear;
				float3 Xfar;
				float3 Xnew;

				// To ray caster is needed bigger cube, so add cube inc
				minBox = getMinBoxIndex2(cube[tid].cubeID, levelC, nLevel) - cubeInc;
				maxBox = dimCube + 2*cubeInc;
				Xnear = origin + tnear * ray;
				Xfar  = Xnear;
				Xnew  = Xnear;
				bool 				primera 	= true;
				float 				ant		= 0.0;
				float				sig		= 0.0;
				int steps = 0;
				float3 vStep = 0.5* ray;
				int  maxStep = ceil((tfar-tnear)/0.5);

	/* CASOS A ESTUDIAR
	tnear==tfar MISS
	tfar<tnear MISS
	tfar-tfar< step STUDY BETWEEN POINTS
	*/

				while(steps <= maxStep)
				{
					if (primera)
					{
						primera = false;
						ant = getElementInterpolate(Xnear, cube[tid].data, minBox, maxBox);
						Xfar = Xnear;
					}
					else
					{
						sig = getElementInterpolate(Xnear, cube[tid].data, minBox, maxBox);
						if (( ((iso-ant)<0.0) && ((iso-sig)<0.0)) || ( ((iso-ant)>0.0) && ((iso-sig)>0.0)))
						{
							ant = sig;
							Xfar=Xnear;
						}
						else
						{
							/*
							Si el valor en los extremos es V_s y V_f y el valor que buscas (el de la isosuperficie) es V, S es el punto inicial y F es el punto final.
							a = (V - V_s) / (V_f - V_s)
							I = S * (1 - a) + V * a  (creo que esta fórmula te la puse al revés en el caso del color, revísala) 
							*/
							
							#if 0
							// Intersection Refinament using an iterative bisection procedure
							float valueE = 0.0;
							for(int k = 0; k<5;k++) // XXX Cuanto más grande mejor debería ser el renderizado
							{
								Xnew = (Xfar - Xnear)*((iso-sig)/(ant-sig))+Xnear;
								valueE = getElementInterpolate(Xnew, cube[tid].data, minBox, maxBox);
								if (valueE>iso)
									Xnear=Xnew;
								else
									Xfar=Xnew;
							}
							#endif
							float a = (iso-ant)/(sig-ant);
							Xnew = Xfar*(1.0f-a)+ Xnear*a;
							hit = true;
							steps = maxStep;
						}
						
					}

					Xnear += vStep;
					steps++;
				}
				if (hit)
				{
					float3 n = getNormal(Xnew, cube[tid].data, minBox,  maxBox);
					float3 l = Xnew - origin;// ligth; light on the camera
					l = normalize(l);	
					float dif = fabs(n.x*l.x + n.y*l.y + n.z*l.z);

					float a = Xnew.y/maxHeight;
					screen[tid*3]   =(1-a)*dif;// + 1.0f*spec;
					screen[tid*3+1] =(a)*dif;// + 1.0f*spec;
					screen[tid*3+2] =0.0f*dif;// + 1.0f*spec;
					cube[tid].state= PAINTED;
				}
				else
				{
					cube[tid].state = NOCUBE;
				}
			}
			#if _DEBUG_
			else
			{
				printf("Error, octree is not working %lld %d \n",cube[tid].id, getIndexLevel(cube[tid].id));
			}
			#endif
		}
	}
}

__global__ void cuda_rayCaster_Cubes(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, float iso, visibleCube_t * cube, int * indexCube, int3 dimCube, int3 cubeInc, int levelO, int levelC, int nLevel, float maxHeight, double * xGrid, double * yGrid, double * zGrid, int3 realDim, float * screen)
{
	unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numRays)
	{
		tid = indexCube[tid];

		if (cube[tid].state == NOCUBE)
		{
			screen[tid*3] = 1.0f; 
			screen[tid*3+1] = 1.0f; 
			screen[tid*3+2] = 1.0f; 
			cube[tid].state = PAINTED;
			return;
		}
		else if (cube[tid].state == CUBE)
		{
			int i = tid % pvpW;
			int j = tid / pvpW;

			float3 ray = LB - origin;
			ray += (j*h)*up + (i*w)*right;
			ray = normalize(ray);

			int3 minBoxC = getMinBoxIndex2(cube[tid].id, levelO, nLevel);
			int dim = powf(2,nLevel-levelO);
			int3 maxBoxC = minBoxC + make_int3(dim,dim,dim);

			float3 minBox = _cuda_BoxToCoordinates(minBoxC, xGrid, yGrid, zGrid, realDim);
			float3 maxBox = _cuda_BoxToCoordinates(maxBoxC, xGrid, yGrid, zGrid, realDim);

			float tnear = 0.0f;
			float tfar = 0.0f;
			_cuda_RayAABB(origin, ray,  &tnear, &tfar, minBox, maxBox);
			float3 hit = origin + tnear *ray;

			float3 n = make_float3(0.0f,0.0f,0.0f);
			float aux = 0.0f;

			if (fabs(maxBox.x - origin.x) < fabs(minBox.x - origin.x))
			{
				aux = minBox.x;
				minBox.x = maxBox.x; 
				maxBox.x = aux;
			}
			if (fabs(maxBox.y - origin.y) < fabs(minBox.y - origin.y))
			{
				aux = minBox.y;
				minBox.y = maxBox.y; 
				maxBox.y = aux;
			}
			if (fabs(maxBox.z - origin.z) < fabs(minBox.z - origin.z))
			{
				aux = minBox.z;
				minBox.z = maxBox.z; 
				maxBox.z = aux;
			}

			if(fabs(hit.x - minBox.x) < 0.00001f) 
				n.x = -1.0f;
			else if(fabs(hit.x - maxBox.x) < 0.00001f) 
				n.x = 1.0f;
			else if(fabs(hit.y - minBox.y) < 0.00001f) 
				n.y = -1.0f;
			else if(fabs(hit.y - maxBox.y) < 0.00001f) 
				n.y = 1.0f;
			else if(fabs(hit.z - minBox.z) < 0.00001f) 
				n.z = -1.0f;
			else if(fabs(hit.z - maxBox.z) < 0.00001f) 
				n.z = 1.0f;


			float3 l = hit - origin;// ligth; light on the camera
			l = normalize(l);	
			float dif = fabs(n.x*l.x + n.y*l.y + n.z*l.z);

			float a = hit.y/maxHeight;
			screen[tid*3]   =(1-a)*dif;
			screen[tid*3+1] =(a)*dif;
			screen[tid*3+2] =0.0f;
			cube[tid].state= PAINTED;
		}
	}
}

	void rayCaster_CUDA(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, float iso, visibleCube_t * cube, int * indexCube, int3 cubeDim, int3 cubeInc, float maxHeight, float * pixelBuffer, double * xGrid, double * yGrid, double * zGrid, int3 realDim, cudaStream_t stream)
{
	dim3 threads = getThreads(numRays);
	dim3 blocks = getBlocks(numRays);
//	std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;

	cuda_rayCaster<<<blocks, threads, 0, stream>>>(origin, LB, up, right, w, h, pvpW, pvpH, numRays, iso, cube, indexCube, cubeDim, cubeInc, levelO, levelC, nLevel, maxHeight, xGrid, yGrid, zGrid, realDim, pixelBuffer);
//	std::cerr<<"Synchronizing rayCaster: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
	return;
}
	void rayCaster_Cubes_CUDA(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, float iso, visibleCube_t * cube, int * indexCube, int3 cubeDim, int3 cubeInc, float maxHeight, float * pixelBuffer, double * xGrid, double * yGrid, double * zGrid, int3 realDim, cudaStream_t stream)
{
	dim3 threads = getThreads(numRays);
	dim3 blocks = getBlocks(numRays);

	cuda_rayCaster_Cubes<<<blocks, threads, 0, stream>>>(origin, LB, up, right, w, h, pvpW, pvpH, numRays, iso, cube, indexCube, cubeDim, cubeInc, levelO, levelC, nLevel, maxHeight, xGrid, yGrid, zGrid, realDim, pixelBuffer);
	return;
}

}
