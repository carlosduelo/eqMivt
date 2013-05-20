/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "octree_CUDA.h"

#include "cuda_help.h"
#include "mortonCodeUtil.h"

#include "cutil_math.h"

#include <iostream>
#include <fstream>

#define STACK_DIM 32

namespace eqMivt
{
/*
 **********************************************************************************************
 ****** GPU Octree functions ******************************************************************
 **********************************************************************************************
 */

__device__ inline bool _cuda_checkRange(index_node_t * elements, index_node_t index, int min, int max)
{
	return  index == elements[min] 	|| 
		index == elements[max]	||
		(elements[min] < index && elements[max] > index);
}

__device__ int _cuda_binary_search_closer(index_node_t * elements, index_node_t index, int min, int max)
{
#if 1
	bool end = false;
	bool found = false;
	int middle = 0;

	while(!end && !found)
	{
		int diff 	= max-min;
		middle	= min + (diff / 2);
		if (middle % 2 == 1) middle--;

		end 		= diff <= 1;
		found 		=  _cuda_checkRange(elements, index, middle, middle+1);
		if (index < elements[middle])
			max = middle-1;
		else //(index > elements[middle+1])
			min = middle + 2;
	}
	return middle;
#endif
#if 0
	while(1)
	{
		int diff = max-min;
		unsigned int middle = min + (diff / 2);
		if (diff <= 1)
		{
			if (middle % 2 == 1) middle--;
			return middle;
		}
		else
		{
			if (middle % 2 == 1) middle--;

			if (_cuda_checkRange(elements, index, middle, middle+1))
				return middle;
			else if (index < elements[middle])
			{
				max = middle-1;
			}
			else if (index > elements[middle+1])
			{
				min = middle + 2;
			}
			#if 0
			// XXX en cuda me arriesgo... malo...
			else
				std::cout<<"Errro"<<std::endl;
			#endif
		}
	}
#endif
}

__device__  bool _cuda_searchSecuential(index_node_t * elements, index_node_t index, int min, int max)
{
	bool find = false;
	for(int i=min; i<max; i+=2)
		if (_cuda_checkRange(elements, index, i, i+1))
			find = true;

	return find;
}

__device__ bool _cuda_RayAABB(index_node_t index, float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels)
{
	int3 minBox;
	int3 maxBox;
	int level;
	minBox = getMinBoxIndex(index, &level, nLevels); 
	int dim = (1<<(nLevels-level));
	maxBox.x = dim + minBox.x;
	maxBox.y = dim + minBox.y;
	maxBox.z = dim + minBox.z;
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

	return *tnear == *tfar ? false : hit;
}

__device__ bool _cuda_RayAABB2(float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels, int3 minBox, int level)
{
	int3 maxBox;
	int dim = (1<<(nLevels-level));
	maxBox.x = dim + minBox.x;
	maxBox.y = dim + minBox.y;
	maxBox.z = dim + minBox.z;
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

	return *tnear == *tfar ? false : hit;

}

__device__ bool _cuda_searchNextChildrenValidAndHit(index_node_t * elements, int size, float3 origin, float3 ray, index_node_t father, float cTnear, float cTfar, int nLevels, int level, int3 minB, index_node_t * child, float * childTnear, float * childTfar)
{
	index_node_t childrenID = father << 3;
	int dim = (1<<(nLevels-level));
	int3 minBox = make_int3(minB.x, minB.y, minB.z);

	float closer = 0x7ff0000000000000;	//infinity
	bool find = false;
	float childTnearT = 0xfff0000000000000; // -infinity
	float childTfarT  = 0xfff0000000000000; // -infinity

	if (size==2)
	{
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.x+=dim;
		minBox.y-=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
	}
	else
	{
		unsigned int closer1 = _cuda_binary_search_closer(elements, childrenID,   0, size-1);
		unsigned int closer8 = _cuda_binary_search_closer(elements, childrenID+7, closer1, size-1) + 1;

		if (closer8 >= size)
			closer8 = size-1;

		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.x+=dim;
		minBox.y-=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
	}

	return find;
}

__device__ int3 _cuda_updateCoordinates(int maxLevel, int cLevel, index_node_t cIndex, int nLevel, index_node_t nIndex, int3 minBox)
{
	if ( 0 == nIndex)
	{
		return make_int3(0,0,0);
	}
	else if (cLevel < nLevel)
	{
		index_node_t mask = (index_node_t) 1;
		minBox.z +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.y +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.x +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		return minBox;

	}
	else if (cLevel > nLevel)
	{
		return	getMinBoxIndex2(nIndex, nLevel, maxLevel);
	}
	else
	{
		index_node_t mask = (index_node_t)1;
		minBox.z +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.y +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.x +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.z -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		minBox.y -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		minBox.x -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		return minBox;
	}
}

__global__ void cuda_getFirtsVoxel(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, visibleCube_t * p_indexNode, int numElements)
{
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
    	int is = i % pvpW;
		int js = i / pvpW;

		float3 ray = LB - origin;
		ray += (js*h)*up + (is*w)*right;
		ray = normalize(ray);

		visibleCube_t * indexNode	= &p_indexNode[i];

		if (indexNode->state ==  NOCUBE)
		{
			float			currentTnear	= 0.0f;
			float			currentTfar		= 0.0f;
			index_node_t 	current			= indexNode->id == 0 ? 1 : indexNode->id;
			int				currentLevel	= 0;
			
			// Update tnear and tfar
			if (!_cuda_RayAABB(current, origin, ray,  &currentTnear, &currentTfar, nLevels) || currentTnear < 0.0f)
			{
				// NO CUBE FOUND
				indexNode->id 	= 0;
				return;
			}

			if (current != 1)
			{
				current  >>= 3;
				currentLevel = finalLevel - 1;
				currentTnear = currentTfar;
				//printf("--> %d %lld %d %f %f\n", i, current, currentLevel, currentTnear, currentTfar);
			}

			int3		minBox 		= getMinBoxIndex2(current, currentLevel, nLevels);

			while(1)
			{
				if (currentLevel == finalLevel)
				{
					indexNode->id = current;
					indexNode->state = CUBE;
					return;
				}

				// Get fitst child >= currentTnear away
				index_node_t	child;
				float			childTnear;
				float			childTfar;
				if (_cuda_searchNextChildrenValidAndHit(octree[currentLevel+1], sizes[currentLevel+1], origin, ray, current, currentTnear, currentTfar, nLevels, currentLevel+1, minBox, &child, &childTnear, &childTfar))
				{
					minBox = _cuda_updateCoordinates(nLevels, currentLevel, current, currentLevel + 1, child, minBox);
					current = child;
					currentLevel++;
					currentTnear = childTnear;
					currentTfar = childTfar;
				//if (currentTnear == currentTfar)
				//	printf("--> %d %lld %d %f %f\n", i, current, currentLevel, currentTnear, currentTfar);
				}
				else if (current == 1) 
				{
					indexNode->id 	= 0;
					return;
				}
				else
				{
					minBox = _cuda_updateCoordinates(nLevels, currentLevel, current, currentLevel - 1, current >> 3, minBox);
					current >>= 3;
					currentLevel--;
					currentTnear = currentTfar;
				}

			}
		}
	}
	return;
}

/*
 ******************************************************************************************************
 ************ METHODS OCTREEMCUDA *********************************************************************
 ******************************************************************************************************
 */

void getBoxIntersectedOctree(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int numElements, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream)
{
	//std::cerr<<"Getting firts box intersected"<<std::endl;

	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	//std::cerr<<"Set HEAP size: "<< cudaGetErrorString(cudaThreadSetLimit(cudaLimitMallocHeapSize , numElements*1216)) << std::endl;

	cuda_getFirtsVoxel<<<blocks,threads, 0,stream>>>(octree, sizes, nLevels, origin, LB, up, right, w, h, pvpW, pvpH, finalLevel, visibleGPU, numElements);

	//std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;

	//std::cerr<<"Coping to host visibleCubes: "<< cudaGetErrorString(
	cudaMemcpyAsync((void*)visibleCPU, (const void*)visibleGPU, numElements*sizeof(visibleCube_t), cudaMemcpyDeviceToHost, stream);//) << std::endl;

	//std::cerr<<"End Getting firts box intersected"<<std::endl;
}

}
