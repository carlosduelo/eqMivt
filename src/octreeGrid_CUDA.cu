/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "octreeGrid_CUDA.h"

#include "cuda_help.h"
#include "mortonCodeUtil.h"

#include "cutil_math.h"

#include <iostream>
#include <fstream>

namespace eqMivt
{
/*
 **********************************************************************************************
 ****** GPU Octree functions ******************************************************************
 **********************************************************************************************
 */

__device__ inline bool _cuda_checkRangeGrid(index_node_t * elements, index_node_t index, int min, int max)
{
	return  index == elements[min] 	|| 
		index == elements[max]	||
		(elements[min] < index && elements[max] > index);
}

__device__ int _cuda_binary_search_closer_Grid(index_node_t * elements, index_node_t index, int min, int max)
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
		found 		=  _cuda_checkRangeGrid(elements, index, middle, middle+1);
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

			if (_cuda_checkRangeGrid(elements, index, middle, middle+1))
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

__device__  bool _cuda_searchSecuentialGrid(index_node_t * elements, index_node_t index, int min, int max)
{
	bool find = false;
	for(int i=min; i<max; i+=2)
		if (_cuda_checkRangeGrid(elements, index, i, i+1))
			find = true;

	return find;
}

inline __device__ float3 _cuda_BoxToCoordinates(int3 pos, float * xGrid, float * yGrid, float * zGrid, int3 realDim)
{
	float3 r;
	r.x = pos.x >= realDim.x ? xGrid[realDim.x-1] : xGrid[pos.x];
	r.y = pos.y >= realDim.y ? yGrid[realDim.y-1] : yGrid[pos.y];
	r.z = pos.z >= realDim.z ? zGrid[realDim.z-1] : zGrid[pos.z];

	return r;
}

__device__ bool _cuda_RayAABB(index_node_t index, float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels, float * xGrid, float * yGrid, float * zGrid, int3 realDim)
{
	int3 minBoxC;
	int3 maxBoxC;
	int level;
	minBoxC = getMinBoxIndex(index, &level, nLevels); 
	if (minBoxC.x >= realDim.x || minBoxC.y >= realDim.y || minBoxC.y >= realDim.y)
		return false;
	int dim = (1<<(nLevels-level));
	maxBoxC.x = dim + minBoxC.x;
	maxBoxC.y = dim + minBoxC.y;
	maxBoxC.z = dim + minBoxC.z;
	float3 minBox = _cuda_BoxToCoordinates(minBoxC, xGrid, yGrid, zGrid, realDim);
	float3 maxBox = _cuda_BoxToCoordinates(maxBoxC, xGrid, yGrid, zGrid, realDim);

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1.0f / dir.x;
	if (divx >= 0.0f)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1.0f / dir.y;
	if (divy >= 0.0f)
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
		return false;

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1.0f / dir.z;
	if (divz >= 0.0f)
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
		return false;

	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin<0.0f)
	 	*tnear=0.0f;
	else
		*tnear=tmin;
	*tfar=tmax;

	return *tnear < *tfar;
}

__device__ bool _cuda_RayAABB2(float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels, int3 minBoxC, int level, float * xGrid, float * yGrid, float * zGrid, int3 realDim)
{
	if (minBoxC.x >= realDim.x || minBoxC.y >= realDim.y || minBoxC.y >= realDim.y)
		return false;

	int3 maxBoxC;
	int dim = (1<<(nLevels-level));
	maxBoxC.x = dim + minBoxC.x;
	maxBoxC.y = dim + minBoxC.y;
	maxBoxC.z = dim + minBoxC.z;
	float3 minBox = _cuda_BoxToCoordinates(minBoxC, xGrid, yGrid, zGrid, realDim);
	float3 maxBox = _cuda_BoxToCoordinates(maxBoxC, xGrid, yGrid, zGrid, realDim);

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1.0f / dir.x;
	if (divx >= 0.0f)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1.0f / dir.y;
	if (divy >= 0.0f)
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
		return false;

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1.0f / dir.z;
	if (divz >= 0.0f)
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
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (fabsf(tmax -tmin) < EPS)
		return false;

	if (tmin<0.0f)
	 	*tnear=0.0f;
	else
		*tnear=tmin;

	*tfar=tmax;

	return *tnear < *tfar;

}

__device__ bool _cuda_searchNextChildrenValidAndHit(index_node_t * elements, int size, float * xGrid, float * yGrid, float * zGrid, int3 realDim, float3 origin, float3 ray, index_node_t father, float cTnear, float cTfar, int nLevels, int level, int3 minB, index_node_t * child, float * childTnear, float * childTfar)
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
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
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
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
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
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
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
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
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
		unsigned int closer1 = _cuda_binary_search_closer_Grid(elements, childrenID,   0, size-1);
		unsigned int closer8 = _cuda_binary_search_closer_Grid(elements, childrenID+7, closer1, size-1) + 1;

		if (closer8 >= size)
			closer8 = size-1;

		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
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
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
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
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
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
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, xGrid, yGrid, zGrid, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
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

__device__ int3 _cuda_updateCoordinatesGrid(int maxLevel, int cLevel, index_node_t cIndex, int nLevel, index_node_t nIndex, int3 minBox)
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

__global__ void cuda_getFirtsVoxelGrid(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, visibleCube_t * p_indexNode, int * indexCube,int numElements, float * xGrid, float * yGrid, float * zGrid, int3 realDim)
{
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
		i = indexCube[i];
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
			if (!_cuda_RayAABB(current, origin, ray,  &currentTnear, &currentTfar, nLevels, xGrid, yGrid, zGrid, realDim) || currentTfar < 0.0f)
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
				if (_cuda_searchNextChildrenValidAndHit(octree[currentLevel+1], sizes[currentLevel+1], xGrid, yGrid, zGrid, realDim, origin, ray, current, currentTnear, currentTfar, nLevels, currentLevel+1, minBox, &child, &childTnear, &childTfar))
				{
					minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel + 1, child, minBox);
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
					minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel - 1, current >> 3, minBox);
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

	void getBoxIntersectedOctreeGrid(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int numElements, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, int * indexVisibleCubesGPU, int * indexVisibleCubesCPU, float * xGrid, float * yGrid, float * zGrid, int3 realDim, cudaStream_t stream)
{

	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	cuda_getFirtsVoxelGrid<<<blocks,threads, 0,stream>>>(octree, sizes, nLevels, origin, LB, up, right, w, h, pvpW, pvpH, finalLevel, visibleGPU, indexVisibleCubesGPU, numElements, xGrid, yGrid,  zGrid, realDim);

}

/*
 ******************************************************************************************************
 ************ METHODS OCTREE CUDA CREATE **************************************************************
 ******************************************************************************************************
 */

__global__ void insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU)
{
	int offset = 0;
	for(int i=0;i<threadIdx.x; i++)
		offset+=sizes[i];

	octreeGPU[threadIdx.x] = &memoryGPU[offset];
}



bool Create_Octree(index_node_t ** octreeCPU, int * sizesCPU, int maxLevel, index_node_t *** octree, index_node_t ** memoryGPU, int ** sizes, int lastLevel, int3 realVolDim, int3 realDim, float ** xGrid, float ** yGrid, float ** zGrid, float * xGridCPU, float * yGridCPU, float * zGridCPU)
{
	// Creating Grid
	if (*xGrid == 0)
	{
		std::cout<< "Allocating xGrid "<<(4+realVolDim.x)*sizeof(float)/1024.0f/1024.0f<< " MB: ";
		if (cudaSuccess != (cudaMalloc(xGrid, (4+realVolDim.x)*sizeof(float))))
		{
			std::cout<< "Octree: error allocating grid x"<<std::endl;
			return false;
		}
		else
			std::cout<<"OK"<<std::endl;

		std::cout<< "Octree: coping x grid ";
		if (cudaSuccess != (cudaMemcpy((void*)*xGrid, (void*)xGridCPU, (4+realVolDim.x)*sizeof(float), cudaMemcpyHostToDevice)))
		{
			std::cout<< "Fail"<<std::endl;
			return false;
		}
		else
			std::cout<< "OK"<<std::endl;
	}
	if (*yGrid == 0)
	{
		std::cout<< "Allocating yGrid "<<(4+realVolDim.y)*sizeof(float)/1024.0f/1024.0f<< " MB: ";
		if (cudaSuccess != (cudaMalloc(yGrid, (4+realVolDim.y)*sizeof(float))))
		{
			std::cout<< "Octree: error allocating grid x"<<std::endl;
			return false;
		}
		else
			std::cout<<"OK"<<std::endl;

		std::cout<< "Octree: coping y grid ";
		if (cudaSuccess != (cudaMemcpy((void*)*yGrid, (void*)yGridCPU, (4+realVolDim.y)*sizeof(float), cudaMemcpyHostToDevice)))
		{
			std::cout<< "Fail"<<std::endl;
			return false;
		}
		else
			std::cout<< "OK"<<std::endl;
	}
	if (*zGrid == 0)
	{
		std::cout<< "Allocating xGrid "<<(4+realVolDim.z)*sizeof(float)/1024.0f/1024.0f<< " MB: ";
		if (cudaSuccess != (cudaMalloc(zGrid, (4+realVolDim.z)*sizeof(float))))
		{
			std::cout<< "Octree: error allocating grid x"<<std::endl;
			return false;
		}
		else
			std::cout<<"OK"<<std::endl;

		std::cout<< "Octree: coping z grid ";
		if (cudaSuccess != (cudaMemcpy((void*)*zGrid, (void*)zGridCPU, (4+realVolDim.z)*sizeof(float), cudaMemcpyHostToDevice)))
		{
			std::cout<< "Fail"<<std::endl;
			return false;
		}
		else
			std::cout<< "OK"<<std::endl;
	}


	// Creating sizes
	if ((*sizes) == 0)
	{
		std::cout<< "Allocating memory octree CUDA sizes "<<(lastLevel+1)*sizeof(int)/1024.0f/1024.0f<< " MB: ";
		if (cudaSuccess != (cudaMalloc(sizes,   (lastLevel+1)*sizeof(int))))
		{
			std::cout<< "Octree: error allocating octree in the gpu"<<std::endl;
			return false;
		}
		else
			std::cout<<"OK"<<std::endl;
	}

	std::cout<< "Octree: coping to device the sizes ";
	if (cudaSuccess != (cudaMemcpy((void*)*sizes, (void*)sizesCPU, (maxLevel+1)*sizeof(int), cudaMemcpyHostToDevice)))
	{
		std::cout<< "Fail"<<std::endl;
		return false;
	}
	else
		std::cout<< "OK"<<std::endl;

	// Create octree
	if ((*octree) == 0)
	{
		std::cout<< "Allocating memory octree CUDA octree "<<(lastLevel+1)*sizeof(index_node_t*)/1024.0f/1024.0f<< " MB: ";
		if (cudaSuccess != (cudaMalloc(octree, (lastLevel+1)*sizeof(index_node_t*))))
		{
			std::cout<< "Octree: error allocating octree in the gpu"<<std::endl;
			return false;
		}
		else
			std::cout<<"OK"<<std::endl;
	}

	int total = 0;
	for(int i=0; i<=maxLevel; i++)
		total+=sizesCPU[i];

	if ((*memoryGPU) != 0)
	{
		if (cudaSuccess != cudaFree(*memoryGPU))
		{
			std::cout<<"Error creating octree"<<std::endl;
			return false;
		}
	}

	std::cout<< "Allocating memory octree CUDA memory "<< total*sizeof(index_node_t)/1024.0f/1024.0f<< " MB: ";
	if (cudaSuccess != (cudaMalloc(memoryGPU, total*sizeof(index_node_t))))
	{
		std::cout<<" Fail"<<std::endl;
		std::cout<< "Octree: error allocating octree in the gpu"<<std::endl;
		return false;
	}
	std::cout<<" OK"<<std::endl;

	/* Copying octree */
	int offset = 0;
	for(int i=0; i<=maxLevel; i++)
	{
		std::cout<< "Coping to device level "<< i<<": ";
		if (cudaSuccess != (cudaMemcpy((void*)((*memoryGPU)+offset), (void*)octreeCPU[i], sizesCPU[i]*sizeof(index_node_t), cudaMemcpyHostToDevice)))
		{
			std::cout<<"Fail"<<std::endl;;
			return false;
		}
		else
			std::cout<< "OK"<<std::endl;

		offset+=sizesCPU[i];
	}

	dim3 blocks(1);
	dim3 threads(maxLevel+1);

	insertOctreePointers<<<blocks,threads>>>(*octree, *sizes,*memoryGPU);
	std::cout<< "Octree: sorting pointers ";
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		std::cout<< "Fail"<<std::endl;
		return false;
	}
	else
		std::cout<< "OK"<<std::endl;

	std::cout<< "End copying octree to GPU"<<std::endl;

	return true;
}


bool Destroy_Octree(int device, index_node_t ** octree, index_node_t * memoryGPU, int * sizes, float * xGrid, float * yGrid, float * zGrid)
{

	if (device < 0)
		return false;

	int d = 40;
	cudaGetDevice(&d);

	if (xGrid != 0)
	{
		if (d != device)
			cudaSetDevice(device);
		cudaFree(xGrid);
	}
	if (xGrid != 0)
	{
		if (d != device)
			cudaSetDevice(device);
		cudaFree(xGrid);
	}
	if (yGrid != 0)
	{
		if (d != device)
			cudaSetDevice(device);
		cudaFree(yGrid);
	}
	if (zGrid != 0)
	{
		if (d != device)
			cudaSetDevice(device);
		cudaFree(zGrid);
	}

	if (memoryGPU != 0)
	{
		if (d != device)
			cudaSetDevice(device);
		cudaFree(memoryGPU);
	}
	if (octree != 0)
	{
		if (d != device)
			cudaSetDevice(device);
		cudaFree(octree);
	}
	if (sizes != 0)
	{
		if (d != device)
			cudaSetDevice(device);
		cudaFree(sizes);
	}

	if (d != device)
		cudaSetDevice(d);

	return true;
}
}