/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_OCTREE_CONTAINER_CUDA_H
#define EQ_MIVT_OCTREE_CONTAINER_CUDA_H

#include <typedef.h>

#include <string>
#include <iostream>

namespace eqMivt
{
bool Create_OctreeContainer(int device, index_node_t ** octreeCPU, int * sizesCPU, int maxLevel, index_node_t *** octree, index_node_t ** memoryGPU, int ** sizes, std::string * result); 

bool Destroy_OctreeContainer(int device, index_node_t ** octree, index_node_t * memoryGPU, int * sizes); 
}

#endif /*EQ_MIVT_OCTREE_CONTAINER_CUDA_H*/
