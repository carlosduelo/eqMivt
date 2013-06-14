/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_OCTREE_H_
#define _EQ_MIVT_OCTREE_H_

#include "typedef.h"

#include "eq/eq.h"

#include "cuda_runtime.h"

namespace eqMivt
{

class Octree
{
	private:
		float					_isosurface;
		int						_maxHeight;
		vmml::vector<3, int>	_realDim;
		int						_dimension;
		int						_nLevels;
		int						_maxLevel;
		uint32_t				_device;

		index_node_t *		_memoryOctree;
		index_node_t ** 	_octree;
		int	*				_sizes;
		int					_currentLevel;

	public:
		Octree();

		~Octree();

		void setGeneralValues(uint32_t device);
		
		bool setCurrentOctree(vmml::vector<3, int> realDim, int dimension, int nLevels, int maxLevel, int currentLevel, float isosurface, int maxHeight, index_node_t ** octree, int * sizes, int lastLevel);

		void increaseLevel() { _currentLevel = _currentLevel == _maxLevel ? _maxLevel : _currentLevel + 1; }

		void decreaseLevel() { _currentLevel = _currentLevel == 1 ? 1 : _currentLevel - 1; }

		int getnLevels() { return _nLevels; }

		int getMaxLevel() { return _maxLevel; }

		int	getOctreeLevel() { return _currentLevel; }

		float getIsosurface() { return _isosurface; }

		int getMaxHeight() { return _maxHeight; }

		vmml::vector<3, int> getRealDim() { return _realDim; }

		/* Dado un rayo devuelve true si el rayo impacta contra el volumen, el primer box del nivel dado contra el que impacta y la distancia entre el origen del rayo y la box */
		void getBoxIntersected(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, int numRays, int * indexVisibleCubesGPU, int * indexVisibleCubesCPU, cudaStream_t stream);

};

}

#endif
