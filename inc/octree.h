/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_OCTREE_H_
#define _EQ_MIVT_OCTREE_H_

#include "typedef.h"
#include "octreeContainer.h"

#include "cuda_runtime.h"

namespace eqMivt
{

class Octree
{
	private:
		int _nLevels;
		int _currentLevel;
		int _maxLevel;

		index_node_t ** _octree;
		int	*			_sizes;

		int		_maxRays;

	public:
		Octree();

		~Octree();
		
		void setOctree(OctreeContainer * oc, int maxRays);

		void	increaseLevel()
		{
			_currentLevel = _currentLevel == _maxLevel ? _maxLevel : _currentLevel + 1;
		}

		void	decreaseLevel()
		{
			_currentLevel = _currentLevel == 1 ? 1 : _currentLevel - 1;
		}

		int 	getnLevels()
		{
			return _nLevels;
		}

		int	getOctreeLevel()
		{
			return _currentLevel;
		}

		void resizeViewport(int width, int height);

		/* Dado un rayo devuelve true si el rayo impacta contra el volumen, el primer box del nivel dado contra el que impacta y la distancia entre el origen del rayo y la box */
		void getBoxIntersected(float3 origin, float * rays, int pvpW, int pvpH, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream);

};

}

#endif
