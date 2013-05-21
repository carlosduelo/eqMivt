/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_NODE_H
#define EQ_MIVT_NODE_H

#include "eqMivt.h"
#include "initData.h"
#include "octreeContainer.h"
#include "cubeCache.h"
#include "cubeCacheCPU.h"

#include <map>
#include <lunchbox/lock.h>

#include <eq/eq.h>

namespace eqMivt
{
	/**
	 * Representation of a node in the cluster
	 * 
	 * Manages node-specific data, namely requesting the mapping of the
	 * initialization data by the local Config instance.
	 */
	class Node : public eq::Node
	{
		public:
			Node( eq::Config* parent ) : eq::Node( parent ) {}

			bool					registerPipeResources(int device);
			OctreeContainer *		getOctreeContainer(int device);
			cubeCache *				getCubeCache(int device);
			int						getNewId();
			vmml::vector<3, int>    getVolumeDim();

		protected:
			virtual ~Node(){}

			virtual bool configInit( const eq::uint128_t& initID );

			virtual bool configExit();

		private:

			lunchbox::Lock								_lock;
			std::map<int , eqMivt::OctreeContainer *> 	_octrees;
			std::map<int , eqMivt::cubeCache *>			_caches;
			cubeCacheCPU								_cubeCacheCPU;
			bool										_initCubeCacheCPU;
			int											_idPipes;
	};
}

#endif // EQ_MIVT_NODE_H

