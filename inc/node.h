/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_NODE_H
#define EQ_MIVT_NODE_H

#include "eqMivt.h"
#include "initData.h"
#include "octreeManager.h"
#include "cacheManager.h"

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

			bool	checkStatus() { return _status; }
			bool updateStatus(uint32_t device, CacheHandler * cacheHandler, int currentOctree, bool grid, bool renderCubes);
			Octree *	getOctree(uint32_t device);
			bool		getCacheHandler(uint32_t device, CacheHandler * cacheHandler);
			vmml::vector<3, float>    getStartCoord();
			vmml::vector<3, float>    getFinishCoord();
			vmml::vector<3, float>    getVolumeCoord();

		protected:
			virtual ~Node(){}

			virtual bool configInit( const eq::uint128_t& initID );

			virtual bool configExit();

		private:

			bool			_status;
			OctreeManager	_octreeManager;
			CacheManager	_cacheManager;
	};
}

#endif // EQ_MIVT_NODE_H

