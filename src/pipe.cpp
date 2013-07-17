/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "pipe.h"
#include "cubeCache.h"

#include "config.h"
#include "node.h"
#include <eq/eq.h>

namespace eqMivt
{
bool Pipe::configInit( const eq::uint128_t& initID )
{
    if( !eq::Pipe::configInit( initID ))
        return false;

    Config*         config      = static_cast<Config*>( getConfig( ));
    const InitData& initData    = config->getInitData();
    const eq::uint128_t&  frameDataID = initData.getFrameDataID();

	int ds = -1;
	if (cudaSuccess != cudaGetDevice(&ds))
	{
		LBERROR<<"Pipe: Error checking cuda device capable"<<std::endl;
		return false;
	}

	if (getDevice() < 32 && ds != getDevice())
		if (cudaSuccess != cudaSetDevice(getDevice()))
		{
			LBERROR<<"Pipe: Error setting cuda device capable"<<std::endl;
			return false;
		}
		
	_render.setName(getName());
	
	_lastState = _render.setColors(0);

    return _lastState && config->mapObject( &_frameData, frameDataID );
}

bool Pipe::configExit()
{
    eq::Config* config = getConfig();
    config->unmapObject( &_frameData );

    return eq::Pipe::configExit();
}

void Pipe::frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber)
{
    eq::Pipe::frameStart( frameID, frameNumber );
    _frameData.sync( frameID );
}

Render * Pipe::getRender()
{
    Node*       node = static_cast<Node*>( getNode( ));

	if (!_lastState)
		return 0;

	if (node->checkStatus())
	{
		_render.setOctree(node->getOctree(getDevice()));

		if (!node->getCacheHandler(getDevice(), _render.getCacheHandler()))
		{
			_lastState = false;
			LBERROR<<"Pipe: error getting cache handler"<<std::endl;
			return 0;
		}

		// Check status node
		const FrameData& frameData = getFrameData();
		if (node->updateStatus(getDevice(), _render.getCacheHandler(), frameData.getCurrentOctree(), frameData.useGrid(), frameData.isRenderCubes()) &&
			_render.checkCudaResources())
			return &_render;
		else
		{
			_lastState = false;
			LBERROR<<"Pipe: Error getting renderer, updating node status"<<std::endl;
			return 0;
		}
	}
	else
	{
		_lastState = 0;
		LBERROR<<"Pipe: Error getting renderer, node status = unstable"<<std::endl;
		return 0;
	}

}

}
