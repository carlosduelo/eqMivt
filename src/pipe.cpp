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
		std::cerr<<"Error checking cuda device capable"<<std::endl;
		return false;
	}

	if (getDevice() < 32 && ds != getDevice())
		if (cudaSuccess != cudaSetDevice(getDevice()))
		{
			std::cerr<<"Error setting cuda device capable"<<std::endl;
			return false;
		}
		
	_render.setName(getName());

    return config->mapObject( &_frameData, frameDataID );
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

	if (node->checkStatus())
	{
		_render.setOctree(node->getOctree(getDevice()));
		node->getCacheHandler(getDevice(), _render.getCacheHandler());

		// Check status node
		const FrameData& frameData = getFrameData();
		if (node->updateStatus(getDevice(), _render.getCacheHandler(), frameData.getCurrentOctree()) &&
			_render.checkCudaResources())
			return &_render;
		else
			return 0;
	}
	else
		return 0;

}

}
