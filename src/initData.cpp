/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "initData.h"

namespace eqMivt
{

InitData::InitData()
        : _frameDataID()
{}

InitData::~InitData()
{
    setFrameDataID( 0 );
}

void InitData::getInstanceData( co::DataOStream& os )
{
    os << _frameDataID << octreeFilename << maxLevelOctree;
}

void InitData::applyInstanceData( co::DataIStream& is )
{
    is >> _frameDataID >> octreeFilename >> maxLevelOctree;
 
    LBASSERT( _frameDataID != 0 );
}

}
