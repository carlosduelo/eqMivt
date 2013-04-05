/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "localInitData.h"

namespace eqMivt
{
LocalInitData::LocalInitData()
        : _maxFrames( 0xffffffffu )
	, _isResident( false )
{
}

const LocalInitData& LocalInitData::operator = ( const LocalInitData& from )
{
    _maxFrames   = from._maxFrames;
    _isResident  = from._isResident;

    return *this;
}


void LocalInitData::parseArguments( const int argc, char** argv )
{
}
}
