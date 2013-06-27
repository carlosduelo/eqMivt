/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "frameData.h"

namespace eqMivt
{

FrameData::FrameData()
        : _rotation( eq::Matrix4f::ZERO )
        , _position( eq::Vector3f::ZERO )
		, _idle( false )
		, _statistics( false )
		, _drawBox( false )
		, _useGrid( false )
		, _renderCubes( false )
		, _currentOctree( 0 )
		, _numOctrees( 0 )
{
    reset();
}

void FrameData::serialize( co::DataOStream& os, const uint64_t dirtyBits )
{
    co::Serializable::serialize( os, dirtyBits );
    if( dirtyBits & DIRTY_CAMERA )
        os << _position << _rotation;
	if( dirtyBits & DIRTY_FLAGS )
		os << _idle << _statistics << _drawBox << _useGrid << _renderCubes;
	if( dirtyBits & DIRTY_VIEW )
		os << _currentViewID;
	if( dirtyBits & DIRTY_MODEL )
		os << _currentOctree << _numOctrees; 
}

void FrameData::deserialize( co::DataIStream& is, const uint64_t dirtyBits )
{
    co::Serializable::deserialize( is, dirtyBits );
    if( dirtyBits & DIRTY_CAMERA )
        is >> _position >> _rotation;
	if( dirtyBits & DIRTY_FLAGS )
		is >> _idle >> _statistics >> _drawBox >> _useGrid >> _renderCubes;
	if( dirtyBits & DIRTY_VIEW )
		is >> _currentViewID;
	if( dirtyBits & DIRTY_MODEL )
		is >> _currentOctree >> _numOctrees; 
}

void FrameData::setNumOctrees(const int numOctrees)
{
	_numOctrees = numOctrees;
    setDirty( DIRTY_MODEL);
}

void FrameData::setCurrentOctree(const int octree)
{
	_currentOctree = octree;
    setDirty( DIRTY_MODEL);
}

void FrameData::setNextOctree()
{
	if (_currentOctree < (_numOctrees-1))
		_currentOctree++;

    setDirty( DIRTY_MODEL);
}

void FrameData::setPreviusOctree()
{
	if (_currentOctree > 0)
		_currentOctree--;
    setDirty( DIRTY_MODEL);
}

void FrameData::spinCamera( const float x, const float y )
{
    if( x == 0.f && y == 0.f )
        return;

    _rotation.pre_rotate_x( x );
    _rotation.pre_rotate_y( y );
    setDirty( DIRTY_CAMERA );
}

void FrameData::moveCamera( const float x, const float y, const float z )
{
    _position.x() += x;
    _position.y() += y;
    _position.z() += z;

    setDirty( DIRTY_CAMERA );
}

void FrameData::setCameraPosition( const eq::Vector3f& position )
{
    _position = position;
    setDirty( DIRTY_CAMERA );
}

void FrameData::setRotation( const eq::Vector3f& rotation )
{
    _rotation = eq::Matrix4f::IDENTITY;
    _rotation.rotate_x( rotation.x() );
    _rotation.rotate_y( rotation.y() );
    _rotation.rotate_z( rotation.z() );
    setDirty( DIRTY_CAMERA );
}

void FrameData::setIdle( const bool idle )
{
	if( _idle == idle )
		return;

	_idle = idle;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setStatistics()
{ 
	_statistics = !_statistics;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setDrawBox()
{
	_drawBox = !_drawBox;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setUseGrid()
{
	_useGrid = !_useGrid;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setRenderCubes()
{
	_renderCubes = !_renderCubes;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setCurrentViewID( const eq::uint128_t& id )
{
	_currentViewID = id;
	setDirty( DIRTY_VIEW );
}

void FrameData::reset()
{
    eq::Matrix4f model = eq::Matrix4f::IDENTITY;
    model.rotate_x( static_cast<float>( -M_PI_2 ));
    model.rotate_y( static_cast<float>( -M_PI_2 ));

    if( _position == eq::Vector3f( 0.f, 0.f, -2.f ) &&
        _rotation == eq::Matrix4f::IDENTITY)
    {
        _position.z() = 0.0f;
    }
    else
    {
        _position   = eq::Vector3f::ZERO;
        _position.z() = 20.f;
        _position.x() = 5.f;
        _position.y() = 5.f;
        _rotation      = eq::Matrix4f::IDENTITY;
    }
    setDirty( DIRTY_CAMERA );
}

}
