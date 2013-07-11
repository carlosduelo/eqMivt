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
        , _center( eq::Vector3f::ZERO )
		, _radio( 0.0f )
		, _angle( 0.0f )
		, _idle( false )
		, _statistics( false )
		, _drawBox( true )
		, _useGrid( false )
		, _renderCubes( false )
		, _currentOctree( 0 )
		, _numOctrees( 0 )
{
    reset(0);
}

void FrameData::serialize( co::DataOStream& os, const uint64_t dirtyBits )
{
    co::Serializable::serialize( os, dirtyBits );
    if( dirtyBits & DIRTY_CAMERA )
        os << _position << _rotation << _center << _radio << _angle;
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
        is >> _position >> _rotation >> _center >> _radio >> _angle;
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

    _rotation.rotate_x( x );
    _rotation.rotate_y( y );
    setDirty( DIRTY_CAMERA );
}

void FrameData::zoom( const float x)
{
	_radio += x;
	_position.x() = _center.x() + _radio * sin(_angle);
	_position.z() = _center.z() + _radio * cos(_angle);
    setDirty( DIRTY_CAMERA );
}

void FrameData::moveCamera( const float x, const float y, const float z )
{
	_angle += x;

	_position.x() = _center.x() + _radio * sin(_angle);
	_position.y() +=  y*100.0f;
	_position.z() = _center.z() + _radio * cos(_angle);

	#if 0
	std::cout<<"----------------------------------------------"<<std::endl;
	std::cout<<_rotation<<std::endl;

	_rotation      = eq::Matrix4f::IDENTITY;
	eq::Vector3f look =  _position - _center; 
	look.normalize();
	eq::Vector3f up(0.0f, 1.0f, 0.0f);
	eq::Vector3f right = up.cross(look); 	
	right.normalize();
	up = look.cross(right);
	up.normalize();
	
	std::cout<<look<<std::endl;
	std::cout<<up<<std::endl;
	std::cout<<right<<std::endl;
	
	_rotation[0][0] =  right[0];
	_rotation[0][1] =  right[1];
	_rotation[0][2] =  right[2];
	_rotation[1][0] =  up[0];
	_rotation[1][1] =  up[1];
	_rotation[1][2] =  up[2];
	_rotation[2][0] =  -look[0];
	_rotation[2][1] =  -look[1];
	_rotation[2][2] =  -look[2];

	std::cout<<_rotation<<std::endl;
	std::cout<<_position<<std::endl;
	std::cout<<"----------------------------------------------"<<std::endl;
	#endif

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

void FrameData::reset(OctreeManager * octreeManager)
{
	_position   = eq::Vector3f::ZERO;
	_center		= eq::Vector3f::ZERO;

	if (octreeManager != 0)
	{
		eq::Vector3f start = octreeManager->getCurrentStartCoord(_currentOctree, _useGrid);
		eq::Vector3f end = octreeManager->getCurrentFinishCoord(_currentOctree, _useGrid);
		_center.x() = start.x() + ((end.x()-start.x())/2.0f);
		_center.y() = start.y() + ((end.y()-start.y())/2.0f);
		_center.z() = start.z() + ((end.z()-start.z())/2.0f);
		_radio = 2.0f * fmax(end.x()-start.x(), fmaxf(end.y()-start.y(),end.z()-start.z()));
		_angle = 0;
		_position.x() = _center.x() + _radio * sin(_angle);
		_position.y() = _center.y() + _radio * sin(_angle);
		_position.z() = _center.z() + _radio * cos(_angle);
	}

	_rotation      = eq::Matrix4f::IDENTITY;
    setDirty( DIRTY_CAMERA );
}

}
