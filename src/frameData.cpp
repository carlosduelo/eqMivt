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
{
    reset();
}

void FrameData::serialize( co::DataOStream& os, const uint64_t dirtyBits )
{
    co::Serializable::serialize( os, dirtyBits );
    if( dirtyBits & DIRTY_CAMERA )
        os << _position << _rotation;
}

void FrameData::deserialize( co::DataIStream& is, const uint64_t dirtyBits )
{
    co::Serializable::deserialize( is, dirtyBits );
    if( dirtyBits & DIRTY_CAMERA )
        is >> _position >> _rotation;
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
        _position.z() = -2.f;
        _rotation      = eq::Matrix4f::IDENTITY;
    }
    setDirty( DIRTY_CAMERA );
}

}
