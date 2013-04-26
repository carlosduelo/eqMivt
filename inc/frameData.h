/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_FRAMEDATA_H
#define EQ_MIVT_FRAMEDATA_H

#include "eqMivt.h"

namespace eqMivt
{
class FrameData : public co::Serializable
{
    public:
		FrameData();
		virtual ~FrameData() {};

		void reset();

		void setCameraPosition( const eq::Vector3f& position );
		void setRotation( const eq::Vector3f& rotation);
		void spinCamera( const float x, const float y );
		void moveCamera( const float x, const float y, const float z );

		const eq::Matrix4f& getCameraRotation() const
		{ return _rotation; }

		const eq::Vector3f& getCameraPosition() const
		{ return _position; }

		void setIdle( const bool idleMode );
		bool isIdle() const { return _idle; }

		void setCurrentViewID( const eq::uint128_t& id );
		eq::uint128_t getCurrentViewID() const { return _currentViewID; }

    protected:
		virtual void serialize( co::DataOStream& os, const uint64_t dirtyBits );
		virtual void deserialize( co::DataIStream& is, const uint64_t dirtyBits );

		enum DirtyBits
		{
			DIRTY_CAMERA = co::Serializable::DIRTY_CUSTOM << 0,
			DIRTY_FLAGS   = co::Serializable::DIRTY_CUSTOM << 1,
			DIRTY_VIEW    = co::Serializable::DIRTY_CUSTOM << 3,
		};
    private:
		eq::Matrix4f _rotation;
		eq::Vector3f _position;
		bool             _idle;
		eq::uint128_t _currentViewID;
};
}
#endif // EQ_MIVT_FRAMEDATA_H
