/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_INITDATA_H
#define EQ_MIVT_INITDATA_H

#include "eqMivt.h"

namespace eqMivt
{
    class InitData : public co::Object
    {
    public:
        InitData();
        virtual ~InitData();

        void setFrameDataID( const eq::uint128_t& id ) { _frameDataID = id; }

        eq::uint128_t getFrameDataID() const  { return _frameDataID; }

    protected:
        virtual void getInstanceData( co::DataOStream& os );
        virtual void applyInstanceData( co::DataIStream& is );

    private:
        eq::uint128_t    _frameDataID;
    };
}


#endif // EQ_MIVT_INITDATA_H
