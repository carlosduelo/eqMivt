/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_INITDATA_H
#define EQ_MIVT_INITDATA_H

#include "eqMivt.h"

#include <string>

namespace eqMivt
{
    class InitData : public co::Object
    {
    public:
        InitData();
        virtual ~InitData();

        void setFrameDataID( const eq::uint128_t& id ) { _frameDataID = id; }

        eq::uint128_t getFrameDataID() const  { return _frameDataID; }

	std::string	getOctreeFilename() { return _octreeFilename; }
	int		getOctreeMaxLevel() { return _maxLevelOctree; }
	std::string	getOctreeFilename() const { return _octreeFilename; }
	int		getOctreeMaxLevel() const { return _maxLevelOctree; }
	void		setOctreeFilename(std::string octFilename) {  _octreeFilename = octFilename; }
	void		setOctreeMaxLevel(int maxLevelOct) { _maxLevelOctree = maxLevelOct; }

	const InitData& operator = ( const InitData& from );
    protected:
        virtual void getInstanceData( co::DataOStream& os );
        virtual void applyInstanceData( co::DataIStream& is );

    private:
        eq::uint128_t    _frameDataID;
	std::string	_octreeFilename;
	int		_maxLevelOctree;
    };
}


#endif // EQ_MIVT_INITDATA_H
