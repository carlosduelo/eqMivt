/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_INITDATA_H
#define EQ_MIVT_INITDATA_H

#include "eqMivt.h"

#include <string>
#include <vector>

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
		std::string	getOctreeFilename() const { return _octreeFilename; }
		void		setOctreeFilename(std::string octFilename) {  _octreeFilename = octFilename; }

		std::string	getDataTypeFile() { return _dataTypeFile; }
		std::vector<std::string>	getDataFilename() { return _dataFilename; }
		std::string	getDataTypeFile() const { return _dataTypeFile; }
		std::vector<std::string>	getDataFilename() const { return _dataFilename; }
		void		setDataTypeFile(std::string dataTypeFile) { _dataTypeFile = dataTypeFile; }
		void		setDataFilename(std::vector<std::string> dataFilename) { _dataFilename = dataFilename; }

    protected:
        virtual void getInstanceData( co::DataOStream& os );
        virtual void applyInstanceData( co::DataIStream& is );

    private:
        eq::uint128_t	_frameDataID;

		std::string		_octreeFilename;

		std::string						_dataTypeFile;
		std::vector<std::string>		_dataFilename;
    };
}


#endif // EQ_MIVT_INITDATA_H
