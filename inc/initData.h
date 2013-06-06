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

		int			getMaxCubesCacheCPU() { return _maxCubesCacheCPU; }
		int			getMaxCubesCacheCPU() const { return _maxCubesCacheCPU; }
		void		setMaxCubesCacheCPU(int maxCubesCacheCPU){ _maxCubesCacheCPU = maxCubesCacheCPU; }

		int			getMaxCubesCacheGPU() { return _maxCubesCacheGPU; }
		int			getMaxCubesCacheGPU() const { return _maxCubesCacheGPU; }
		void		setMaxCubesCacheGPU(int maxCubesCacheGPU){ _maxCubesCacheGPU = maxCubesCacheGPU; }

    protected:
        virtual void getInstanceData( co::DataOStream& os );
        virtual void applyInstanceData( co::DataIStream& is );

    private:
        eq::uint128_t	_frameDataID;

		std::string		_octreeFilename;

		std::string						_dataTypeFile;
		std::vector<std::string>		_dataFilename;

		int				_maxCubesCacheCPU;
		int				_maxCubesCacheGPU;
    };
}


#endif // EQ_MIVT_INITDATA_H
