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
		int			getOctreeMaxLevel() { return _maxLevelOctree; }
		std::string	getOctreeFilename() const { return _octreeFilename; }
		int			getOctreeMaxLevel() const { return _maxLevelOctree; }
		void		setOctreeFilename(std::string octFilename) {  _octreeFilename = octFilename; }
		void		setOctreeMaxLevel(int maxLevelOct) { _maxLevelOctree = maxLevelOct; }

		std::string	getDataTypeFile() { return _dataTypeFile; }
		std::vector<std::string>	getDataFilename() { return _dataFilename; }
		int			getCubeLevelData() { return _cubeLevelData; }
		std::string	getDataTypeFile() const { return _dataTypeFile; }
		std::vector<std::string>	getDataFilename() const { return _dataFilename; }
		int			getCubeLevelData() const { return _cubeLevelData; }
		void		setDataTypeFile(std::string dataTypeFile) { _dataTypeFile = dataTypeFile; }
		void		setDataFilename(std::vector<std::string> dataFilename) { _dataFilename = dataFilename; }
		void		setCubeLevelData(int cubeLevelData) { _cubeLevelData = cubeLevelData; }

		int			getMaxCubesCacheCPU() { return _maxCubesCacheCPU; }
		int			getMaxCubesCacheCPU() const { return _maxCubesCacheCPU; }
		void		setMaxCubesCacheCPU(int maxCubesCacheCPU){ _maxCubesCacheCPU = maxCubesCacheCPU; }

		int			getMaxCubesCacheGPU() { return _maxCubesCacheGPU; }
		int			getMaxCubesCacheGPU() const { return _maxCubesCacheGPU; }
		void		setMaxCubesCacheGPU(int maxCubesCacheGPU){ _maxCubesCacheGPU = maxCubesCacheGPU; }

    protected:
        virtual void getInstanceData( co::DataOStream& os );
        virtual void applyInstanceData( co::DataIStream& is );

		bool checkCubeLevels() { return _maxLevelOctree < _cubeLevelData; }
    private:
        eq::uint128_t	_frameDataID;

		std::string		_octreeFilename;
		int				_maxLevelOctree;

		std::string		_dataTypeFile;
		std::vector<std::string>		_dataFilename;
		int				_cubeLevelData;

		int				_maxCubesCacheCPU;
		int				_maxCubesCacheGPU;

    };
}


#endif // EQ_MIVT_INITDATA_H
