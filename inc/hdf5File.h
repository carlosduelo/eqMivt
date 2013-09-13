/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_HDF5_FILE_H
#define EQ_MIVT_HDF5_FILE_H

#include <FileManager.h>

#include <hdf5.h>

namespace eqMivt
{
class hdf5File : public FileManager
{
	struct cubeBuffer_t 
	{
		float *			cube;
		int				nLevels;
		int				levelCube;
		vmml::vector<3, int>    cubeDim;
		vmml::vector<3, int>	cubeInc;
		vmml::vector<3, int>	realCubeDim;
		index_node_t	id;

		void copy();

		bool operator < (const cubeBuffer_t& str) const
		{
			return id < str.id;
		}
	};

	private:
		// HDF5 stuff
		hid_t           _file_id;
		hid_t           _dataset_id;
		hid_t           _spaceid;
		int             _ndim;
		hsize_t         _dims[3];
		hid_t			_datatype;

		std::string		_xGrid;
		std::string		_yGrid;
		std::string		_zGrid;

		std::vector<cubeBuffer_t>		_buffer;
	public:

		virtual bool init(std::vector<std::string> file_params);

		virtual bool checkInit(std::string octree_file_name);

		~hdf5File();

		virtual bool getxGrid(double ** xGrid);
		virtual bool getyGrid(double ** yGrid);
		virtual bool getzGrid(double ** zGrid);

		virtual void readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim);

		virtual void read(vmml::vector<3, int> start, vmml::vector<3, int> end, float * data);

		virtual void addCubeToBuffer(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim);

		virtual void readBufferedCubes();

		virtual vmml::vector<3, int> getRealDimension();
};
}

#endif /* EQ_MIVT_HDF5_FILE */
