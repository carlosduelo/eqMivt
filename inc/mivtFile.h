/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_MIVT_FILE_H
#define EQ_MIVT_MIVT_FILE_H

#include <FileManager.h>

#include <openssl/md5.h>

namespace eqMivt
{
class mivtFile : public FileManager
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

		bool operator < (const cubeBuffer_t& str) const
		{
			return id < str.id;
		}
	};

	private:
		unsigned char			_md5sum[MD5_DIGEST_LENGTH];
		vmml::vector<3, int>	_realDimVolume;
		vmml::vector<3, int>	_startC;
		vmml::vector<3, int>	_finishC;
		int						_nLevels;
		int						_levelCube;
		int						_dimCube;
		int						_sizeCube;	

		std::ifstream			_file;

		int						_sizeNodes;
		index_node_t	*		_nodes;
		int				*		_offsets;
		int						_startOffset;

		std::vector<cubeBuffer_t>		_buffer;

		int seekFile(index_node_t index);

	public:

		virtual bool init(std::vector<std::string> file_params);

		virtual bool checkInit(std::string octree_file_name);

		~mivtFile();

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

#endif /* EQ_MIVT_MIVT_FILE_H */
