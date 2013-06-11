/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <hdf5File.h>

#include <mortonCodeUtil_CPU.h>

#include <iostream>
#include <strings.h>

namespace eqMivt
{

bool hdf5File::init(std::vector<std::string> file_params) 
{
	if ((_file_id    = H5Fopen(file_params[0].c_str(), H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
	{
		LBERROR<<"hdf5: opening "<<file_params[0]<<std::endl;
		return false;
	}

	if ((_dataset_id = H5Dopen1(_file_id, file_params[1].c_str())) < 0 )
	{
		LBERROR<<"hdf5: unable to open the requested data set "<<file_params[1]<<std::endl;
		return false;
	}

	if (( _datatype = H5Dget_type(_dataset_id)) < 0)
	{
		LBERROR<<"hdf5: unable to data set type"<<std::endl;
		return false;
	}

	if ((_spaceid    = H5Dget_space(_dataset_id)) < 0)
	{
		LBERROR<<"hdf5: unable to open the requested data space"<<std::endl;
		return false;
	}

	if ((_ndim       = H5Sget_simple_extent_dims (_spaceid, _dims, NULL)) < 0)
	{
		LBERROR<<"hdf5: handling file"<<std::endl;
		return false;
	}

	if (file_params.size() == 2)
	{
		_xGrid = "";
		_yGrid = "";
		_zGrid = "";
	}
	else
	{
		_xGrid = file_params[2];
		_yGrid = file_params[3];
		_zGrid = file_params[4];
	}

	_isInit = true;

	return true;
}

hdf5File::~hdf5File()
{
	herr_t      status;

	if ((status = H5Dclose(_dataset_id)) < 0)
	{
		LBERROR<<"hdf5: unable to close the data set"<<std::endl;
	}


	if ((status = H5Fclose(_file_id)) < 0);
	{
		LBERROR<<"hdf5: unable to close the file"<<std::endl;
		/*
		 * XXX cduelo: No entiendo porque falla al cerrar el fichero....
		 *
		 */
	}
}

vmml::vector<3, int> hdf5File::getRealDimension()
{
	return vmml::vector<3, int>(_dims[0],_dims[1],_dims[2]);
}

bool hdf5File::getxGrid(double ** xGrid)
{
	if (!_isInit)
		return false;
	
	(*xGrid) = new double[_dims[0]];

	if (_xGrid == "")
	{
		for(int i=0; i<_dims[0]; i++)
			(*xGrid)[i] = i;
	}
	else
	{
		hid_t           dataset_id;
		hid_t           spaceid;
		hid_t			datatype;
		herr_t	status;

		if ((dataset_id = H5Dopen1(_file_id, _xGrid.c_str())) < 0 )
		{
			delete [] (*xGrid);
			LBERROR<<"hdf5: unable to open the requested data set "<<_xGrid<<std::endl;
			return false;
		}

		if (( datatype = H5Dget_type(dataset_id)) < 0)
		{
			delete [] (*xGrid);
			LBERROR<<"hdf5: unable to data set type"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((spaceid    = H5Dget_space(dataset_id)) < 0)
		{
			delete [] (*xGrid);
			LBERROR<<"hdf5: unable to open the requested data space"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		int ndim = 0;
		hsize_t dims[3];
		if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
		{
			delete [] (*xGrid);
			LBERROR<<"hdf5: handling file"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}
		if (ndim != 1 || dims[0] != _dims[0])
		{
			LBERROR<<"hdf5: x grid dimension no equal to data dimension"<<std::endl;
			delete [] (*xGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((status = H5Dread(dataset_id, H5T_IEEE_F64LE/*datatype*/, H5S_ALL, spaceid, H5P_DEFAULT, (*xGrid))) < 0)
		{
			delete [] (*xGrid);
			LBERROR<<"hdf5: reading x grid"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((status = H5Dclose(dataset_id)) < 0)
		{
			delete [] (*xGrid);
			LBERROR<<"hdf5: unable to close the data set"<<std::endl;
			return false;
		}

	}
	return true;
}

bool hdf5File::getyGrid(double ** yGrid)
{
	if (!_isInit)
		return false;
	(*yGrid) = new double[_dims[1]];

	if (_yGrid == "")
	{
		for(int i=0; i<_dims[1]; i++)
			(*yGrid)[i] = i;
	}
	else
	{
		hid_t           dataset_id;
		hid_t           spaceid;
		hid_t			datatype;
		herr_t	status;

		if ((dataset_id = H5Dopen1(_file_id, _yGrid.c_str())) < 0 )
		{
			LBERROR<<"hdf5: unable to open the requested data set "<<_yGrid<<std::endl;
			delete [] (*yGrid);
			return false;
		}

		if (( datatype = H5Dget_type(dataset_id)) < 0)
		{
			LBERROR<<"hdf5: unable to data set type"<<std::endl;
			delete [] (*yGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((spaceid    = H5Dget_space(dataset_id)) < 0)
		{
			LBERROR<<"hdf5: unable to open the requested data space"<<std::endl;
			delete [] (*yGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		int ndim = 0;
		hsize_t dims[3];
		if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
		{
			LBERROR<<"hdf5: handling file"<<std::endl;
			delete [] (*yGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}
		if (ndim != 1 || dims[0] != _dims[1])
		{
			LBERROR<<"hdf5: y grid dimension no equal to data dimension"<<std::endl;
			delete [] (*yGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((status = H5Dread(dataset_id, H5T_IEEE_F64LE/*datatype*/, H5S_ALL, spaceid, H5P_DEFAULT, (*yGrid))) < 0)
		{
			LBERROR<<"hdf5: reading y grid"<<std::endl;
			delete [] (*yGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((status = H5Dclose(dataset_id)) < 0)
		{
			LBERROR<<"hdf5: unable to close the data set"<<std::endl;
			delete [] (*yGrid);
			return false;
		}

	}
	return true;
}

bool hdf5File::getzGrid(double ** zGrid)
{
	if (!_isInit)
		return false;
	(*zGrid) = new double[_dims[2]];

	if (_zGrid == "")
	{
		for(int i=0; i<_dims[2]; i++)
			(*zGrid)[i] = i;
	}
	else
	{
		hid_t           dataset_id;
		hid_t           spaceid;
		hid_t			datatype;
		herr_t	status;

		if ((dataset_id = H5Dopen1(_file_id, _zGrid.c_str())) < 0 )
		{
			LBERROR<<"hdf5: unable to open the requested data set "<<_zGrid<<std::endl;
			delete [] (*zGrid);
			return false;
		}

		if (( datatype = H5Dget_type(dataset_id)) < 0)
		{
			LBERROR<<"hdf5: unable to data set type"<<std::endl;
			delete [] (*zGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((spaceid    = H5Dget_space(dataset_id)) < 0)
		{
			LBERROR<<"hdf5: unable to open the requested data space"<<std::endl;
			delete [] (*zGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		int ndim = 0;
		hsize_t dims[3];
		if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
		{
			LBERROR<<"hdf5: handling file"<<std::endl;
			delete [] (*zGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}
		if (ndim != 1 || dims[0] != _dims[2])
		{
			LBERROR<<"hdf5: z grid dimension no equal to data dimension"<<std::endl;
			delete [] (*zGrid);
			return false;
		}

		if ((status = H5Dread(dataset_id, H5T_IEEE_F64LE/*datatype*/, H5S_ALL, spaceid, H5P_DEFAULT, (*zGrid))) < 0)
		{
			LBERROR<<"hdf5: reading z grid"<<std::endl;
			delete [] (*zGrid);
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				LBERROR<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((status = H5Dclose(dataset_id)) < 0)
		{
			LBERROR<<"hdf5: unable to close the data set"<<std::endl;
			delete [] (*zGrid);
			return false;
		}

	}

	return true;
}

void hdf5File::readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> cubeInc, vmml::vector<3, int> realCubeDim)
{
	if (!_isInit)
		return;

	vmml::vector<3, int> coord 	= getMinBoxIndex2(index, levelCube, nLevels);
	// ADD offset
	coord += _offset;
	vmml::vector<3, int> s 		= coord - cubeInc;
	vmml::vector<3, int> e 		= s + realCubeDim;

	hsize_t dim[3] = {abs(e.x()-s.x()),abs(e.y()-s.y()),abs(e.z()-s.z())};

	// Set zeros's
	bzero(cube, dim[0]*dim[1]*dim[2]*sizeof(float));

	// The data required is completly outside of the dataset
	if (s.x() >= (int)this->_dims[0] || s.y() >= (int)this->_dims[1] || s.z() >= (int)this->_dims[2] || e.x() < 0 || e.y() < 0 || e.z() < 0)
	{
		LBERROR<<"Warning: reading cube outsite the volume "<<std::endl;
		LBERROR<<"Dimension volume "<<this->_dims[0]<<" "<<this->_dims[1]<<" "<<this->_dims[2]<<std::endl;
		LBERROR<<"Cube dimension "<<cubeDim<<" in level "<<levelCube<<std::endl;
		LBERROR<<"Cube inc "<<cubeInc<<std::endl;
		LBERROR<<"Dimension volume "<<this->_dims[0]<<" "<<this->_dims[1]<<" "<<this->_dims[2]<<std::endl;
		LBERROR<<"Index "<<index<<" Coordinate "<<coord<<std::endl;
		LBERROR<<"start "<<s.x()<<" "<<s.y()<<" "<<s.z()<<std::endl;
		LBERROR<<"end "<<e.x()<<" "<<e.y()<<" "<<e.z()<<std::endl;
		LBERROR<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;

		return;
	}

	herr_t	status;
	hid_t	memspace; 
	hsize_t offset_out[3] 	= {s.x() < 0 ? abs(s.x()) : 0, s.y() < 0 ? abs(s.y()) : 0, s.z() < 0 ? abs(s.z()) : 0};
	hsize_t offset[3] 	= {s.x() < 0 ? 0 : s.x(), s.y() < 0 ? 0 : s.y(), s.z() < 0 ? 0 : s.z()};
	hsize_t dimR[3]		= {e.x() > (int)this->_dims[0] ? this->_dims[0] - offset[0] : e.x() - offset[0],
				   e.y() > (int)this->_dims[1] ? this->_dims[1] - offset[1] : e.y() - offset[1],
				   e.z() > (int)this->_dims[2] ? this->_dims[2] - offset[2] : e.z() - offset[2]};

	#ifdef NDEBUG
	LBINFO<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
	LBINFO<<"Dimension hyperSlab "<<dimR[0]<<" "<<dimR[1]<<" "<<dimR[2]<<std::endl;
	LBINFO<<"Offset in "<<offset[0]<<" "<<offset[1]<<" "<<offset[2]<<std::endl;
	LBINFO<<"Offset out "<<offset_out[0]<<" "<<offset_out[1]<<" "<<offset_out[2]<<std::endl;
	#endif
    
	/* 
	* Define hyperslab in the dataset. 
	*/
	if ((status = H5Sselect_hyperslab(_spaceid, H5S_SELECT_SET, offset, NULL, dimR, NULL)) < 0)
	{
		LBERROR<<"hdf5: defining hyperslab in the dataset"<<std::endl;
	}

	/*
	* Define the memory dataspace.
	*/
	if ((memspace = H5Screate_simple(3, dim, NULL)) < 0)
	//if ((memspace = H5Screate_simple(3, dimR, NULL)) < 0)
	{
		LBERROR<<"hdf5: defining the memory space"<<std::endl;
	}


	/* 
	* Define memory hyperslab. 
	*/
	if ((status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, dimR, NULL)) < 0)
	{
		LBERROR<<"hdf5: defining the memory hyperslab"<<std::endl;
	}

	/*
	* Read data from hyperslab in the file into the hyperslab in 
	* memory and display.
	*/
	if ((status = H5Dread(_dataset_id, H5T_IEEE_F32LE/*_datatype*/, memspace, _spaceid, H5P_DEFAULT, cube)) < 0)
	{
		LBERROR<<"hdf5: reading data from hyperslab un the file"<<std::endl;
	}


	if ((status = H5Sclose(memspace)) < 0)
	{
		LBERROR<<"hdf5: closing dataspace"<<std::endl;
	}
}
}
