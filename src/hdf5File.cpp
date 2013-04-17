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

bool hdf5File::init(std::vector<std::string> file_params, int p_levelCube, int p_nLevels, vmml::vector<3, int> p_cubeDim, vmml::vector<3, int> p_cubeInc) 
{
	if (!FileManager::init(file_params, p_levelCube, p_nLevels, p_cubeDim, p_cubeInc))
	{
		LBERROR<<"FileManager: error initialization"<<file_params[0]<<std::endl;
		error = true;
		return false;
	}

	if ((file_id    = H5Fopen(file_params[0].c_str(), H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
	{
		LBERROR<<"hdf5: opening "<<file_params[0]<<std::endl;
		error = true;
		return false;
	}

	if ((dataset_id = H5Dopen1(file_id, file_params[1].c_str())) < 0 )
	{
		LBERROR<<"hdf5: unable to open the requested data set "<<file_params[1]<<std::endl;
		error = true;
		return false;
	}

	if ((spaceid    = H5Dget_space(dataset_id)) < 0)
	{
		LBERROR<<"hdf5: unable to open the requested data space"<<std::endl;
		error = true;
		return false;
	}

	if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
	{
		LBERROR<<"hdf5: handling file"<<std::endl;
		error = true;
		return false;
	}

	return true;
}

hdf5File::~hdf5File()
{
	herr_t      status;

	if ((status = H5Dclose(dataset_id)) < 0)
	{
		LBERROR<<"hdf5: unable to close the data set"<<std::endl;
		error = true;
	}


	if ((status = H5Fclose(file_id)) < 0);
	{
		LBERROR<<"hdf5: unable to close the file"<<std::endl;
		/*
		 * XXX cduelo: No entiendo porque falla al cerrar el fichero....
		 *
		 */
		 // error = true execpGen;
	}
}


void hdf5File::readCube(index_node_t index, float * cube)
{
	vmml::vector<3, int> coord 	= getMinBoxIndex2(index, levelCube, nLevels);
	vmml::vector<3, int> s 		= coord - cubeInc;
	vmml::vector<3, int> e 		= s + realCubeDim;

	hsize_t dim[3] = {abs(e.x()-s.x()),abs(e.y()-s.y()),abs(e.z()-s.z())};

	// Set zeros's
	bzero(cube, dim[0]*dim[1]*dim[2]*sizeof(float));

	// The data required is completly outside of the dataset
	if (s.x() >= (int)this->dims[0] || s.y() >= (int)this->dims[1] || s.z() >= (int)this->dims[2] || e.x() < 0 || e.y() < 0 || e.z() < 0)
	{
		#if DEBUG
		LBERROR<<"Warning: reading cube outsite the volume "<<std::endl;
		LBERROR<<"Dimension valume "<<this->dims[0]<<" "<<this->dims[1]<<" "<<this->dims[2]<<std::endl;
		LBERROR<<"start "<<s.x()<<" "<<s.y()<<" "<<s.z()<<std::endl;
		LBERROR<<"end "<<e.x()<<" "<<e.y()<<" "<<e.z()<<std::endl;
		LBERROR<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
		#endif
		return;
	}

	herr_t	status;
	hid_t	memspace; 
	hsize_t offset_out[3] 	= {s.x() < 0 ? abs(s.x()) : 0, s.y() < 0 ? abs(s.y()) : 0, s.z() < 0 ? abs(s.z()) : 0};
	hsize_t offset[3] 	= {s.x() < 0 ? 0 : s.x(), s.y() < 0 ? 0 : s.y(), s.z() < 0 ? 0 : s.z()};
	hsize_t dimR[3]		= {e.x() > (int)this->dims[0] ? this->dims[0] - offset[0] : e.x() - offset[0],
				   e.y() > (int)this->dims[1] ? this->dims[1] - offset[1] : e.y() - offset[1],
				   e.z() > (int)this->dims[2] ? this->dims[2] - offset[2] : e.z() - offset[2]};

	#if DEBUG 
	std::cout<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
	std::cout<<"Dimension hyperSlab "<<dimR[0]<<" "<<dimR[1]<<" "<<dimR[2]<<std::endl;
	std::cout<<"Offset in "<<offset[0]<<" "<<offset[1]<<" "<<offset[2]<<std::endl;
	std::cout<<"Offset out "<<offset_out[0]<<" "<<offset_out[1]<<" "<<offset_out[2]<<std::endl;
	#endif


	/* 
	* Define hyperslab in the dataset. 
	*/
	if ((status = H5Sselect_hyperslab(spaceid, H5S_SELECT_SET, offset, NULL, dimR, NULL)) < 0)
	{
		LBERROR<<"hdf5: defining hyperslab in the dataset"<<std::endl;
		error = true;
	}

	/*
	* Define the memory dataspace.
	*/
	if ((memspace = H5Screate_simple(3, dim, NULL)) < 0)
	//if ((memspace = H5Screate_simple(3, dimR, NULL)) < 0)
	{
		LBERROR<<"hdf5: defining the memory space"<<std::endl;
		error = true;
	}


	/* 
	* Define memory hyperslab. 
	*/
	if ((status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, dimR, NULL)) < 0)
	{
		LBERROR<<"hdf5: defining the memory hyperslab"<<std::endl;
		error = true;
	}

	/*
	* Read data from hyperslab in the file into the hyperslab in 
	* memory and display.
	*/
	if ((status = H5Dread(dataset_id, H5T_IEEE_F32LE, memspace, spaceid, H5P_DEFAULT, cube)) < 0)
	{
		LBERROR<<"hdf5: reading data from hyperslab un the file"<<std::endl;
		error = true;
	}


	if ((status = H5Sclose(memspace)) < 0)
	{
		LBERROR<<"hdf5: closing dataspace"<<std::endl;
		error = true;
	}
}
}
