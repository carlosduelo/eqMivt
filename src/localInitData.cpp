/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "localInitData.h"
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>

#include <vector>

namespace eqMivt
{
LocalInitData::LocalInitData()
        : _maxFrames( 0xffffffffu )
	, _isResident( false )
{
}

const LocalInitData& LocalInitData::operator = ( const LocalInitData& from )
{
    _maxFrames   = from._maxFrames;
    _isResident  = from._isResident;

    setOctreeFilename(from.getOctreeFilename());
    setOctreeMaxLevel(from.getOctreeMaxLevel());

	setDataTypeFile(from.getDataTypeFile());
	setDataFilename(from.getDataFilename());
	setCubeLevelData(from.getCubeLevelData());

	setMaxCubesCacheCPU(from.getMaxCubesCacheCPU());
	setMaxCubesCacheGPU(from.getMaxCubesCacheGPU());

    return *this;
}


bool LocalInitData::parseArguments( const int argc, char** argv )
{
    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("version,v", "print version")
    ("help", "produce help message")
    ("eq-config", "Select equalizer configuration file")
    ("eq-layout", "Select equalizer layout in configuration file")
    ("octree-file,o", boost::program_options::value< std::vector<std::string> >()->multitoken(), "octree-file-path maximum-level")
    ("data-file,d", boost::program_options::value< std::vector<std::string> >()->multitoken(), "type-data-file data-file-path level-cube-data\nType file supported: hdf5_file file-path:data-set-name level-cube-data")
	("size-cpu-cache,c", boost::program_options::value<int>(), "set size in cubes cpu cache")
	("size-gpu-cache,g", boost::program_options::value<int>(), "set size in cubes gpu cache")
    ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);    

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
	return false;
    }
    if (vm.count("version"))
    {
        std::cout << "Version eqMivt: "<<VERSION_EQ_MIVT << "\n";
	return false;
    }

    if (vm.count("octree-file"))
    {
        std::vector<std::string> octreefiles = vm["octree-file"].as< std::vector<std::string> >();

		if (octreefiles.size() != 2)
		{
			std::cout <<"octree-file option: octree-file-path<string> maximum-level<int>" << std::endl;
			return false;
		}

		setOctreeFilename(octreefiles[0]);

		try
		{
			setOctreeMaxLevel(boost::lexical_cast<int>(octreefiles[1]));
		} 
		catch( boost::bad_lexical_cast const& )
		{
			std::cout <<"octree-file option: octree-file-path<string> maximum-level<int>" << std::endl;
			return false;
		}
    }
    // Parameter needed
    else
    {
		setOctreeFilename("");
		setOctreeMaxLevel(0);
    }

	if (vm.count("data-file"))
	{
		std::vector<std::string> dataParam = vm["data-file"].as< std::vector<std::string> >();

		if (dataParam.size() != 3)
		{
			LBERROR <<"data-file option: type-file-data<string> file-path<string> level-cube<int>" << std::endl;
			return false;
		}

		setDataTypeFile(dataParam[0]);

		std::vector<std::string> fileParams;
		boost::char_separator<char> sep(":");
		boost::tokenizer< boost::char_separator<char> > tokensO(dataParam[1], sep);

		BOOST_FOREACH (const std::string& t, tokensO)
		{
			fileParams.push_back(t);
		}

		if (dataParam[0] == "hdf5_file" && fileParams.size() != 2)
		{
			LBERROR <<"data-file option: hdf5_file  file-path:data-set-name<string> level-cube<int>" << std::endl;
			return false;

		}

		setDataFilename(fileParams);

		try
		{
			setCubeLevelData(boost::lexical_cast<int>(dataParam[2]));
		} 
		catch( boost::bad_lexical_cast const& )
		{
			LBERROR <<"data-file option: type-file-data<string> file-path<string> level-cube<int>" << std::endl;
			return false;
		}
	}
    // Parameter needed
    else
    {
		setDataTypeFile("");
		std::vector<std::string> fileParams;
		setDataFilename(fileParams);
		setCubeLevelData(0);
    }

	if (checkCubeLevels())
	{
		LBERROR<<"Cube level have to be <= max level octree"<<std::endl;
		return false;
	}

	if (vm.count("size-cpu-cache"))
	{
		setMaxCubesCacheCPU(vm["size-cpu-cache"].as<int>());
	}
	else 
	{
		setMaxCubesCacheCPU(1);
	}

	if (vm.count("size-gpu-cache"))
	{
		setMaxCubesCacheGPU(vm["size-gpu-cache"].as<int>());
	}
	else 
	{
		setMaxCubesCacheGPU(1);
	}

    return true;
}
}
