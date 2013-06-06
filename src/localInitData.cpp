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

	setDataTypeFile(from.getDataTypeFile());
	setDataFilename(from.getDataFilename());
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
    ("octree-file,o", boost::program_options::value< std::vector<std::string> >()->multitoken(), "octree-file-path")
    ("data-file,d", boost::program_options::value< std::vector<std::string> >()->multitoken(), "type-data-file data-file-path\nType file supported: hdf5_file file-path:data-set-name")
	("max-elements-cpu,c", boost::program_options::value<int>(), "set cpu cache, optional")
	("max-elements-gpu,g", boost::program_options::value<int>(), "set gpu cache, optional")
    ;

	boost::program_options::variables_map vm;
	try
	{
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);
	}
	catch( ... )
	{
        std::cerr << desc << std::endl;
		return false;
	}

    if (argc == 1 || vm.count("help"))
    {
        std::cout << desc << std::endl;
		return false;
    }
    if (vm.count("version"))
    {
        std::cout << "Version eqMivt: "<<VERSION_EQ_MIVT << std::endl;
		return false;
    }
	if (vm.count("max-elements-gpu"))
	{
		setMaxCubesCacheGPU(vm["max-elements-gpu"].as<int>() <= 0 ? 1 : vm["max-elements-gpu"].as<int>());
	}
	else 
	{
		setMaxCubesCacheGPU(0);
	}
	if (vm.count("max-elements-cpu"))
	{
		setMaxCubesCacheCPU(vm["max-elements-cpu"].as<int>() <= 0 ? 1 : vm["max-elements-cpu"].as<int>());
	}
	else 
	{
		setMaxCubesCacheCPU(0);
	}

	bool printHelp = false;
    if (vm.count("octree-file"))
    {
        std::vector<std::string> octreefiles = vm["octree-file"].as< std::vector<std::string> >();

		if (octreefiles.size() != 1)
			printHelp = true;
		else
			setOctreeFilename(octreefiles[0]);
    }
    // Parameter needed
    else
    {
		setOctreeFilename("");
		printHelp = true;
    }

	if (vm.count("data-file"))
	{
		std::vector<std::string> dataParam = vm["data-file"].as< std::vector<std::string> >();

		if (dataParam.size() != 2)
			printHelp = true;
		else
		{
			setDataTypeFile(dataParam[0]);

			std::vector<std::string> fileParams;
			boost::char_separator<char> sep(":");
			boost::tokenizer< boost::char_separator<char> > tokensO(dataParam[1], sep);

			BOOST_FOREACH (const std::string& t, tokensO)
			{
				fileParams.push_back(t);
			}

			if (dataParam[0] == "hdf5_file" && fileParams.size() != 2)
				printHelp = true;
			else
				setDataFilename(fileParams);
		}
	}
    // Parameter needed
    else
    {
		setDataTypeFile("");
		std::vector<std::string> fileParams;
		setDataFilename(fileParams);
		printHelp = true;
    }

	if (printHelp)
        std::cerr << desc << std::endl;

    return !printHelp;
}
}
