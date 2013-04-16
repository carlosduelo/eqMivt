/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "localInitData.h"
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

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
    ("octree-file,o", boost::program_options::value< std::vector<std::string> >()->multitoken(), "octree-file-path maximum-level")
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
        std::cout << desc << "\n";
	return false;
    }


    return true;
}
}
