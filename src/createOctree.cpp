/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#define CREATE_OCTREE_VERSION 0.1

#include "octreeConstructor.h"

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>

// GLOBAL VARS
std::string					type_file;
std::vector<std::string>	file_params;
int							maxLevel;
std::vector<float>			isosurfaceList;
std::string					octree_file_name;

bool checkParameters(const int argc, char ** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("version,v", "print version")
    ("help", "produce help message")
    ("data-file,d", boost::program_options::value< std::vector<std::string> >()->multitoken(), "type-data-file data-file-path level-cube-data\nType file supported:\nhdf5_file file-path:data-set-name level-cube-data")
	("list-isosurfaces,l", boost::program_options::value< std::vector<float> >()->multitoken(), "list isosurfaces: iso0<float> iso1<float> iso2<float> ...")
	("range-isosurfaces,r", boost::program_options::value< std::vector<float> >()->multitoken(), "set by range [isoA, isoB] chunk: isoA<float> isoB<float> chunk<float>")
    ;

	boost::program_options::variables_map vm;
	try
	{
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);
	}
	catch( ... )
	{
        std::cout << desc << "\n";
		return false;
	}

    if (argc == 1 || vm.count("help"))
    {
        std::cout << desc << "\n";
		return false;
    }
    if (vm.count("version"))
    {
        std::cout << "Version eqMivt: "<<CREATE_OCTREE_VERSION<< "\n";
		return false;
    }
	if (vm.count("data-file"))
	{
		std::vector<std::string> dataParam = vm["data-file"].as< std::vector<std::string> >();

		if (dataParam.size() != 3)
		{
			std::cerr <<"data-file option: type-file-data<string> file-path<string> level-cube<int>" << std::endl;
			return false;
		}

		type_file = dataParam[0];

		std::vector<std::string> fileParams;
		boost::char_separator<char> sep(":");
		boost::tokenizer< boost::char_separator<char> > tokensO(dataParam[1], sep);

		BOOST_FOREACH (const std::string& t, tokensO)
		{
			fileParams.push_back(t);
		}

		if (dataParam[0] == "hdf5_file" && fileParams.size() != 2)
		{
			std::cerr <<"data-file option: hdf5_file  file-path:data-set-name<string> level-cube<int>" << std::endl;
			return false;

		}

		file_params = fileParams;
		octree_file_name = file_params[0];
		octree_file_name.erase(octree_file_name.find_last_of("."), std::string::npos);
		octree_file_name += ".octree";

		try
		{
			 maxLevel = boost::lexical_cast<int>(dataParam[2]);
			 if (maxLevel <= 0)
			 {
				std::cerr<<"Max level has to be > 0"<<std::endl;
				return false;
			 }
		} 
		catch( boost::bad_lexical_cast const& )
		{
			std::cerr <<"data-file option: type-file-data<string> file-path<string> level-cube<int>" << std::endl;
			return  false;
		}
	}
	else
		return false;

	bool setIso = false;
	if (vm.count("list-isosurfaces"))
	{
		setIso = true;

		isosurfaceList = vm["list-isosurfaces"].as< std::vector<float> >();
	}
	if (vm.count("range-isosurfaces"))
	{
		if (setIso)
		{
			std::cout << desc << "\n";
			return false;
		}
		else
			setIso = true;

		std::vector<float> ranges = vm["range-isosurfaces"].as< std::vector<float> >();

		if (ranges.size() != 3 || ranges[0] > ranges[1] || (ranges[1]-ranges[0]) < ranges[2])
		{
			std::cout << desc << "\n";
			return false;
		}

		float iso = ranges[0];
		while(iso <= ranges[1])
		{
			isosurfaceList.push_back(iso);
			iso += ranges[2];
		}
	}

	if (!setIso)
	{
		std::cout << desc << "\n";
		return false;
	}


	return true;
}


int main( const int argc, char ** argv)
{
	if (!checkParameters(argc, argv))
		return 0;

	for (std::vector<float>::iterator it = isosurfaceList.begin() ; it != isosurfaceList.end(); ++it)
	    std::cout << ' ' << *it<<std::endl;

	eqMivt::octreeConstructor octree;

	if (!octree.createOctree(type_file, file_params, maxLevel, isosurfaceList, octree_file_name))
		return 0;

	return 0;
}
