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

#include <fstream>

// GLOBAL VARS
std::string					type_file;
std::vector<std::string>	file_params;
std::vector<int>			maxLevel;
std::vector< std::vector<float>	>		isosurfaceList;
std::vector<int>			octreesPieces;
std::vector< vmml::vector<3, int> > startCoordinates;
std::vector< int >			octreeDimension;

std::string					config_file_name;
std::string					octree_file_name;
bool						octree_name_set = false;
bool						useCUDA;

float toFloat(std::string  s)
{
	try
	{
		float r = -1.0f;
		r = boost::lexical_cast<float>(s);
		if (r < 0)
		{
			std::cerr<<"coordinates should be > 0"<<std::endl;
			return -1.0f;
		}
		return r;
	}
	catch( boost::bad_lexical_cast const& )
	{
		std::cerr <<"error parsing file" << std::endl;
		return  -1.0f;
	}

}

int toInt(std::string  s)
{
	try
	{
		int r = 0;
		r = boost::lexical_cast<int>(s);
		if (r < 0)
		{
			std::cerr<<"coordinates should be > 0"<<std::endl;
			return -1;
		}
		return r;
	}
	catch( boost::bad_lexical_cast const& )
	{
		std::cerr <<"error parsing file" << std::endl;
		return  -1;
	}

}

bool parseConfigFile(std::string file_name)
{
	std::ifstream infile;
	try
	{
		infile.open(file_name.c_str());
	}
	catch(...)
	{
		std::cerr<<file_name<<" do not exist"<<std::endl;
		return false;
	}

	if (!infile.is_open())
	{
		std::cerr<<file_name<<" do not exist"<<std::endl;
		return false;
	}

	std::string line;
	while (std::getline(infile, line))
	{
		boost::char_separator<char> sep(" ");
		boost::tokenizer< boost::char_separator<char> > tokens(line, sep);
		
		vmml::vector<3, int> start;
		int dimension = 0;	
		boost::tokenizer< boost::char_separator<char> >::iterator tok_iter = tokens.begin();
		start[0] = toInt(*tok_iter);	if (start[0] < 0) {infile.close(); return false;} tok_iter++;
		start[1] = toInt(*tok_iter);	if (start[1] < 0) {infile.close(); return false;} tok_iter++;
		start[2] = toInt(*tok_iter);	if (start[2] < 0) {infile.close(); return false;} tok_iter++;
		dimension = toInt(*tok_iter);	tok_iter++;

		if(dimension == 0 && ((dimension & (dimension - 1)) == 0))
		{
			std::cerr<<"Dimension should be power of 2 and > 0"<<std::endl;
			return false;
		}

		int mL = toInt(*tok_iter); tok_iter++;
		if (mL <= 0)
		{
			std::cerr<<"Errror: max level should be > 0"<<std::endl;
			return false;
		}

		int nLevels = 0;
		/* Calcular dimension del árbol*/
		float aux = logf(dimension)/logf(2.0);
		float aux2 = aux - floorf(aux);
		nLevels = aux2>0.0 ? aux+1 : aux;

		if (mL > nLevels)
		{
			std::cerr<<"Octree level "<<nLevels<<", max level "<<mL<< " should be <= "<<nLevels<<std::endl;
			return false;
		}
		
		startCoordinates.push_back(start);
		octreeDimension.push_back(dimension);
		maxLevel.push_back(mL);

		bool error = false;
		int num = 0;

		if ((*tok_iter).compare("r") == 0)
		{
			tok_iter++;
			float ranges[3];
			ranges[0] = toFloat(*tok_iter); tok_iter++;
			ranges[1] = toFloat(*tok_iter); tok_iter++;
			ranges[2] = toFloat(*tok_iter); tok_iter++;

			if (ranges[0] > ranges[1] || (ranges[1]-ranges[0]) < ranges[2])
			{
				error = true;
			}
			else
			{
				std::vector<float> isos;
				float iso = ranges[0];
				while(iso <= ranges[1])
				{
					isos.push_back(iso);
					iso += ranges[2];
					num++;
				}
				isosurfaceList.push_back(isos);
			}
		}
		else if ((*tok_iter).compare("l") == 0)
		{
			std::vector<float> isos;
			tok_iter++;
			while(tok_iter != tokens.end())
			{
				num++;
				float i = toFloat(*tok_iter);
				if (i <= 0.0f)
				{
					std::cerr<<"Isosurface should be > 0.0"<<std::endl;
					error = true;
					break;
				}
				isos.push_back(i);
				tok_iter++;
			}
			isosurfaceList.push_back(isos);
		}

		if (num > 0)
			octreesPieces.push_back(num);
		else
			error = true;

		if (error)
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			infile.close(); 
			return false;
		}
	}

	infile.close();
	return true;

}

bool checkParameters(const int argc, char ** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("version,v", "print version")
    ("help", "produce help message")
    ("use-CUDA,c", "activate CUDA acceleration, by default CUDA acceleration disable")
    ("data-file,d", boost::program_options::value< std::vector<std::string> >()->multitoken(), "type-data-file data-file-path\nType file supported:\nhdf5_file file-path:data-set-name[:x_grid:y_grid:z_grid]")
	("output-file-name,o", boost::program_options::value< std::vector<std::string> >()->multitoken(), "set name of output file, optional, by default same name as data with extension octree")
	("config-file,f", boost::program_options::value< std::vector<std::string> >()->multitoken(), "config file")
    ;

	boost::program_options::variables_map vm;
	try
	{
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);
	}
	catch( ... )
	{
		std::cout<<"Octree constructor: allows create a octree"<<std::endl; 
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
	if (vm.count("output-file-name"))
	{
		std::vector<std::string> dataParam = vm["output-file-name"].as< std::vector<std::string> >();

		octree_name_set = true;
		octree_file_name = dataParam[0];
	}
	if (vm.count("data-file"))
	{
		std::vector<std::string> dataParam = vm["data-file"].as< std::vector<std::string> >();

		if (dataParam.size() != 2)
		{
			std::cerr <<"data-file option: type-file-data<string> file-path<string>" << std::endl;
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

		if (dataParam[0] == "hdf5_file" && (fileParams.size() != 2 && fileParams.size() != 5))
		{
			std::cerr <<"data-file option: hdf5_file  file-path:data-set-name<string>:[x_grid<string>:y_grid<string>:z_grid<string>]" << std::endl;
			return false;

		}

		file_params = fileParams;
		if (!octree_name_set)
		{
			octree_file_name = file_params[0];
			octree_file_name.erase(octree_file_name.find_last_of("."), std::string::npos);
			octree_file_name += ".octree";
		}

		if (vm.count("use-CUDA"))
		{
			useCUDA = true;
		}
		else
			useCUDA = false;

		if (vm.count("config-file"))
		{
			std::vector<std::string> dataParam = vm["config-file"].as< std::vector<std::string> >();

			config_file_name = dataParam[0];
		}
		else
		{
			std::cout<<desc<<std::endl;
			return false;
		}

	}
	else
	{
		std::cout << desc << "\n";
		return false;
	}

	if (parseConfigFile(config_file_name))
	{
		std::cout<<"Parsing config file..... OK"<<std::endl;
		return true;
	}
	else
	{
		return false;
	}
}


int main( const int argc, char ** argv)
{
	if (!checkParameters(argc, argv))
		return 0;

	#if 0
	for (std::vector<int>::iterator it = octreesPieces.begin() ; it != octreesPieces.end(); ++it)
	    std::cout << ' ' << *it<<std::endl;

	for (std::vector<float>::iterator it = isosurfaceList.begin() ; it != isosurfaceList.end(); ++it)
	    std::cout << ' ' << *it<<std::endl;
	#endif

	if (!eqMivt::createOctree(type_file, file_params, maxLevel, isosurfaceList, octreesPieces, startCoordinates, octreeDimension, octree_file_name, useCUDA))
		return 0;

	return 0;
}
