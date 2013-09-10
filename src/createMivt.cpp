/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#define CREATE_MIVT_FILE_VERSION 0.1

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>

#include <fileFactory.h>
#include <octreeManager.h>
#include <typedef.h>
#include <mortonCodeUtil_CPU.h>

#include <openssl/md5.h>
#include <cstdio>

// Files
std::string					output_name;
std::string					octree_name;
std::string					type_file;
std::vector<std::string>	file_params;

std::vector<eqMivt::index_node_t> nodes;
std::vector<int> offsets;


int getMD5(const char * filename, unsigned char ** md5sum)
{
	FILE * inFile = fopen (filename, "rb");
	*md5sum = new unsigned char[MD5_DIGEST_LENGTH];
	MD5_CTX mdContext;
	int bytes;
	unsigned char data[1024];
	MD5_Init (&mdContext);
	while ((bytes = fread (data, 1, 1024, inFile)) != 0)
		MD5_Update (&mdContext, data, bytes);
	MD5_Final(*md5sum, &mdContext);
	for(int i = 0; i < MD5_DIGEST_LENGTH; i++) printf("%02x", (*md5sum)[i]);
	printf (" %s\n", filename);
	fclose (inFile);
	return 0;
}

bool addElement(eqMivt::index_node_t id)
{
	int size = nodes.size();

	try
	{
		// Firts
		if (size == 0)
		{
			nodes.push_back(id);
			nodes.push_back(id);
		}
		else if (nodes.back() == (id - (eqMivt::index_node_t)1))
		{
			nodes[size-1] = id;
		}
		else if(nodes.back() == id)
		{
			//std::cout<<"repetido in level "<<level<<" "<< id <<std::endl;
			return true;
		}
		else if(nodes.back() > id)
		{
			std::cout<<"=======>   ERROR: insert index in order "<< id <<" last inserted "<<nodes.back()<<std::endl;
			throw;
		}
		else
		{
			nodes.push_back(id);
			nodes.push_back(id);
		}
	}
	catch (...)
	{
		std::cerr<<"No enough memory aviable"<<std::endl;
		throw;
	}

	return false;
}

bool checkParameters(const int argc, char ** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("version,v", "print version")
    ("help", "produce help message")
    ("data-file,d", boost::program_options::value< std::vector<std::string> >()->multitoken(), "type-data-file data-file-path octree_file\nType file supported:\nhdf5_file file-path:data-set-name octree_file")
	("output-file-name,o", boost::program_options::value< std::vector<std::string> >()->multitoken(), "set name of output file")
    ;

	boost::program_options::variables_map vm;
	try
	{
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);
	}
	catch( ... )
	{
		std::cout<<"Mivt constructor: allows create a mivt data file"<<std::endl; 
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
        std::cout << "Version eqMivt: "<<CREATE_MIVT_FILE_VERSION<< "\n";
		return false;
    }
	if (vm.count("output-file-name") && vm.count("data-file"))
	{
		std::vector<std::string> dataParam = vm["output-file-name"].as< std::vector<std::string> >();

		output_name = dataParam[0];

		dataParam = vm["data-file"].as< std::vector<std::string> >();

		if (dataParam.size() != 3)
		{
			std::cerr <<"data-file option: type-file-data<string> file-path<string> octree-file<string>" << std::endl;
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

		if (dataParam[0] == "hdf5_file" && (fileParams.size() != 2))
		{
			std::cerr <<"data-file option: hdf5_file  file-path:data-set-name<string> octree-file" << std::endl;
			return false;

		}

		file_params = fileParams;

		octree_name = dataParam[2];
	}
	else
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

	eqMivt::FileManager * file = eqMivt::CreateFileManage(type_file, file_params);
	eqMivt::OctreeManager  octree;
	if (!octree.init(octree_name))
	{
		delete file;
		return 0;
	}

	vmml::vector<3,int> startC = octree.getRealDimVolumeData(); 
	vmml::vector<3,int> finishC(0,0,0); 
	int dimCube = 0;

	for(int i=0; i<octree.getNumOctrees(); i++)
	{
		vmml::vector<3, int> currentStartC= octree.getStartCoord(i);
		//std::cout<<startC<<" - "<<currentStartC<<" ==> ";
		if (currentStartC[0] < startC[0] && currentStartC[1] < startC[1] && currentStartC[2] < startC[2])
			startC = currentStartC;
		//std::cout<<startC<<std::endl;

		vmml::vector<3, int> currentFinishC = octree.getFinishCoord(i);
		currentFinishC[1] = octree.getMaxHeight(i);
		//std::cout<<finishC<<" - "<<currentFinishC<<" ==> ";
		if (currentFinishC[0] > finishC[0] && currentFinishC[1] > finishC[1] && currentFinishC[2] > finishC[2])
			finishC = currentFinishC;
		//std::cout<<finishC<<std::endl;

		int currentDim = exp2(octree.getNLevels(i) - octree.getBestCubeLevelCPU(i));
		//std::cout<<currentDim<<" - "<<dimCube<<" ==> ";
		if (dimCube < currentDim)
			dimCube = currentDim;
		//std::cout<<dimCube<<std::endl;

		//std::cout<<std::endl;
	}

	std::cout<<"Selecting Start and finish coordinates...."<<std::endl;
	
	if (startC[0] % dimCube)
		startC[0] -= startC[0] % dimCube;
	if (startC[1] % dimCube)
		startC[1] -= startC[1] % dimCube;
	if (startC[2] % dimCube)
		startC[2] -= startC[2] % dimCube;
	if (finishC[0] % dimCube)
		finishC[0] += dimCube - (finishC[0] % dimCube);
	if (finishC[1] % dimCube)
		finishC[1] += dimCube - (finishC[1] % dimCube);
	if (finishC[2] % dimCube)
		finishC[2] += dimCube - (finishC[2] % dimCube);

	vmml::vector<3, int> realDimVolume = octree.getRealDimVolumeData();
	vmml::vector<3, int> cubeDim(dimCube, dimCube, dimCube);
	vmml::vector<3, int> cubeInc(CUBE_INC, CUBE_INC, CUBE_INC);
	vmml::vector<3, int> realCubeDim = cubeDim + 2 * cubeInc;

	int nLevels = 0;
	int levelCube = 0;
	int dimension = fmaxf(realDimVolume[0], fmaxf(realDimVolume[1],realDimVolume[2]));
	float aux = logf(dimension)/logf(2.0);
	float aux2 = aux - floorf(aux);
	nLevels = aux2>0.0 ? aux+1 : aux;
	dimension = pow(2,nLevels);
	levelCube = nLevels - logf(dimCube)/logf(2.0);

	std::cout<<"Volume dimension "<<realDimVolume<<std::endl;
	std::cout<<"Octree dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<std::endl;
	std::cout<<"Octree nLeves "<<nLevels<<std::endl;

	std::cout<<"Start and Finish Coordinates: "<<startC<<" "<<finishC<<std::endl;
	std::cout<<"Cube dimension "<<dimCube<<" cube level "<<levelCube<<std::endl;

	eqMivt::index_node_t idStart = eqMivt::coordinateToIndex(vmml::vector<3, int>(0,0,0), levelCube, nLevels);
	eqMivt::index_node_t idFinish = eqMivt::coordinateToIndex(vmml::vector<3, int>(dimension-1, dimension-1, dimension-1), levelCube, nLevels);

	for(eqMivt::index_node_t id = idStart; id <= idFinish; id++)
	{
		vmml::vector<3, int> currentSC = eqMivt::getMinBoxIndex2(id, levelCube, nLevels);
		vmml::vector<3, int> currentFC = currentSC + cubeDim; 
		
		if (startC[0] <= currentSC[0] && startC[1] <= currentSC[1] && startC[2] <= currentSC[2] &&
			finishC[0] >= currentFC[0] && finishC[1] >= currentFC[1] && finishC[2] >= currentFC[2])
		{
			addElement(id);
		}
	}

	// set offsets
	int offset = 0;
	for(int i=0; i<nodes.size(); i+=2)
	{
		offsets.push_back(offset);
		#if 1
		offset = nodes[i+1] - nodes[i] + 1;
		#else
		offset += nodes[i+1] - nodes[i] + 1;
		#endif
	}
	#if 0
	for(int i=0; i<nodes.size(); i+=2)
		std::cout<<nodes[i]<<" "<<nodes[i+1]<<" offset "<<offsets[i/2]<<std::endl;
	#endif

	unsigned char * md5sum = 0;
	getMD5(octree_name.c_str(), &md5sum);

	std::ofstream output_file(output_name.c_str(), std::ofstream::binary);

	output_file.write((char*) md5sum, MD5_DIGEST_LENGTH*sizeof(unsigned char));
	output_file.write((char*) realDimVolume.array, 3*sizeof(int));
	output_file.write((char*) &nLevels, sizeof(int));
	output_file.write((char*) &levelCube, sizeof(int));
	output_file.write((char*) startC.array, 3*sizeof(int));
	output_file.write((char*) finishC.array, 3*sizeof(int));

	int sA = nodes.size();
	output_file.write((char*) &sA, sizeof(int));
	output_file.write((char*) nodes.data(), sA*sizeof(eqMivt::index_node_t));
	output_file.write((char*) offsets.data(), (sA/2)*sizeof(int));

	int cS = realCubeDim.x()*realCubeDim.y()*realCubeDim.z();
	float * cube = new float[cS];

	for(int i=0; i<sA; i+=2)
		for(eqMivt::index_node_t id = nodes[i]; id <=nodes[i+1]; id++)
		{
			std::cout<<id<<std::endl;
			file->readCube(id, cube, levelCube, nLevels, cubeDim, cubeInc, realCubeDim);
			output_file.write((char*)cube, cS*sizeof(float));
		}


	output_file.close();

	delete file;
	delete[] cube;
	delete[] md5sum;

	return 0;
}
