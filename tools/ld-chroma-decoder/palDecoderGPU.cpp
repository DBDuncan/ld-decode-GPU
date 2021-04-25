



#include <array>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>


//#include <CL/cl.hpp>

#include <CL/sycl.hpp>


#include "rgbframe.h"
#include "lddecodemetadata.h"

#include "palDecoderGPU.h"

//a change


DecodePAL::DecodePAL(double sine[], double cosine[])
{

	
	//cl::sycl::buffer<double> bufSin{sine, cl::sycl::range<1>(1135)};
	//cl::sycl::buffer<double> bufCosin{cosine, cl::sycl::range<1>(1135)};

	//bufSine{sine, cl::sycl::range<1>(1135)};

	//bufCosine{cosine, cl::sycl::range<1>(1135)};

	
	cl::sycl::buffer<double> bufSineTemp(sine, cl::sycl::range<1>(1135));
	cl::sycl::buffer<double> bufCosineTemp(cosine, cl::sycl::range<1>(1135));

	myQueue.submit([&](cl::sycl::handler& cgh)
	{

		auto accessSineTemp = bufSineTemp.get_access<cl::sycl::access::mode::read>(cgh);

		auto accessCosineTemp = bufCosineTemp.get_access<cl::sycl::access::mode::read>(cgh);

		auto accessSine = bufSine.get_access<cl::sycl::access::mode::discard_write>(cgh);
		auto accessCosine = bufCosine.get_access<cl::sycl::access::mode::discard_write>(cgh);


		cgh.parallel_for<class initData>(cl::sycl::range<1>{1135}, [=](cl::sycl::item<1> tid)
		{
			int i = tid.get_id(0);


			accessSine[i] = accessSineTemp[i];
			accessCosine[i] = accessCosineTemp[i];



		});






	});

}

DecodePAL::~DecodePAL()
{}

/*
void DecodePAL::setupSine(double sine[], double cosine[])
{

	m_sine = sine;

	m_cosine = cosine;



}
*/





__host__ __device__
double opti(double num1, double num2)
{

	#ifdef SYCL_DEVICE_ONLY
		#ifdef HIPSYCL_PLATFORM_CUDA
		__syncthreads();
		return max(num1, num2);
		#endif
	#endif
	return 0;	
}

/*
void decodeFieldGPU(const SourceField &inputField, const double *chromaData, double chromaGain, RGBFrame &outputFrame)
{

	int a;

}
*/

/*
	struct LineInfo {
		//explicit LineInfo(qint32 number);

		qint32 number;
		double bp, bq;
		double Vsw;
	};
*/


/*
struct InInfo {

	const unsigned short* in0;
	const unsigned short* in1;
	const unsigned short* in2;
	const unsigned short* in3;
	const unsigned short* in4;
	const unsigned short* in5;
	const unsigned short* in6;

};


struct FilterComponents {

	double pu;
	double qu;
	double pv;
	double qv;
	double py;
	double qy;



};
*/


void DecodePAL::decodeFieldGPU(const SourceField &inputField, const SourceField &inputFieldTwo, const double *chromaData, double chromaGain, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, double sine[], double cosine[], double cfilt[][4], double yfilt[][2])
{
	//work in progress

	//std::cout << "Line Width: " <<videoParameters.activeVideoEnd - videoParameters.activeVideoStart << std::endl;

	const qint32 firstLine = inputField.getFirstActiveLine(videoParameters);
	const qint32 lastLine = inputField.getLastActiveLine(videoParameters);
	//22 310

	const int offset = inputField.getOffset();

	//std::cout << offset << std::endl;

	const qint32 firstLineFieldTwo = inputFieldTwo.getFirstActiveLine(videoParameters);

	const int numOfLines = lastLine - firstLine;

	int arraySize = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

	//std::vector<int> lines(lastLine - firstLine);

	int numLinesField = lastLine - firstLine;

	//576
	const size_t numLinesFrame = numLinesField * 2;


	//std::cout << "numLinesFrame" << numLinesFrame << std::endl;


	//std::iota(lines.begin(), lines.end(), firstLine);

	int colourBurstLength = videoParameters.colourBurstEnd - videoParameters.colourBurstStart;

	//array for test outputs
	//std::vector<double> c(100);


	//std::cout << numLinesFrame << " : " << numLinesFrame << std::endl;

	//std::cout << "width: " << lastLine - firstLine << std::endl;

	//std::cout << "colour burst length: " << videoParameters.colourBurstEnd - videoParameters.colourBurstStart << std::endl;


	//bracketed to make sure buffers deconstruct when calcuation is done and transfer data back.
	{

		//queue which is used to execure kernel jobs
		//cl::sycl::queue myQueue;

		//buffer for accessing test data
		//cl::sycl::buffer<double> buff_c(c.data(), c.size());

		//buffer for sine and cosine. need to look into calculating data on GPU
		//cl::sycl::buffer<double> bufSine(sine, cl::sycl::range<1>(1135));
		//cl::sycl::buffer<double> bufCosine(cosine, cl::sycl::range<1>(1135));

		//buffer for input dataa
		const qint32 frameHeightTwo = (videoParameters.fieldHeight * 2) - 1;
		cl::sycl::buffer<unsigned short> bufInputData(inputField.data.data(), cl::sycl::range<1>((inputField.data.size())));//was arraySize * 288

		cl::sycl::buffer<unsigned short> bufInputDataTwo(inputFieldTwo.data.data(), cl::sycl::range<1>(inputFieldTwo.data.size()));


		//buffer containing offset of first line (prob 22) also prob needs to be removed
		cl::sycl::buffer<int> bufFirstLineNum(&firstLine, cl::sycl::range<1>(1));

		//buffer for lineinfo structs
		//cl::sycl::buffer<LineInfo> bufLineInfo(lineInfos.data(), cl::sycl::range<1>(lastLine - firstLine));
		//cl::sycl::buffer<LineInfo> bufLineInfo{cl::sycl::range<1>(numLinesFrame)};//was lastLine - firstLine


		//test buffer
		//cl::sycl::buffer<LineInfo> bufTest{cl::sycl::range<1>(200)};

		//buffer of structs containing pointers.
		//cl::sycl::buffer<InInfo> bufInInfo{cl::sycl::range<1>(numLinesFrame)};

		//first two nums are set, last is of how many lines.
		//cl::sycl::buffer<double, 3> bufM{cl::sycl::range<3>(4, 1135, numLinesFrame)};
		//cl::sycl::buffer<double, 3> bufN{cl::sycl::range<3>(4, 1135, numLinesFrame)};


		//accessor of filter component structs to help calculate colour of each pixel.
		//cl::sycl::buffer<FilterComponents, 2> bufFilterComponents{cl::sycl::range<2>(numLinesFrame, 1135)};// was 1135 288
		
		//buffer of c and y filt. maybe can be calculated on GPU?
		//cl::sycl::buffer<double, 2> bufCfilt(*cfilt, cl::sycl::range<2>(7 + 1, 4));
		//cl::sycl::buffer<double, 2> bufYfilt(*yfilt, cl::sycl::range<2>(7 + 1, 2));


		//output buffer
		int frameHeight = (videoParameters.fieldHeight * 2) - 1;
		cl::sycl::buffer<unsigned short> bufOutput{outputFrame.data(), cl::sycl::range<1>(videoParameters.fieldWidth * frameHeight * 3)};


		//test PU buffer. needs to be removed.
		//cl::sycl::buffer<double, 2> bufPU{cl::sycl::range<2>(288, 1135)};//was 1135 288


		cl::sycl::buffer<LdDecodeMetaData::VideoParameters> bufVideoPara(&videoParameters, cl::sycl::range<1>(1));


		//cl::sycl::buffer<unsigned short> bufBlackLine{cl::sycl::range<1>(1135)};

		//keep for easy output of GPU device on system
		//std::cout << "Running on "
		//<< myQueue.get_device().get_info<cl::sycl::info::device::name>()
		//<< "\n";


	//std::cout << "max work group size: " << myQueue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();




		myQueue.submit([&](cl::sycl::handler& cgh)
		{
			//accessor of buffer used to output test data
			//auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

			//accessor of sine and cosine data. Need to look into calculating sine and cosine when needed on GPU
			auto accessSine = bufSine.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessCosine = bufCosine.get_access<cl::sycl::access::mode::read>(cgh);

			//accessor of input data
			auto accessInputData = bufInputData.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessInputDataTwo = bufInputDataTwo.get_access<cl::sycl::access::mode::read>(cgh);

			//accessor of the number of the first line offset. prob no longer need.
			auto accessFirstLineNum = bufFirstLineNum.get_access<cl::sycl::access::mode::read>(cgh);

			//line info accessor of structs of line info
			auto accessLineInfo = bufLineInfo.get_access<cl::sycl::access::mode::discard_read_write>(cgh);


			//test accessor
			//auto accessTest = cl::sycl::accessor<LineInfo, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local, cl::sycl::access::placeholder::false_t>(cl::sycl::range<1>(lastLine - firstLine), cgh);

			//buffer test
			//auto accessBufTest = bufTest.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

			//accessor of In pointers.
			///auto accessInInfo = bufInInfo.get_access<cl::sycl::access::mode::discard_read_write>(cgh);


			//M and N array accessors
			///auto accessM = bufM.get_access<cl::sycl::access::mode::discard_read_write>(cgh);
			///auto accessN = bufN.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

			//accessor of filter component structs to store colour components
			//auto accessFilterComponents = bufFilterComponents.get_access<cl::sycl::access::mode::read_write>(cgh);

			//accessor of cfilt and yfilt. maybe better to generate on GPU?
			//auto accessCfilt = bufCfilt.get_access<cl::sycl::access::mode::read>(cgh);
			//auto accessYfilt = bufYfilt.get_access<cl::sycl::access::mode::read>(cgh); 

			//accessor of output
			//auto accessOutput = bufOutput.get_access<cl::sycl::access::mode::write>(cgh);//changed from read_write

			//test accessor of PU
			//auto accessPU = bufPU.get_access<cl::sycl::access::mode::read_write>(cgh);


			auto accessVideoPara = bufVideoPara.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessBlackLine = bufBlackLine.get_access<cl::sycl::access::mode::read>(cgh);


			cgh.parallel_for<class detectBursts>(cl::sycl::range<1>{numLinesFrame}, [=](cl::sycl::item<1> tid)
			{

				int lineNum = tid.get_id(0);



				




				//not used at all
				//accessLineInfo[lineNum].number = (lineNum / 2) + accessFirstLineNum[0];//lineNum + accessFirstLineNum[0];


				//static constexpr quint16 blackLine[1135] = {0};


				unsigned short *blackLine = accessBlackLine.get_pointer();


				//unsigned short *pointerInputData = accessInputData.get_pointer();

				unsigned short *temp;

				if ((tid.get_id(0) % 2) == 0)//was == 0
				{
					temp = accessInputData.get_pointer();
				}
				else
				{
					temp = accessInputDataTwo.get_pointer();
				}



				unsigned short *pointerInputData = temp;


				int fullLineNum = (lineNum / 2) + accessFirstLineNum[0];


				// Get pointers to the surrounding lines of input data.
				// If a line we need is outside the field, use blackLine instead.
				// (Unlike below, we don't need to stay in the active area, since we're
				// only looking at the colourburst.)
				const unsigned short *in0, *in1, *in2, *in3, *in4;
				in0 =                                                                 pointerInputData +  (fullLineNum      * accessVideoPara[0].fieldWidth);
				in1 = (fullLineNum - 1) <  0                           ? blackLine : (pointerInputData + ((fullLineNum - 1) * accessVideoPara[0].fieldWidth));
				in2 = (fullLineNum + 1) >= videoParameters.fieldHeight ? blackLine : (pointerInputData + ((fullLineNum + 1) * accessVideoPara[0].fieldWidth));
				in3 = (fullLineNum - 2) <  0                           ? blackLine : (pointerInputData + ((fullLineNum - 2) * accessVideoPara[0].fieldWidth));
				in4 = (fullLineNum + 2) >= videoParameters.fieldHeight ? blackLine : (pointerInputData + ((fullLineNum + 2) * accessVideoPara[0].fieldWidth));





				double bp = 0.0, bq = 0.0, bpo = 0.0, bqo = 0.0;

				//double pi = 3.14159265359;

				for (unsigned int i = accessVideoPara[0].colourBurstStart; i < accessVideoPara[0].colourBurstEnd; i++) {
				
				//const double rad = 2 * 3.14159265358979323846264338327950288 * i * videoParameters.fsc / videoParameters.sampleRate;
				//double sine = cl::sycl::sin(rad);
				//double cosine = cl::sycl::cos(rad);



				bp += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * accessSine[i];
				bq += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * accessCosine[i];
				bpo += ((in2[i] - in1[i]) / 2.0) * accessSine[i];
				bqo += ((in2[i] - in1[i]) / 2.0) * accessCosine[i];
				
				//accessSine[i] = sine;
				//accessCosine[i] = cosine;

				}


				// Normalise the sums above
				const unsigned int colourBurstLength = accessVideoPara[0].colourBurstEnd - accessVideoPara[0].colourBurstStart;
				bp /= colourBurstLength;
				bq /= colourBurstLength;
				bpo /= colourBurstLength;
				bqo /= colourBurstLength;





				accessLineInfo[lineNum].Vsw = -1;


				if ((((bp - bpo) * (bp - bpo) + (bq - bqo) * (bq - bqo)) < (bp * bp + bq * bq) * 2))
				{
					accessLineInfo[lineNum].Vsw = 1;
				}



				// Average the burst phase to get -U (reference) phase out -- burst
				// phase is (-U +/-V). bp and bq will be of the order of 1000.
				accessLineInfo[lineNum].bp = (bp - bqo) / 2;
				accessLineInfo[lineNum].bq = (bq + bpo) / 2;

				// Normalise the magnitude of the bp/bq vector to 1.
				// Kill colour if burst too weak.
				// XXX magic number 130000 !!! check!
				const double burstNorm = cl::sycl::max(cl::sycl::sqrt(accessLineInfo[lineNum].bp * accessLineInfo[lineNum].bp + accessLineInfo[lineNum].bq * accessLineInfo[lineNum].bq), 130000.0 / 128);
				accessLineInfo[lineNum].bp /= burstNorm;
				accessLineInfo[lineNum].bq /= burstNorm;





/*
				//inserted part from another function because this needs to run just one on each line anyway

				//static constexpr unsigned int blackLine[1135] = {0};

				// Get pointers to the surrounding lines of input data.
				// If a line we need is outside the active area, use blackLine instead.
				const qint32 firstLine2 = firstLine;//inputField.getFirstActiveLine(videoParameters);//look into replacing with accessors
				const qint32 lastLine2 = lastLine;//inputField.getLastActiveLine(videoParameters);
				const unsigned short *in0Two, *in1Two, *in2Two, *in3Two, *in4Two, *in5Two, *in6Two;
				in0Two =                                               pointerInputData +  (fullLineNum      * accessVideoPara[0].fieldWidth);
				in1Two = (fullLineNum - 1) <  firstLine2 ? blackLine : (pointerInputData + ((fullLineNum - 1) * accessVideoPara[0].fieldWidth));
				in2Two = (fullLineNum + 1) >= lastLine2  ? blackLine : (pointerInputData + ((fullLineNum + 1) * accessVideoPara[0].fieldWidth));
				in3Two = (fullLineNum - 2) <  firstLine2 ? blackLine : (pointerInputData + ((fullLineNum - 2) * accessVideoPara[0].fieldWidth));
				in4Two = (fullLineNum + 2) >= lastLine2  ? blackLine : (pointerInputData + ((fullLineNum + 2) * accessVideoPara[0].fieldWidth));
				in5Two = (fullLineNum - 2) <  firstLine2 ? blackLine : (pointerInputData + ((fullLineNum - 3) * accessVideoPara[0].fieldWidth));
				in6Two = (fullLineNum + 3) >= lastLine2  ? blackLine : (pointerInputData + ((fullLineNum + 3) * accessVideoPara[0].fieldWidth));

				accessInInfo[lineNum].in0 = in0Two;
				accessInInfo[lineNum].in1 = in1Two;
				accessInInfo[lineNum].in2 = in2Two;
				accessInInfo[lineNum].in3 = in3Two;
				accessInInfo[lineNum].in4 = in4Two;
				accessInInfo[lineNum].in5 = in5Two;
				accessInInfo[lineNum].in6 = in6Two;
*/
/*
				//test code
				if (tid.get_id(0) == 1)
				{

					access_c[0] = accessLineInfo[lineNum].bp;//in0[0];
					//access_c[1] = (lineNum + accessFirstLineNum[0])      * accessVideoPara[0].fieldWidth;
					//access_c[0] = 5.0;
					//access_c[2] = lineNum + accessFirstLineNum[0];
					access_c[3] = in0[182];
				}
*/
			});

		});


		myQueue.submit([&](cl::sycl::handler& cgh)
		{


			//accessor of input data
			auto accessInputData = bufInputData.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessInputDataTwo = bufInputDataTwo.get_access<cl::sycl::access::mode::read>(cgh);


			auto accessInInfo = bufInInfo.get_access<cl::sycl::access::mode::discard_write>(cgh);

			auto accessBlackLine = bufBlackLine.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessVideoPara = bufVideoPara.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessFirstLineNum = bufFirstLineNum.get_access<cl::sycl::access::mode::read>(cgh);

			cgh.parallel_for<class detectBurstsTwo>(cl::sycl::range<1>{numLinesFrame}, [=](cl::sycl::item<1> tid)
			{
			
				int lineNum = tid.get_id(0);

				int fullLineNum = (lineNum / 2) + accessFirstLineNum[0];
				unsigned short *blackLine = accessBlackLine.get_pointer();
				unsigned short *temp;

				if ((tid.get_id(0) % 2) == 0)//was == 0
				{
					temp = accessInputData.get_pointer();
				}
				else
				{
					temp = accessInputDataTwo.get_pointer();
				}



				unsigned short *pointerInputData = temp;



				//inserted part from another function because this needs to run just one on each line anyway

				//static constexpr unsigned int blackLine[1135] = {0};

				// Get pointers to the surrounding lines of input data.
				// If a line we need is outside the active area, use blackLine instead.
				const qint32 firstLine2 = firstLine;//inputField.getFirstActiveLine(videoParameters);//look into replacing with accessors
				const qint32 lastLine2 = lastLine;//inputField.getLastActiveLine(videoParameters);
				const unsigned short *in0Two, *in1Two, *in2Two, *in3Two, *in4Two, *in5Two, *in6Two;
				in0Two =                                               pointerInputData +  (fullLineNum      * accessVideoPara[0].fieldWidth);
				in1Two = (fullLineNum - 1) <  firstLine2 ? blackLine : (pointerInputData + ((fullLineNum - 1) * accessVideoPara[0].fieldWidth));
				in2Two = (fullLineNum + 1) >= lastLine2  ? blackLine : (pointerInputData + ((fullLineNum + 1) * accessVideoPara[0].fieldWidth));
				in3Two = (fullLineNum - 2) <  firstLine2 ? blackLine : (pointerInputData + ((fullLineNum - 2) * accessVideoPara[0].fieldWidth));
				in4Two = (fullLineNum + 2) >= lastLine2  ? blackLine : (pointerInputData + ((fullLineNum + 2) * accessVideoPara[0].fieldWidth));
				in5Two = (fullLineNum - 2) <  firstLine2 ? blackLine : (pointerInputData + ((fullLineNum - 3) * accessVideoPara[0].fieldWidth));
				in6Two = (fullLineNum + 3) >= lastLine2  ? blackLine : (pointerInputData + ((fullLineNum + 3) * accessVideoPara[0].fieldWidth));

				accessInInfo[lineNum].in0 = in0Two;
				accessInInfo[lineNum].in1 = in1Two;
				accessInInfo[lineNum].in2 = in2Two;
				accessInInfo[lineNum].in3 = in3Two;
				accessInInfo[lineNum].in4 = in4Two;
				accessInInfo[lineNum].in5 = in5Two;
				accessInInfo[lineNum].in6 = in6Two;










			});


		});


			//this cuda function works here!!!!!
			//cudaDeviceSynchronize();

			const size_t lineWidthCustom = videoParameters.activeVideoEnd - videoParameters.activeVideoStart + 1 + 7;

			//std::cout << "numLinesFrame: " << numLinesFrame << std::endl;


		myQueue.submit([&](cl::sycl::handler& cgh)
		{
			//accessor of In pointers.
			auto accessInInfo = bufInInfo.get_access<cl::sycl::access::mode::read>(cgh);


			//M and N array accessors
			auto accessM = bufM.get_access<cl::sycl::access::mode::discard_read_write>(cgh);
			auto accessN = bufN.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

			//accessor of filter component structs to store colour components
			//auto accessFilterComponents = bufFilterComponents.get_access<cl::sycl::access::mode::read_write>(cgh);

			//accessor of cfilt and yfilt. maybe better to generate on GPU?
			//auto accessCfilt = bufCfilt.get_access<cl::sycl::access::mode::read>(cgh);
			//auto accessYfilt = bufYfilt.get_access<cl::sycl::access::mode::read>(cgh); 

			//accessor of output
			auto accessOutput = bufOutput.get_access<cl::sycl::access::mode::write>(cgh);//changed from read_write

			
			//accessor of input data
			auto accessInputData = bufInputData.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessInputDataTwo = bufInputDataTwo.get_access<cl::sycl::access::mode::read>(cgh);



			//accessor of sine and cosine data. Need to look into calculating sine and cosine when needed on GPU
			auto accessSine = bufSine.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessCosine = bufCosine.get_access<cl::sycl::access::mode::read>(cgh);


			auto accessLineInfo = bufLineInfo.get_access<cl::sycl::access::mode::read>(cgh);


			auto accessVideoPara = bufVideoPara.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessFirstLineNum = bufFirstLineNum.get_access<cl::sycl::access::mode::read>(cgh);


			cgh.parallel_for<class createMandN>(cl::sycl::range<2>{numLinesFrame, lineWidthCustom}, [=](cl::sycl::item<2> tid)
			{

				int line = tid.get_id(0);
				//plus active video start for offset
				int col = tid.get_id(1) + accessVideoPara[0].activeVideoStart - 7;

				//line = 576;

				//if (line > 575)
				//line = 3;
				//accessM[0][col][line] = 55;
				//unsigned short num = accessInInfo[line].in0[col];
				//accessM[0][col][line] =  accessInInfo[line].in0[col];


				//accessM[0][col][line] = line + 1;
				//unsigned short num = accessSine[col];
				//unsigned short numTwo = accessInInfo[line].in0[col];		
				//unsigned short numThree = num * numTwo;
				//accessM[0][col][line] = numTwo;


				//accessM[0][col][line] = num * numTwo;




				//const double rad = 2 * 3.14159265358979323846264338327950288 * col * videoParameters.fsc / videoParameters.sampleRate;
				//double sine = cl::sycl::sin(rad);
				//double cosine = cl::sycl::cos(rad);




				accessM[0][col][line] =  accessInInfo[line].in0[col] * accessSine[col];
				accessM[2][col][line] =  accessInInfo[line].in1[col] * accessSine[col] - accessInInfo[line].in2[col] * accessSine[col];
				accessM[1][col][line] = -accessInInfo[line].in3[col] * accessSine[col] - accessInInfo[line].in4[col] * accessSine[col];
				accessM[3][col][line] = -accessInInfo[line].in5[col] * accessSine[col] + accessInInfo[line].in6[col] * accessSine[col];

				accessN[0][col][line] =  accessInInfo[line].in0[col] * accessCosine[col];
				accessN[2][col][line] =  accessInInfo[line].in1[col] * accessCosine[col] - accessInInfo[line].in2[col] * accessCosine[col];
				accessN[1][col][line] = -accessInInfo[line].in3[col] * accessCosine[col] - accessInInfo[line].in4[col] * accessCosine[col];
				accessN[3][col][line] = -accessInInfo[line].in5[col] * accessCosine[col] + accessInInfo[line].in6[col] * accessCosine[col];

			});








			const size_t lineWidth = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;


			//std::cout << "Width of lines:::>>>> " << lineWidth << std::endl;
			//std::cout << "Number of Lines:::>>>>" << lines.size() << std::endl;


			cgh.parallel_for<class chromaFilter>(cl::sycl::range<2>{numLinesFrame, 1135}, [=](cl::sycl::item<2> tid)
			{


				double testValue = 0.0;
				int i = tid.get_id(1) + accessVideoPara[0].activeVideoStart;
				int lineNum = tid.get_id(0);

				double QU = 0.0, PV = 0.0, QV = 0.0, PY = 0.0, QY = 0.0, PU = 0.0;

				double newPU;

				int startTwo = 0;

				//for (int i = videoParameters.activeVideoStart; i < videoParameters.activeVideoEnd; i++) {

				//QU = 0.0, PV = 0.0, QV = 0.0, PY = 0.0, QY = 0.0;

				//double PU[8];

				//newPU = 0.0;

				// Carry out 2D filtering. P and Q are the two arbitrary SINE & COS
				// phases components. U filters for U, V for V, and Y for Y.
				//
				// U and V are the same for lines n ([0]), n+/-2 ([1]), but
				// differ in sign for n+/-1 ([2]), n+/-3 ([3]) owing to the
				// forward/backward axis slant.


				double accessYfilt[8][2] =
				{
					{0.0577985, 0.00517164},
					{0.110596, 0.00975213},
					{0.0964581, 0.00810742},
					{0.0756302, 0.00577103},
					{0.0517164, 0.00326939},
					{0.0288551, 0.00119304},
					{0.0110026, 8.27732e-05},
					{0.00124825, 0.0}

				};


				double accessCfilt[8][4] =
				{
					
					{0.0190385, 0.00851754, 0.0158864, 0.0018121},
					{0.0364297, 0.0160615, 0.0303127, 0.003246},
					{0.0317727, 0.0133527, 0.0261979, 0.00225146},
					{0.0249121, 0.00950472, 0.0201777, 0.00103026},
					{0.0170351, 0.00538458, 0.0133527, 0.000136325},
					{0.00950472, 0.00196491, 0.0069803, 0},
					{0.0036242, 0.000136325, 0.00225146, 0},
					{0.000411166, 0, 7.84703e-05, 0}



				};



				int start = 0;
				int offset = 28;


				startTwo = 0;

				for (int b = 0; b <= 7; b++) {
				int l = i - b;
				int r = i + b;

				PY += (accessM[0][r][lineNum] + accessM[0][l][lineNum]) * accessYfilt[b][0] + (accessM[1][r][lineNum] + accessM[1][l][lineNum]) * accessYfilt[b][1];

				QY += (accessN[0][r][lineNum] + accessN[0][l][lineNum]) * accessYfilt[b][0] + (accessN[1][r][lineNum] + accessN[1][l][lineNum]) * accessYfilt[b][1];








				PU += (accessM[0][r][lineNum] + accessM[0][l][lineNum]) * accessCfilt[b][0] + (accessM[1][r][lineNum] + accessM[1][l][lineNum]) * accessCfilt[b][1]
					+ (accessN[2][r][lineNum] + accessN[2][l][lineNum]) * accessCfilt[b][2] + (accessN[3][r][lineNum] + accessN[3][l][lineNum]) * accessCfilt[b][3];



				//testValue = newPU;

				//testValue = i;


				QU += (accessN[0][r][lineNum] + accessN[0][l][lineNum]) * accessCfilt[b][0] + (accessN[1][r][lineNum] + accessN[1][l][lineNum]) * accessCfilt[b][1]
					- (accessM[2][r][lineNum] + accessM[2][l][lineNum]) * accessCfilt[b][2] - (accessM[3][r][lineNum] + accessM[3][l][lineNum]) * accessCfilt[b][3];
                 
				PV += (accessM[0][r][lineNum] + accessM[0][l][lineNum]) * accessCfilt[b][0] + (accessM[1][r][lineNum] + accessM[1][l][lineNum]) * accessCfilt[b][1]
					- (accessN[2][r][lineNum] + accessN[2][l][lineNum]) * accessCfilt[b][2] - (accessN[3][r][lineNum] + accessN[3][l][lineNum]) * accessCfilt[b][3];

				QV += (accessN[0][r][lineNum] + accessN[0][l][lineNum]) * accessCfilt[b][0] + (accessN[1][r][lineNum] + accessN[1][l][lineNum]) * accessCfilt[b][1]
					+ (accessM[2][r][lineNum] + accessM[2][l][lineNum]) * accessCfilt[b][2] + (accessM[3][r][lineNum] + accessM[3][l][lineNum]) * accessCfilt[b][3];

				//test code here
/*
				if (lineNum == 250)
				{
					if (i == 500 + accessVideoPara[0].activeVideoStart)
					{

						access_c[offset + start] += (accessM[0][r][lineNum] + accessM[0][l][lineNum]) * accessCfilt[b][0] + (accessM[1][r][lineNum] + accessM[1][l][lineNum]) * accessCfilt[b][1]
							+ (accessN[2][r][lineNum] + accessN[2][l][lineNum]) * accessCfilt[b][2] + (accessN[3][r][lineNum] + accessN[3][l][lineNum]) * accessCfilt[b][3];

						//access_c[37] = 

						if (start == 7)
						{

							access_c[offset + 8] = PU;

							access_c[39] = lineNum;
							access_c[40] = i;

							//accessPU[lineNum][i] = testValue;


							//accessFilterComponents[lineNum][i].pu = 1224;
						}

						start++;

					}
				}
			startTwo++;

			//}
					//}

*/
				}


				//accessFilterComponents[lineNum][i].pu = PU;//PU;//testValue;//newPU;
				//accessFilterComponents[lineNum][i].qu = QU;
				//accessFilterComponents[lineNum][i].pv = PV;
				//accessFilterComponents[lineNum][i].qv = QV;
				//accessFilterComponents[lineNum][i].py = PY;
				//accessFilterComponents[lineNum][i].qy = QY;




			

			//});

/*
			myQueue.submit([&](cl::sycl::handler& cgh)
			{
			auto accessInputData = bufInputData.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessVideoPara = bufVideoPara.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessOutput = bufOutput.get_access<cl::sycl::access::mode::read_write>(cgh);

			auto accessFilterComponents = bufFilterComponents.get_access<cl::sycl::access::mode::read_write>(cgh);

			auto accessSine = bufSine.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessCosine = bufCosine.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessInInfo = bufInInfo.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

			auto accessLineInfo = bufLineInfo.get_access<cl::sycl::access::mode::discard_read_write>(cgh);


			auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

			const size_t lineWidth = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
*/

			//cgh.parallel_for<class decodeRGB>(cl::sycl::range<2>{numLinesFrame, lineWidth}, [=](cl::sycl::item<2> tid)
			//{

				int lineNumber = (int)tid.get_id(0) / 2;
				lineNum = lineNumber;
				int linePixel = tid.get_id(1) + accessVideoPara[0].activeVideoStart;
				i = linePixel;

				int realLineNum = lineNumber + firstLine;


				int lineNumFull = tid.get_id(0);

				//unsigned short *tempTwo = accessInputData.get_pointer();

				unsigned short *inputTemp;
				//unsigned short *temp = accessOutput.get_pointer();	

				int addedNum = 0;

				if ((tid.get_id(0) % 2) == 0)
				{
					inputTemp = accessInputData.get_pointer();
				}
				else
				{
					inputTemp = accessInputDataTwo.get_pointer();
					//realLineNum = lineNumber + firstLineFieldTwo;
					//temp = temp + 1; 
					//realLineNum += 1;
					addedNum = 1;
					realLineNum = lineNumber + firstLineFieldTwo;
				}

				unsigned short *tempTwo = inputTemp;



				unsigned short *temp = accessOutput.get_pointer();

				// Pointer to composite signal data
				const unsigned short *comp = tempTwo + (realLineNum * accessVideoPara[0].fieldWidth);

				// Define scan line pointer to output buffer using 16 bit unsigned words //was inputField.getOffset() where zero is
				unsigned short *ptr = temp + (((realLineNum * 2) + 0) * accessVideoPara[0].fieldWidth * 3) + (addedNum * accessVideoPara[0].fieldWidth * 3);

				// Gain for the Y component, to put reference black at 0 and reference white at 65535
				const double scaledContrast = 65535.0 / (accessVideoPara[0].white16bIre - accessVideoPara[0].black16bIre);

				// Gain for the U/V components.
				// The scale is the same as for Y above, doubled because the U/V filters
				// extract the result with half its original amplitude, and with the
				// burst-based correction applied.
				const double scaledSaturation = 2.0 * scaledContrast * chromaGain;


				double rY;


				//const double rad = 2 * 3.14159265358979323846264338327950288 * i * videoParameters.fsc / videoParameters.sampleRate;
				//double sine = cl::sycl::sin(rad);
				//double cosine = cl::sycl::cos(rad);



				//if statement will need to be around the line bellow if prefiltered chroma is being used, but prefiltered chroma is not supported at all at the moment
				rY = comp[i] - ((PY * accessSine[i] + QY * accessCosine[i]) * 2.0);
        


				rY = cl::sycl::clamp((rY - accessVideoPara[0].black16bIre) * scaledContrast, 0.0, 65535.0);



				const double rU = -(PU * accessLineInfo[lineNumFull].bp + QU * accessLineInfo[lineNumFull].bq) * scaledSaturation;
				const double rV = accessLineInfo[lineNumFull].Vsw * -(QV * accessLineInfo[lineNumFull].bp - PV * accessLineInfo[lineNumFull].bq) * scaledSaturation;


				const double R = cl::sycl::clamp(rY + (1.139883 * rV), 0.0, 65535.0);
				const double G = cl::sycl::clamp(rY + (-0.394642 * rU) + (-0.580622 * rV), 0.0, 65535.0);
				const double B = cl::sycl::clamp(rY + (2.032062 * rU), 0.0, 65535.0);



				const int pp = i * 3;

				ptr[pp + 0] = (unsigned short)R;
				ptr[pp + 1] = (unsigned short)G;
				ptr[pp + 2] = (unsigned short)B;

/*
				//extracting data from a spercific pixel for testing purposes
				if (3 == tid.get_id(0))//was lineNumber == 0
				{
					if (linePixel == accessVideoPara[0].activeVideoStart + 0)
					{
						int coll = accessVideoPara[0].activeVideoStart + 0;

						access_c[0] = R;
						access_c[1] = G;
						access_c[2] = B;
						access_c[3] = accessLineInfo[lineNum].bp;
						access_c[4] = accessLineInfo[lineNum].bq;
						access_c[5] = accessLineInfo[lineNum].Vsw;

						access_c[6] = accessInInfo[lineNum].in0[coll];
						access_c[7] = accessInInfo[lineNum].in1[coll];
						access_c[8] = accessInInfo[lineNum].in2[coll];
						access_c[9] = accessInInfo[lineNum].in3[coll];
						access_c[10] = accessInInfo[lineNum].in4[coll];
						access_c[11] = accessInInfo[lineNum].in5[coll];
						access_c[12] = accessInInfo[lineNum].in6[coll]; 

						access_c[13] = accessM[0][coll][lineNum];
						access_c[14] = accessM[2][coll][lineNum];
						access_c[15] = accessM[1][coll][lineNum];
						access_c[16] = accessM[3][coll][lineNum];

						access_c[17] = accessN[0][coll][lineNum];
						access_c[18] = accessN[2][coll][lineNum];
						access_c[19] = accessN[1][coll][lineNum];
						access_c[20] = accessN[3][coll][lineNum];


						access_c[21] = accessFilterComponents[lineNum][i].pu;
						access_c[22] = accessFilterComponents[lineNum][i].qu;
						access_c[23] = accessFilterComponents[lineNum][i].pv;
						access_c[24] = accessFilterComponents[lineNum][i].qv;
						access_c[25] = accessFilterComponents[lineNum][i].py;
						access_c[26] = accessFilterComponents[lineNum][i].qy;

						access_c[27] = pp;


						access_c[41] = lineNum;
						access_c[42] = i;
						access_c[43] = accessFilterComponents[250][282].pu;

						//access_c[44] = accessPU[lineNum][i];

						access_c[45] = rU;
						access_c[46] = rV;
						access_c[47] = rY;
						access_c[48] = realLineNum;//comp[0];

						}
					}
*/


			});

			//little test kernel. will be removed later.
			/*
			cgh.parallel_for<class test>(cl::sycl::range<1>{1}, [=](cl::sycl::item<1> tid)
			{

				//access_c[0] = accessM[0][1134][0];
				//access_c[4] = accessFilterComponents[0][200].pu;
				//num, col, line
				//access_c[5] = accessN[0][500][150];

				//access_c[5] = accessYfilt[5][1];


				//access_c[0] = accessOutput[2046 + (((22 * 2) + inputField.getOffset()) * videoParameters.fieldWidth * 3)];

				//item, col, line
				//access_c[13] = accessM[0][500][1];
				//access_c[14] = accessM[1][500][1];
				//access_c[15] = accessM[2][500][1];
				//access_c[16] = accessM[3][500][1];


			});


*/



		});




	}


	const size_t lineWidth = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;


	//test outputs for verifying decoding
	/*
	std::cout << "Width of lines:::>>>> " << lineWidth << std::endl;
	std::cout << "Number of Lines:::>>>>" << lines.size() << std::endl;



	//for (const auto x: lineInfos)
		//std::cout << "Test Output:::" << "output: " << lineInfos[0].bq << std::endl;


	std::cout << "R: " << c[0] << std::endl;
	std::cout << "::::::::::::::::::" << c[1] << std::endl;
	std::cout << "::::::::::::::::::" << c[2] << std::endl;
	//std::cout << "In[0]: " << c[3] << std::endl;
	std::cout << "bq: " << c[4] << std::endl;
	std::cout << "Vsw: " << c[5] << std::endl;


	std::cout << "In0: " << c[6] << std::endl;
	std::cout << "In1: " << c[7] << std::endl;
	std::cout << "In2: " << c[8] << std::endl;
	std::cout << "In3: " << c[9] << std::endl;
	std::cout << "In4: " << c[10] << std::endl;
	std::cout << "In5: " << c[11] << std::endl;
	std::cout << "In6: " << c[12] << std::endl;


	std::cout << "M1: " << c[13] << std::endl;

	std::cout << "M2: " << c[14] << std::endl;
	std::cout << "M3: " << c[15] << std::endl;
	std::cout << "M4: " << c[16] << std::endl;

	std::cout << "N1: " << c[17] << std::endl;
	std::cout << "N2: " << c[18] << std::endl;
	std::cout << "N3: " << c[19] << std::endl;
	std::cout << "N4: " << c[20] << std::endl;

	std::cout << "pu: " << c[21] << std::endl;
	std::cout << "qu: " << c[22] << std::endl;
	std::cout << "pv: " << c[23] << std::endl;
	std::cout << "qv: " << c[24] << std::endl;
	std::cout << "py: " << c[25] << std::endl;
	std::cout << "qy: " << c[26] << std::endl;


	*/


	//std::cout << "output: " << outputFrame.data()[2000] << std::endl;
	//std::cout << "PP: " << c[27] << std::endl;
	/*
	std::cout << "PU 1: " << c[28] << std::endl;
	std::cout << "PU 2: " << c[29] << std::endl;
	std::cout << "PU 3: " << c[30] << std::endl;
	std::cout << "PU 4: " << c[31] << std::endl;
	std::cout << "PU 5: " << c[32] << std::endl;
	std::cout << "PU 6: " << c[33] << std::endl;
	std::cout << "PU 7: " << c[34] << std::endl;
	std::cout << "PU 8: " << c[35] << std::endl;
	std::cout << "PU 9: " << c[36] << std::endl;

	double total = 0;
	for (int i = 28; i <= 28+7; i++)
		total += c[i];

	std::cout << "PU Total: " << total << std::endl;
	std::cout << "PU Total Two: " << c[37] << std::endl;
	std::cout << "Test: " << c[38] << std::endl;

	std::cout << "LineNum: " << c[39] << std::endl;
	std::cout << "Col: " << c[40] << std::endl;
	std::cout << "LineNum: " << c[41] << std::endl;
	std::cout << "Col: " << c[42] << std::endl;
	std::cout << "test: " << c[43] << std::endl;
	std::cout << "accessPU Value: " << c[44] << std::endl;
	*/
	/*
	std::cout << "rU: " << c[45] << std::endl;
	std::cout << "rV: " << c[46] << std::endl;
	std::cout << "rY: " << c[47] << std::endl;
	std::cout << "comp: " << c[48] << std::endl;
	*/
}



