



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


/*
void detectBurst(LineInfo &line, const quint16 *inputData)
{








}
*/

void decodeFieldGPU(const SourceField &inputField, const double *chromaData, double chromaGain, RGBFrame &outputFrame)
{



}


    struct LineInfo {
        //explicit LineInfo(qint32 number);

        qint32 number;
        double bp, bq;
        double Vsw;
    };



void decodeFieldGPU(const SourceField &inputField, const double *chromaData, double chromaGain, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, double sine[], double cosine[])
{
	//work in progress

	//std::cout << "ran function" << std::endl;


    	const qint32 firstLine = inputField.getFirstActiveLine(videoParameters);
    	const qint32 lastLine = inputField.getLastActiveLine(videoParameters);
	//22 310


	const int numOfLines = lastLine - firstLine;

	int arraySize = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;


	std::vector<int> lines(lastLine - firstLine);

	std::iota(lines.begin(), lines.end(), firstLine);

	//lastLine - firstLine
	std::vector<LineInfo> lineInfos(lastLine - firstLine);


	//lineInfos[0].number = 999;


	std::cout << "Line Info Number:" << lineInfos[0].number << std::endl;

	//lineInfos.resize(1, {0, 0, 0, 0});

	//std::cout << lines[0] << std::endl;
	//std::cout << "lines" << lastLine - firstLine << std::endl;

	int colourBurstLength = videoParameters.colourBurstEnd - videoParameters.colourBurstStart;


	//std::cout << " colour burst value: " << colourBurstLength << std::endl;

	std::vector<double> c(10);



	std::cout << "width: " << lastLine - firstLine << std::endl;



	{

		cl::sycl::queue myQueue;

		//cl::sycl::buffer<unsigned short> bufChroma(inputField.data.data(), inputField.data.size());


		//cl::sycl::buffer<int> bufColourBurstLength(&colourBurstLength, cl::sycl::range<1>(1));
		cl::sycl::buffer<double> buff_c(c.data(), c.size());

		//cl::sycl::buffer<int> bufFieldWidth(&videoParameters.fieldWidth, cl::sycl::range<1>(1));
		//cl::sycl::buffer<int> bufFieldHeight(&videoParameters.fieldHeight, cl::sycl::range<1>(1));

		cl::sycl::buffer<double> bufSine(sine, cl::sycl::range<1>(1135));
		cl::sycl::buffer<double> bufCosine(cosine, cl::sycl::range<1>(1135));

		cl::sycl::buffer<unsigned short> bufInputData(inputField.data.data(), cl::sycl::range<1>(arraySize * 288));


		cl::sycl::buffer<int> bufFirstLineNum(&firstLine, cl::sycl::range<1>(1));

		cl::sycl::buffer<LineInfo> bufLineInfo(lineInfos.data(), cl::sycl::range<1>(lastLine - firstLine));

		std::cout << c.size() << std::endl;


std::cout << "Running on "
        << myQueue.get_device().get_info<cl::sycl::info::device::name>()
        << "\n";


	std::cout << "max work group size: " << myQueue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();




		myQueue.submit([&](cl::sycl::handler& cgh)
		{
			//auto accessChroma = bufChroma.get_access<cl::sycl::access::mode::read>(cgh);
			//auto accessLines = 
			//auto accessColourBurstLength = bufColourBurstLength.get_access<cl::sycl::access::mode::read>(cgh);
			auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

			//auto accessFieldWidth = bufFieldWidth.get_access<cl::sycl::access::mode::read>(cgh);
			//auto accessFieldHeight = bufFieldHeight.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessSine = bufSine.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessCosine = bufCosine.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessInputData = bufInputData.get_access<cl::sycl::access::mode::read_write>(cgh);

			auto accessFirstLineNum = bufFirstLineNum.get_access<cl::sycl::access::mode::read>(cgh);

			auto accessLineInfo = bufLineInfo.get_access<cl::sycl::access::mode::write>(cgh);


			cgh.parallel_for<class vector_chroma>(cl::sycl::range<1>{lines.size()}, [=](cl::sycl::item<1> tid)
			{
				//access_c[0] = accessColourBurstLength[0];
				//access_c[0] = videoParameters.fieldHeight;
				//access_c[0] = accessFirstLineNum[0];


				//accessLineInfo[lineNum].number = 

				if (tid.get_id(0) == 287)
					access_c[0] = 999;


				int lineNum = tid.get_id(0);



				accessLineInfo[lineNum].number = lineNum + accessFirstLineNum[0];//lineNum + accessFirstLineNum[0];


				    static constexpr quint16 blackLine[1135] = {0};



				unsigned short *pointerInputData = accessInputData.get_pointer();



	int fullLineNum = lineNum + accessFirstLineNum[0];


	// Get pointers to the surrounding lines of input data.
    // If a line we need is outside the field, use blackLine instead.
    // (Unlike below, we don't need to stay in the active area, since we're
    // only looking at the colourburst.)
    const unsigned short *in0, *in1, *in2, *in3, *in4;
    in0 =                                                                 pointerInputData +  (fullLineNum      * videoParameters.fieldWidth);
    in1 = (fullLineNum - 1) <  0                           ? blackLine : (pointerInputData + ((fullLineNum - 1) * videoParameters.fieldWidth));
    in2 = (fullLineNum + 1) >= videoParameters.fieldHeight ? blackLine : (pointerInputData + ((fullLineNum + 1) * videoParameters.fieldWidth));
    in3 = (fullLineNum - 2) <  0                           ? blackLine : (pointerInputData + ((fullLineNum - 2) * videoParameters.fieldWidth));
    in4 = (fullLineNum + 2) >= videoParameters.fieldHeight ? blackLine : (pointerInputData + ((fullLineNum + 2) * videoParameters.fieldWidth));





    double bp = 0.0, bq = 0.0, bpo = 0.0, bqo = 0.0;
	//int i = 1;

    for (unsigned int i = videoParameters.colourBurstStart; i < videoParameters.colourBurstEnd; i++) {
        bp += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * accessSine[i];
        bq += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * accessCosine[i];
        bpo += ((in2[i] - in1[i]) / 2.0) * accessSine[i];
        bqo += ((in2[i] - in1[i]) / 2.0) * accessCosine[i];
    }


//bp = 4443.0;






    // Normalise the sums above
    const unsigned int colourBurstLength = videoParameters.colourBurstEnd - videoParameters.colourBurstStart;
    bp /= colourBurstLength;
    bq /= colourBurstLength;
    bpo /= colourBurstLength;
    bqo /= colourBurstLength;





    //accessLineInfo[lineNum].Vsw = -1;

	//double num1 = 5.0;//(((double)bp - (double)bpo) * ((double)bp - (double)bpo) + ((double)bq - (double)bqo) * ((double)bq - (double)bqo));


	//double part1 = 5.3 * 3.3;
	//double part2 = bq * bq;

	//double num2 = 6.0;//(((double)bp * (double)bp + (double)bq * (double)bq) * (double)2.0);









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















				//access_c[0] = in0[0];

				if (tid.get_id(0) == 0)
				{

					access_c[0] = bp;//in0[0];
					access_c[1] = (lineNum + accessFirstLineNum[0])      * videoParameters.fieldWidth;
					//access_c[0] = 5.0;
				access_c[2] = lineNum + accessFirstLineNum[0];
				}



				//access_c[0] = bp;
				access_c[3] = 55.0;





			});
















		});





	}


	//for (const auto x: lineInfos)
		std::cout << "Test Output:::" << "output: " << lineInfos[0].bq << std::endl;



}




