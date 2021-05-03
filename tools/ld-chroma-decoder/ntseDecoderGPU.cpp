
#include <array>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
 

//#include <CL/cl.hpp>

#include <CL/sycl.hpp>


#include "rgbframe.h"
#include "lddecodemetadata.h"

#include "ntseDecoderGPU.h"

#include "yiq.h"

#include "deemp.h"


//these two functions are used in kernels. They are automaticly converted to be able to run on the GPU
inline qint32 getFieldID(qint32 lineNumber, int firstFieldPhaseID, int secondFieldPhaseID)
{
    bool isFirstField = ((lineNumber % 2) == 0);

    return isFirstField ? firstFieldPhaseID : secondFieldPhaseID;
}

// NOTE:  lineNumber is presumed to be starting at 1.  (This lines up with how splitIQ calls it)
inline bool getLinePhase(qint32 lineNumber, int firstFieldPhaseID, int secondFieldPhaseID)
{
    qint32 fieldID = getFieldID(lineNumber, firstFieldPhaseID, secondFieldPhaseID);
    bool isPositivePhaseOnEvenLines = (fieldID == 1) || (fieldID == 4);

    int fieldLine = (lineNumber / 2);
    bool isEvenLine = (fieldLine % 2) == 0;

    return isEvenLine ? isPositivePhaseOnEvenLines : !isPositivePhaseOnEvenLines;
}


/*
__host__ __device__
void sync()
{
	#ifdef SYCL_DEVICE_ONLY
		#ifdef HIPSYCL_PLATFORM_CUDA
		__syncthreads();
		#endif
	#endif

}
*/



DecodeNTSC::DecodeNTSC()
{}

DecodeNTSC::~DecodeNTSC()
{}




void DecodeNTSC::decodeFrameGPU(const SourceField &inputFieldOne, const SourceField &inputFieldTwo, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, double yNRLevel, double irescale, double chromaGain, bool whitePoint75)
{

	int frameHeight = ((videoParameters.fieldHeight * 2) - 1);

	outputFrame.resize(videoParameters.fieldWidth * frameHeight * 3);

	{

		cl::sycl::buffer<unsigned short> bufInputDataOne(inputFieldOne.data.data(), inputFieldOne.data.size());
		cl::sycl::buffer<unsigned short> bufInputDataTwo(inputFieldTwo.data.data(), inputFieldTwo.data.size());
		cl::sycl::buffer<unsigned short> bufOutput{outputFrame.data(), cl::sycl::range<1>(videoParameters.fieldWidth * frameHeight * 3)};

		const size_t lines = videoParameters.lastActiveFrameLine - videoParameters.firstActiveFrameLine;
		const size_t width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;




        myQueue.submit([&](cl::sycl::handler& cgh)
		{

			auto accessInputDataOne = bufInputDataOne.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessInputDataTwo = bufInputDataTwo.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessClpBuffer2D = bufClpBuffer2D.get_access<cl::sycl::access::mode::discard_write>(cgh);


//these are two function that were used to test weather performance would be improved by having two split1D functions
//removing the need for an if statement
/*
			cgh.parallel_for<class split1D>(cl::sycl::range<2>{lines/2, width}, [=](cl::sycl::item<2> tid)
            {

				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int col = tid.get_id(1) + videoParameters.activeVideoStart;


				//int lineNumTwo = tid.get_id(0);

				//int finalLineNum;

				unsigned short *temp;



				//if ((tid.get_id(0) % 2) == 0)
				//{
					//temp = accessInputDataOne.get_pointer();
					
				//}
				//else
				//{
					//temp = accessInputDataTwo.get_pointer();

				//}



				temp = accessInputDataOne.get_pointer();




				//temp = accessInputData.get_pointer();

				//dont forget divide by two here!!!
				const unsigned short *line = temp + ((lineNum) * videoParameters.fieldWidth);

				//was twos where ones
				double tc1 = (line[col] - ((line[col - 2] + line[col + 2]) / 2.0)) / 2.0;

				accessClpBuffer2D[(lineNum * 2)][col] = tc1;
			

			});


			cgh.parallel_for<class split1DTwo>(cl::sycl::range<2>{lines/2, width}, [=](cl::sycl::item<2> tid)
            {

				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int col = tid.get_id(1) + videoParameters.activeVideoStart;


				//int lineNumTwo = tid.get_id(0);

				//int finalLineNum;

				unsigned short *temp;


				
				//if ((tid.get_id(0) % 2) == 0)
				//{
					//temp = accessInputDataOne.get_pointer();
					
				//}
				//else
				//{
					//temp = accessInputDataTwo.get_pointer();

				//}
				


				temp = accessInputDataTwo.get_pointer();


				//temp = accessInputData.get_pointer();

				//dont forget divide by two here!!!
				const unsigned short *line = temp + ((lineNum) * videoParameters.fieldWidth);

				//was twos where ones
				double tc1 = (line[col] - ((line[col - 2] + line[col + 2]) / 2.0)) / 2.0;

				accessClpBuffer2D[(lineNum * 2) + 1][col] = tc1;
			
			});


*/
			

			cgh.parallel_for<class split1D>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
            {

				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int col = tid.get_id(1) + videoParameters.activeVideoStart;

				//pointer to access the correct field
				unsigned short *temp;
				
				if ((tid.get_id(0) % 2) == 0)
				{
					temp = accessInputDataOne.get_pointer();
					
				}
				else
				{
					temp = accessInputDataTwo.get_pointer();

				}
				
				//dont forget divide by two here!!!
				const unsigned short *line = temp + ((lineNum / 2) * videoParameters.fieldWidth);

				//was twos where ones
				double tc1 = (line[col] - ((line[col - 2] + line[col + 2]) / 2.0)) / 2.0;

				accessClpBuffer2D[lineNum][col] = tc1;
			});
		});


		myQueue.submit([&](cl::sycl::handler& cgh)
		{
			auto accessClpBuffer2D = bufClpBuffer2D.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

			cgh.parallel_for<class split2D>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
            {

				static constexpr double blackLine[910] = {0};

				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int h = tid.get_id(1) + videoParameters.activeVideoStart;

				double *temp = accessClpBuffer2D.get_pointer();

				const double *previousLine = blackLine;
				if (lineNum - 2 >= videoParameters.firstActiveFrameLine) 
				{
					previousLine = temp + ((lineNum - 2) * videoParameters.fieldWidth);
				}
				const double *currentLine = temp + (lineNum * videoParameters.fieldWidth);
				const double *nextLine = blackLine;
				if (lineNum + 2 < videoParameters.lastActiveFrameLine) 
				{
					nextLine = temp + ((lineNum+ 2) * videoParameters.fieldWidth);
				}

				double kp, kn;

				// Summing the differences of the *absolute* values of the 1D chroma samples
				// will give us a low value if the two lines are nearly in phase (strong Y)
				// or nearly 180 degrees out of phase (strong C) -- i.e. the two cases where
				// the 2D filter is probably usable. Also give a small bonus if
				// there's a large signal (we think).
				kp  = cl::sycl::fabs(cl::sycl::fabs(currentLine[h]) - cl::sycl::fabs(previousLine[h]));
				kp += cl::sycl::fabs(cl::sycl::fabs(currentLine[h - 1]) - cl::sycl::fabs(previousLine[h - 1]));
				kp -= (cl::sycl::fabs(currentLine[h]) + cl::sycl::fabs(previousLine[h - 1])) * .10;
				kn  = cl::sycl::fabs(cl::sycl::fabs(currentLine[h]) - cl::sycl::fabs(nextLine[h]));
				kn += cl::sycl::fabs(fabs(currentLine[h - 1]) - cl::sycl::fabs(nextLine[h - 1]));
				kn -= (cl::sycl::fabs(currentLine[h]) + cl::sycl::fabs(nextLine[h - 1])) * .10;


				double irescale = (videoParameters.white16bIre - videoParameters.black16bIre) / 100;
				// Map the difference into a weighting 0-1.
				// 1 means in phase or unknown; 0 means out of phase (more than kRange difference).
				const double kRange = 45 * irescale;
				kp = cl::sycl::clamp(1 - (kp / kRange), 0.0, 1.0);
				kn = cl::sycl::clamp(1 - (kn / kRange), 0.0, 1.0);

				double sc = 1.0;

				if ((kn > 0) || (kp > 0)) {
					// At least one of the next/previous lines has a good phase relationship.

					// If one of them is much better than the other, only use that one
					if (kn > (3 * kp)) kp = 0;
					else if (kp > (3 * kn)) kn = 0;

					sc = (2.0 / (kn + kp));
					if (sc < 1.0) sc = 1.0;
				} else {
					// Neither line has a good phase relationship.

					// But are they similar to each other? If so, we can use both of them!
					if ((cl::sycl::fabs(cl::sycl::fabs(previousLine[h]) - cl::sycl::fabs(nextLine[h])) - cl::sycl::fabs((nextLine[h] + previousLine[h]) * .2)) <= 0) {
						kn = kp = 1;
					}

					// Else kn = kp = 0, so we won't extract any chroma for this sample.
					// (Some NTSC decoders fall back to the 1D chroma in this situation.)
				}

				// Compute the weighted sum of differences, giving the 2D chroma value
				double tc1;
				tc1  = ((currentLine[h] - previousLine[h]) * kp * sc);
				tc1 += ((currentLine[h] - nextLine[h]) * kn * sc);
				tc1 /= 4;

				//save data to same buffer used to calcuate the data. saves needing another buffer.
				accessClpBuffer2D[lineNum][h] = tc1;

			});

		});

		myQueue.submit([&](cl::sycl::handler& cgh)
		{
			auto accessOutput = bufOutput.get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto accessClpBuffer2D = bufClpBuffer2D.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessInputDataOne = bufInputDataOne.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessInputDataTwo = bufInputDataTwo.get_access<cl::sycl::access::mode::read>(cgh);


			cgh.parallel_for<class splitIQ>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
			{
			
				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int h = tid.get_id(1) + videoParameters.activeVideoStart;

				unsigned short *temp;

				//access approperate input frame field
				if ((tid.get_id(0) % 2) == 0)
				{
					temp = accessInputDataOne.get_pointer();
				}
				else
				{
					temp = accessInputDataTwo.get_pointer();
				}
				
				const unsigned short *line = temp + ((lineNum / 2) * videoParameters.fieldWidth);
				bool linePhase = getLinePhase(lineNum, inputFieldOne.field.fieldPhaseID, inputFieldTwo.field.fieldPhaseID);


				double si = 0;
				double sq = 0;

				int phase = h % 4;

				int phaseBefore = (h - 1) % 4;

				double cavg = accessClpBuffer2D[lineNum][h];
				double cavgBefore = accessClpBuffer2D[lineNum][h - 1];


				if (linePhase)
				{
					cavg = -cavg;
					cavgBefore = -cavgBefore;
				}

				double comp = 0;


				switch (phase) {
					case 0: sq = cavg; 
					break;
					case 1: si = -cavg;
					break;
					case 2: sq = -cavg;
					break;
					case 3: si = cavg;
					break;
					default: break;
				}

				//here, calculating the phase of the previous pixel. 
				//prevents needing additional buffer to access it
				switch (phaseBefore) {
					case 0: sq = cavgBefore; 
					break;
					case 1: si = -cavgBefore; 
					break;
					case 2: sq = -cavgBefore; 
					break;
					case 3: si = cavgBefore; 
					break;
					default: break;
				}


				double i = si;
				double q = sq;
				

				switch (phase) {
					case 0: comp = -sq; break;
					case 1: comp = si; break;
					case 2: comp = sq; break;
					case 3: comp = -si; break;
					default: break;
				}

				if (!linePhase)
				{
					comp = -comp;
				}


				double y = line[h] - comp;


				//here is the start of converting the final data to RGB
				temp = accessOutput.get_pointer();
				
				unsigned short *pixelPointer = temp + (videoParameters.fieldWidth * 3 * lineNum) + (h * 3);

				double yBlackLevel = videoParameters.black16bIre;
				double yScale = 65535.0 / (videoParameters.white16bIre - videoParameters.black16bIre);
 
				// Compute I & Q scaling factor.
				// This is the same as for Y, i.e. when 7.5% setup is in use the chroma
				// scale is reduced proportionately.
				const double iqScale = yScale * chromaGain;

				if (whitePoint75) {
					// NTSC uses a 75% white point; so here we scale the result by
					// 25% (making 100 IRE 25% over the maximum allowed white point).
					// This doesn't affect the chroma scaling.
					yScale *= 125.0 / 100.0;
				}

				// Scale the Y to 0-65535 where 0 = blackIreLevel and 65535 = whiteIreLevel
				y = (y - yBlackLevel) * yScale;
				y = qBound(0.0, y, 65535.0);

				// Scale the I & Q components
				i *= iqScale;
				q *= iqScale;

				// Y'IQ to R'G'B' colour-space conversion.
				// Coefficients from Poynton, "Digital Video and HDTV" first edition, p367 eq 30.3.
				double r = y + (0.955986 * i) + (0.620825 * q);
				double g = y - (0.272013 * i) - (0.647204 * q);
				double b = y - (1.106740 * i) + (1.704230 * q);

				r = cl::sycl::clamp(r, 0.0, 65535.0);
				g = cl::sycl::clamp(g, 0.0, 65535.0);
				b = cl::sycl::clamp(b, 0.0, 65535.0);

				// Place the 16-bit RGB values in the output array
				pixelPointer[0] = static_cast<unsigned short>(r);
				pixelPointer[1] = static_cast<unsigned short>(g);
				pixelPointer[2] = static_cast<unsigned short>(b);




			});

//this kernel was combined with other kernels and now is no longer needed on its own.
/*
			cgh.parallel_for<class adjustY>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
			{

				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int h = tid.get_id(1) + videoParameters.activeVideoStart;



				bool linePhase = getLinePhase(lineNum, inputFieldOne.field.fieldPhaseID, inputFieldTwo.field.fieldPhaseID);

				double comp = 0;
				qint32 phase = h % 4;

				YIQ y = accessYIQ[lineNum][h];

				switch (phase) {
					case 0: comp = -y.q; break;
					case 1: comp = y.i; break;
					case 2: comp = y.q; break;
					case 3: comp = -y.i; break;
					default: break;
				}

				if (!linePhase) comp = -comp;
				y.y -= comp;

				accessYIQ[lineNum][h] = y;



			});
*/


//this is the previous yiqToRGBFrame code used before it was combined with other kernels
/*
			cgh.parallel_for<class yiqToRgbFrame>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
			{


				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int h = tid.get_id(1) + videoParameters.activeVideoStart;


				unsigned short *temp;

				temp = accessOutput.get_pointer();


				unsigned short *pixelPointer = temp + (videoParameters.fieldWidth * 3 * lineNum) + (h * 3);


				double yBlackLevel = videoParameters.black16bIre;
				double yScale = 65535.0 / (videoParameters.white16bIre - videoParameters.black16bIre);
 
				// Compute I & Q scaling factor.
				// This is the same as for Y, i.e. when 7.5% setup is in use the chroma
				// scale is reduced proportionately.
				const double iqScale = yScale * chromaGain;

				if (whitePoint75) {
					// NTSC uses a 75% white point; so here we scale the result by
					// 25% (making 100 IRE 25% over the maximum allowed white point).
					// This doesn't affect the chroma scaling.
					yScale *= 125.0 / 100.0;
				}




				double y = accessYIQ[lineNum][h].y;
				double i = accessYIQ[lineNum][h].i;
				double q = accessYIQ[lineNum][h].q;

				// Scale the Y to 0-65535 where 0 = blackIreLevel and 65535 = whiteIreLevel
				y = (y - yBlackLevel) * yScale;
				y = qBound(0.0, y, 65535.0);

				// Scale the I & Q components
				i *= iqScale;
				q *= iqScale;

				// Y'IQ to R'G'B' colour-space conversion.
				// Coefficients from Poynton, "Digital Video and HDTV" first edition, p367 eq 30.3.
				double r = y + (0.955986 * i) + (0.620825 * q);
				double g = y - (0.272013 * i) - (0.647204 * q);
				double b = y - (1.106740 * i) + (1.704230 * q);

				r = cl::sycl::clamp(r, 0.0, 65535.0);
				g = cl::sycl::clamp(g, 0.0, 65535.0);
				b = cl::sycl::clamp(b, 0.0, 65535.0);

				// Place the 16-bit RGB values in the output array
				pixelPointer[0] = static_cast<unsigned short>(r);
				pixelPointer[1] = static_cast<unsigned short>(g);
				pixelPointer[2] = static_cast<unsigned short>(b);




			});

*/


//kernel only used to move data from device to host side for testing purposes
/*

			cgh.parallel_for<class test>(cl::sycl::range<1>{1}, [=](cl::sycl::item<1> tid)
			{

				accessTestOutput[0] = accessClpBuffer1D[200][200];
				accessTestOutput[1] = accessClpBuffer2D[200][200];
				accessTestOutput[2] = videoParameters.firstActiveFrameLine;
				accessTestOutput[3] = videoParameters.activeVideoStart;
				accessTestOutput[4] = accessYIQ[200][200].y;
				accessTestOutput[5] = accessYIQ[200][200].i;
				accessTestOutput[6] = accessYIQ[200][200].q;

				accessTestOutput[7] = accessOutput[2000];
				accessTestOutput[8] = accessOutput[2001];

				//accessTestOutput[0] = accessInputData[200];
				//accessTestOutput[1] = accessInputDataOne[200];


			});

*/



		});

	}

//here is test code for verifying values
/*
	std::cout << "----------------------------- GPU DATA -----------------------" << std::endl;
	std::cout << "value: " << testData[0] << std::endl;
	std::cout << "2D value: " << testData[1] << std::endl;
	std::cout << "offset: " << testData[2] << std::endl;
	std::cout << "hoz offset: " << testData[3] << std::endl;
	std::cout << "current YIQ Y: " << testData[4] << std::endl;
	std::cout << "current YIQ I: " << testData[5] << std::endl;
	std::cout << "current YIQ Q: " << testData[6] << std::endl;
	std::cout << "colour value one: " << testData[7] << std::endl;
	std::cout << "colour value two: " << testData[8] << std::endl;
*/
}


