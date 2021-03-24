
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


void decodeFrameGPU(const SourceField &inputFieldOne, const SourceField &inputFieldTwo, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, QVector<quint16> rawbuffer, double yNRLevel, double irescale, double chromaGain, bool whitePoint75)
{



	//int a = 1;

	//std::cout << "call test" << std::endl;
	std::vector<double> testData(100);

	int frameHeight = ((videoParameters.fieldHeight * 2) - 1);

	outputFrame.resize(videoParameters.fieldWidth * frameHeight * 3);


	//std::cout << "Frame Height: " << frameHeight << std::endl;

	{

		cl::sycl::queue myQueue;


		cl::sycl::buffer<double> bufTestOutput(testData.data(), testData.size());


		//cl::sycl::buffer<unsigned short> bufInputData(rawbuffer.data(), rawbuffer.size());

		cl::sycl::buffer<unsigned short> bufInputDataOne(inputFieldOne.data.data(), inputFieldOne.data.size());
		cl::sycl::buffer<unsigned short> bufInputDataTwo(inputFieldTwo.data.data(), inputFieldTwo.data.size());



		cl::sycl::buffer<double, 2> bufClpBuffer1D{cl::sycl::range<2>(525, 910)};//was 525
		cl::sycl::buffer<double, 2> bufClpBuffer2D{cl::sycl::range<2>(525, 910)};

		cl::sycl::buffer<YIQ, 2> bufYIQ{cl::sycl::range<2>(525, 910)};


		cl::sycl::buffer<unsigned short> bufOutput{outputFrame.data(), cl::sycl::range<1>(videoParameters.fieldWidth * frameHeight * 3)};


        myQueue.submit([&](cl::sycl::handler& cgh)
		{


			auto accessTestOutput = bufTestOutput.get_access<cl::sycl::access::mode::write>(cgh);
	

			//auto accessInputData = bufInputData.get_access<cl::sycl::access::mode::read>(cgh);


			auto accessInputDataOne = bufInputDataOne.get_access<cl::sycl::access::mode::read>(cgh);
			auto accessInputDataTwo = bufInputDataTwo.get_access<cl::sycl::access::mode::read>(cgh);



			auto accessClpBuffer1D = bufClpBuffer1D.get_access<cl::sycl::access::mode::discard_read_write>(cgh);

			auto accessClpBuffer2D = bufClpBuffer2D.get_access<cl::sycl::access::mode::discard_read_write>(cgh);


			auto accessYIQ = bufYIQ.get_access<cl::sycl::access::mode::discard_read_write>(cgh);


			auto accessOutput = bufOutput.get_access<cl::sycl::access::mode::discard_read_write>(cgh);//change later to discard_write


			const size_t lines = videoParameters.lastActiveFrameLine - videoParameters.firstActiveFrameLine;

			const size_t width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

			//std::cout << "lines: " << lines << std::endl;

			cgh.parallel_for<class split1D>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
            {

				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int col = tid.get_id(1) + videoParameters.activeVideoStart;


				int lineNumTwo = tid.get_id(0) + (videoParameters.firstActiveFrameLine / 2);

				//int finalLineNum;

				unsigned short *temp;



				if ((tid.get_id(0) % 2) == 0)
				{
					temp = accessInputDataOne.get_pointer();
					
				}
				else
				{
                   	temp = accessInputDataTwo.get_pointer();

				}



				//temp = accessInputData.get_pointer();

				//dont forget divide by two here!!!
				const unsigned short *line = temp + ((lineNum / 2) * videoParameters.fieldWidth);

				//was twos where ones
				double tc1 = (line[col] - ((line[col - 2] + line[col + 2]) / 2.0)) / 2.0;

				accessClpBuffer1D[lineNum][col] = tc1;
			

			});


			cgh.parallel_for<class split2D>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
            {

				static constexpr double blackLine[910] = {0};

				int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
				int h = tid.get_id(1) + videoParameters.activeVideoStart;




				double *temp = accessClpBuffer1D.get_pointer();

				const double *previousLine = blackLine;
				if (lineNum - 2 >= videoParameters.firstActiveFrameLine) 
				{
					//double *temp = accessClpBuffer1D.get_pointer();
					previousLine = temp + ((lineNum - 2) * videoParameters.fieldWidth); //accessClpBuffer1D.pixel[lineNumber - 2];
				}
				const double *currentLine = temp + (lineNum * videoParameters.fieldWidth);//accessClpBuffer.pixel[lineNumber];
				const double *nextLine = blackLine;
				if (lineNum + 2 < videoParameters.lastActiveFrameLine) 
				{
					nextLine = temp + ((lineNum+ 2) * videoParameters.fieldWidth);//clpbuffer[0].pixel[lineNumber + 2];
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

            	//clpbuffer[1].pixel[lineNumber][h] = tc1;
				accessClpBuffer2D[lineNum][h] = tc1;

			});




			cgh.parallel_for<class splitIQ>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
            {
			
				
                int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
                int h = tid.get_id(1) + videoParameters.activeVideoStart;


				unsigned short *temp;


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


				//double si = 0;
				//double sq = 0;

				//for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {

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

				//double si = 0;
				//double sq = 0;

             	switch (phase) {
                	case 0: sq = cavg; break;
                	case 1: si = -cavg; break;
                	case 2: sq = -cavg; break;
                	case 3: si = cavg; break;
                	default: break;
             	}

                 switch (phaseBefore) {
                    case 0: sq = cavgBefore; break;
                    case 1: si = -cavgBefore; break;
                    case 2: sq = -cavgBefore; break;
                    case 3: si = cavgBefore; break;
                	default: break;
                }


             	accessYIQ[lineNum][h].y = line[h];
             	accessYIQ[lineNum][h].i = si;
             	accessYIQ[lineNum][h].q = sq;

				//}

			});



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


/*
			cgh.parallel_for<class doYNR>(cl::sycl::range<2>{lines, width}, [=](cl::sycl::item<2> tid)
			{


             	int lineNum = tid.get_id(0) + videoParameters.firstActiveFrameLine;
            	int h = tid.get_id(1) + videoParameters.activeVideoStart;


		    	// High-pass filter for Y
				auto yFilter(f_nr);

     			// nr_y is the coring level
     			double nr_y = yNRLevel * irescale;

				//yFilter.feed(23344);	

				double a = yFilter.feed(accessYIQ[lineNum][h + 12].y);



				if (cl::sycl::fabs(a) > nr_y) {
                	a = (a > 0) ? nr_y : -nr_y;
             	}


				//accessYIQ[lineNum][h].y = a;//yFilter.a[1];//accessYIQ[lineNum][h].y;







			});
*/

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





		});

	}
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

void decodeFrameGPU(RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters)
{


}


