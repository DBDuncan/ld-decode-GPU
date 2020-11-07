

double maxVal(double a, double b)
{


if (a > b)
{

return a;

}


return b;

}


double minVal(double a, double b)
{

if (a < b)
{

return a;

}

return b;


}

double boundVal(double minV, double value, double maxV)
{


return maxVal(minV, minVal(value, maxV));



}


__kernel void vector_calc_color(__global const double *sine, 
__global const double *cosine, 
__global const double *py, 
__global const double *qy,
__global const double *pu,
__global const double *qu,
__global const double *qv,
__global const double *pv,
__global const unsigned short *comp,
__global const unsigned short *In0,
const int prefilChroma,
 
const int black16bIre,
const double scaledContrast,
const double bp,
const double bq,
const double Vsw,
const double scaledSaturation,

__global double *output,
__global unsigned short *outputFinal
) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    //double rY;
	
	//printf("%f\n", A[0]);
	//double test = 454.444; 
	//output[i] = 125.453 + i;
	output[i] = 999;



	double rY;
	
	if (prefilChroma == 1)
	{
		rY = comp[i] - In0[i];

	}
	else
	{
		rY = comp[i] - ((py[i] * sine[i] + qy[i] * cosine[i]) * 2.0);
	}


	rY = boundVal(0.0, (rY - black16bIre) * scaledContrast, 65535.0);



	const double rU = -(pu[i] * bp + qu[i] * bq) * scaledSaturation;
	const double rV = Vsw * (qv[i] * bp - pv[i] * bq) * scaledSaturation;


	const double R = boundVal(0.0, rY + (1.139883 * rV), 65535.0);
	const double G = boundVal(0.0, rY + (-0.394642 * rU) + (-0.580622 * rV), 65535.0);
	const double B = boundVal(0.0, rY + (2.032062 * rU), 65535.0);



	const int pp = i * 3;

	outputFinal[pp + 0] = (unsigned short)R;
	outputFinal[pp + 1] = (unsigned short)G;
	outputFinal[pp + 2] = (unsigned short)B;


}
