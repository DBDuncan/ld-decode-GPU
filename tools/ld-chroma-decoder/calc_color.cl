__kernel void vector_calc_color(__global const double *A, __global double *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    //double rY;
	
	//printf("%f\n", A[0]);
	double test = 454.444; 
	C[i] = A[200] * 2;
}
