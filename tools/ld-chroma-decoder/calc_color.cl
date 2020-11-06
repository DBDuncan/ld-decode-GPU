__kernel void vector_calc_color(__global const int *A,  __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    //double rY;
	
	printf("%f\n", A[0]);
 
	C[i] = 123;
}