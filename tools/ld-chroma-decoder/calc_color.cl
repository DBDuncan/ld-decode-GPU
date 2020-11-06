__kernel void vector_calc_color(__global const int *A) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    //double rY;
	
	printf("%f\n", A[0]);
 
 
}