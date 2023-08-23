#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <ctime>
#include <cuda_runtime.h>


#define N  50000

#define THREADS_PER_BLOCK 256

// CUDA kernel - even comparisons
__global__ void even_swapper(int *X)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i % 2 == 0 && i < N-1){
        if(X[i+1] < X[i]){
            // switch in the x array
            int temp = X[i];
            X[i] = X[i+1];
            X[i+1] = temp;
        }
    }
}

// CUDA kernel - odd comparisons
__global__ void odd_swapper(int *X)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i % 2 != 0 && i < N-2){
        if(X[i+1] < X[i]){
            // switch in the x array
            int temp = X[i];
            X[i] = X[i+1];
            X[i+1] = temp;
        }
    }
}



int arraySortedOrNot(int arr[], int n)
{
    // Array has one or no element or the
    // rest are already checked and approved.
    if (n == 1 || n == 0)
        return 1;
 
    // Unsorted pair found (Equal values allowed)
    if (arr[n - 1] < arr[n - 2])
        return 0;
 
    // Last pair was sorted
    // Keep on checking
    return arraySortedOrNot(arr, n - 1);
}


int main()
{

    int *h_arr;
	
    h_arr= (int*)malloc( N * sizeof(int) );

 for(int i = 0; i < N; i++){
         h_arr[i] = (rand() % 1000) + 1;
    } 
    // allocate device memory 
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

		int numBlocks = (N+THREADS_PER_BLOCK-1) / (THREADS_PER_BLOCK);
		int numThreads= (THREADS_PER_BLOCK);
   
    clock_t par_begin = clock();
    for(int i = 0; i < N;  i++)
	{
        even_swapper<<<numBlocks,numThreads>>>(d_arr);
        odd_swapper<<<numBlocks,numThreads>>>(d_arr);
    }
    clock_t par_end = clock();

    cudaDeviceSynchronize(); 
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout <<"dataset size:"<< N << std::endl; 
    
    /* std::cout << "Parallel: " << "\n";
     for(int k = 0; k < N; k++)
          std::cout << h_output_array[k] << " ";
     		 std::cout<<"\n";*/
			 
	
	
	/*if(arraySortedOrNot(h_output_array,N))
	std::cout<<"parellel yes"<<"\n";*/


    
    

    double par_elapsed_secs = double(par_end - par_begin)/CLOCKS_PER_SEC;
    std::cout << "Elapsed Time for Parallel Bubblesort: "  <<par_elapsed_secs<<"\n";
    

    
    cudaFree(d_arr);
 

   
    free(h_arr);


    return 0;
}
