#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include<iostream>

#define N 512*512

#define THREADS 256

__device__ void swap(int *values, int i, int j)
{
  int temp = values[i];
  values[i] = values[j];
  values[j] = temp;
}

__global__ void bitonic_sort_step(int *dev_values, int j, int k)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int ixj = i ^ j;

  if (ixj > i && i < N)
  {
    if ((i & k) == 0)
    {
      if (dev_values[i] > dev_values[ixj])
        swap(dev_values, i, ixj);
    }

    if ((i & k) != 0)
    {
      if (dev_values[i] < dev_values[ixj])
        swap(dev_values, i, ixj);
    }
	__syncthreads();
  }
  
}

/*int arraySortedOrNot(int arr[], int n)
{
  
    if (n == 1 || n == 0)
        return 1;
 
   
    if (arr[n - 1] < arr[n - 2])
       {
			//printf("%d %d \n" ,  arr[n - 1] , arr[n - 2]);
			return 0;
			
       }
   
    return arraySortedOrNot(arr, n - 1);
}*/

int main()
{
	  int *h_values, *d_values;

	  h_values = (int *)malloc(N * sizeof(int));
	  

	  cudaMalloc((void **)&d_values, N * sizeof(int));

	  for (int i = 0; i < N; i++)
	  {
		h_values[i] = rand() % 1000;
		
	  }
	  clock_t par_begin = clock();
	  cudaMemcpy(d_values, h_values, N * sizeof(int), cudaMemcpyHostToDevice);

	  int numBlocks = (N + THREADS - 1) / THREADS;

	  int j, k;
	   
	  for (k = 2; k <= N; k = 2 * k)
	  {
		for (j = k >> 1; j > 0; j = j >> 1)
		 bitonic_sort_step<<<numBlocks, THREADS>>>(d_values, j, k);
		
		cudaDeviceSynchronize(); 
	  }
	  
	  cudaMemcpy(h_values, d_values, N * sizeof(int), cudaMemcpyDeviceToHost);

			clock_t par_end = clock();
	
		 /* if(arraySortedOrNot(h_values,N))
		  std::cout<<"parellel yes"<<"\n";*/

	    double par_elapsed_secs = double(par_end - par_begin)/CLOCKS_PER_SEC;
        std::cout << "Elapsed Time for Parallel bitonic sort: "  <<par_elapsed_secs<<"\n";
 
 
 
	  free(h_values);
	  
	  cudaFree(d_values);

  return 0;
}
