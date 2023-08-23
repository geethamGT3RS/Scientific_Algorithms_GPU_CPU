#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include<iostream>

#define N 4096
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

 void swap_seq(int *values, int i, int j)
{
  int temp = values[i];
  values[i] = values[j];
  values[j] = temp;
}


void seq_bitonic_sort(int *v) 
{
    for (int k = 2; k <= N; k = 2 * k) 
	{
        for (int j = k /2; j > 0; j = j /2) 
		{
            for (int i = 0; i < N; ++i) 
			{
                int ixj = i ^ j;
                if ((ixj) > i) 
				{
                    if ((i & k) == 0 and v[i] > v[ixj]) 
						swap_seq(v , i , ixj);
                    if ((i & k) != 0 and v[i] < v[ixj]) 
						swap_seq(v , i , ixj);
                }
            }
        }
    }
}

int arraySortedOrNot(int arr[], int n)
{
  
    if (n == 1 || n == 0)
        return 1;
 
   
    if (arr[n - 1] < arr[n - 2])
       {
			//printf("%d %d \n" ,  arr[n - 1] , arr[n - 2]);
			return 0;
			
       }
   
    return arraySortedOrNot(arr, n - 1);
}

int main()
{
  int *h_values, *d_values, *seq_values;

  h_values = (int *)malloc(N * sizeof(int));
  seq_values = (int *)malloc(N * sizeof(int));

  cudaMalloc((void **)&d_values, N * sizeof(int));

  for (int i = 0; i < N; i++)
  {
    h_values[i] = rand() % 1000;
    seq_values[i] = h_values[i];
  }
  
  cudaMemcpy(d_values, h_values, N * sizeof(int), cudaMemcpyHostToDevice);

  int numBlocks = (N + THREADS - 1) / THREADS;

 // parellel implementation
 
     
	
	  int j, k;
	   clock_t par_begin = clock();
	  for (k = 2; k <= N; k = 2 * k)
	  {
		for (j = k >> 1; j > 0; j = j >> 1)
		{
		  bitonic_sort_step<<<numBlocks, THREADS>>>(d_values, j, k);
		}
		cudaDeviceSynchronize(); // Add synchronization after each iteration
	  }
	  
	  
	  
	  cudaMemcpy(h_values, d_values, N * sizeof(int), cudaMemcpyDeviceToHost);

		clock_t par_end = clock();
	// sequential Implementation
	
	clock_t seq_begin = clock();
	seq_bitonic_sort(seq_values);
	clock_t seq_end = clock();
	
	
	
	if(arraySortedOrNot(seq_values,N))
	      std::cout<<"sequential yes"<<"\n";
	
		  if(arraySortedOrNot(h_values,N))
		  std::cout<<"parellel yes"<<"\n";

		  
		  double seq_elapsed_secs = double(seq_end - seq_begin)/CLOCKS_PER_SEC;
		std::cout << "Elapsed Time for seqential bitonic sort: "  <<seq_elapsed_secs<<"\n";
		
	
	    double par_elapsed_secs = double(par_end - par_begin)/CLOCKS_PER_SEC;
        std::cout << "Elapsed Time for Parallel bitonic sort: "  <<par_elapsed_secs<<"\n";
 
 
  /*for (int i = 0; i < N; i++)
    printf("%d ", seq_values[i] - h_values[i]);
	
	printf("\n");*/

  free(h_values);
  free(seq_values);
  cudaFree(d_values);

  return 0;
}
