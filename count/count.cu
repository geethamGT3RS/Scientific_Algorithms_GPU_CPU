

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include<iostream>
#include<time.h>

#define N  50000000

#define THREADS 256

__global__ void count_sort(const int *d_A, int *d_count , int count_N , int min)
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int count_shared[1000  + 1];

   
    for (int i = threadIdx.x; i < count_N; i += blockDim.x)
        count_shared[i] = 0;

    __syncthreads();

  
    if (idx < N)
        atomicAdd(&count_shared[d_A[idx] - min], 1);

    __syncthreads();

    for (int i = threadIdx.x; i < count_N; i += blockDim.x)
        atomicAdd(&d_count[i], count_shared[i]);
}



void array_min_max(const int *array, long long size, int *min, int *max) {
    *min = array[0];
    *max = array[0];
    long long i = 0;

    
    for (i = 0; i < size; i++) 
	{
        if (array[i] < *min) {
        
            {
                if (array[i] < *min)
                    *min = array[i];
            }
        }
        else if (array[i] > *max) {
            {
                if (array[i] > *max)
                    *max = array[i];
            }
        }
    }
}

void countsort(int A[] , int max)
{
	
	
	int count[max+1] = {0};
	
	for(int i =0 ; i<N ; i++)
	{
	  count[A[i]] +=1;
	}
	
	for(int i =0 , j = 0; i<max+1;)
	{
	  if(count[i]!=0)
	  {
		  A[j] = i;
		  j++;
		  count[i]--;
	  }
	  if(count[i] == 0)
	  {
		  i++;
	  }
	
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

int main() {
    
    long long i = 0, j = 0, k = 0;
    int min = 0, max = 0;

    
    int *array = (int *)malloc(N * sizeof(int));
	
	int *d_A;
    
	srand(time(NULL));
		for (int i = 0; i<N; i++)
			array[i] = rand() % 1000;
   
        cudaMalloc((void **)&d_A, N * sizeof(int));
        cudaMemcpy(d_A, array, N * sizeof(int),cudaMemcpyHostToDevice);

	
		
		
    array_min_max(array, N, &min, &max);
    const int count_N = max - min + 1;

		int *count = (int *)malloc(sizeof(int) * count_N);
		int *d_count;
    
     cudaMalloc((void **)&d_count, count_N * sizeof(int));


		int numBlocks = (N+THREADS-1)/THREADS;
	
		// parellel implementation	
		
		clock_t par_begin = clock();
		
		count_sort<<<numBlocks, THREADS>>>(d_A, d_count , count_N , min);		
		cudaDeviceSynchronize();
  
		clock_t par_end = clock();
    
        cudaMemcpy(count, d_count, count_N * sizeof(int),cudaMemcpyDeviceToHost);


    /* Last section of the algorithm is not parallelizable. */
    for (i = min; i < max + 1; i++)
        for (j = 0; j < count[i - min]; j++)
            array[k++] = i;
			
	
	
	
		/*if(arraySortedOrNot(seq_array,N))
	      std::cout<<"sequential yes"<<"\n";
	
		  if(arraySortedOrNot(array,N))
		  std::cout<<"parellel yes"<<"\n";*/
	

		
		
		
		
	
	    double par_elapsed_secs = double(par_end - par_begin)/CLOCKS_PER_SEC;
        std::cout << "Elapsed Time for Parallel count sort: "  <<par_elapsed_secs<<"\n";

    
	cudaFree(d_A);
    cudaFree(d_count);
    free(array);
	
    free(count);

}





