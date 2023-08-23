#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <algorithm>
#include <time.h>
#include <limits.h>
#include<iostream>
#include<math.h>

#define N 1000


int max(int A[] , int n)
{
	int max{0};
	
	for(int i =0 ; i<n ; i++)
	{
		if(A[i]>max)
		{
			max = A[i];
		}
	}
	
	return max;
}


void countingSort(int arr[], int exp) 
{
    int output[N];
    int count[10] = {0};

    
    for (int i = 0; i < N; i++) 
	{
        count[(arr[i] / exp) % 10]++;
    }

   
    for (int i = 1; i < 10; i++) 
	{
        count[i] += count[i - 1];
    }

    
    for (int i = N - 1; i >= 0; i--) 
	{
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

   
    for (int i = 0; i < N; i++) {
        arr[i] = output[i];
    }
}


void radixsort(int arr[]) 
{
    int maxVal = max(arr, N);

   
    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        countingSort(arr, exp);
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
	thrust::host_vector<int> h_values(N);
  
	for (int i = 0; i < N; i++)
    h_values[i] = rand() % 1000;
	
	thrust::device_vector<unsigned int> d_values = h_values;

	
	// parellel Implementation
	clock_t par_begin = clock();
    thrust::sort(d_values.begin(), d_values.end());
	clock_t par_end = clock();
	
	
    thrust::copy(d_values.begin(), d_values.end(), h_values.begin());

    bool sorted = thrust::is_sorted(h_values.begin(), h_values.end());

	// sequential Implementation
	
	int *seq_values = (int *) malloc(N * sizeof(int));
	
	for(int i =0 ; i< N ; i++)
		seq_values[i] = h_values[i];
	
	clock_t seq_begin = clock();
	radixsort(seq_values);
	clock_t seq_end = clock();
	
	if(arraySortedOrNot(seq_values,N))
	std::cout<<"sequential yes"<<"\n";
	
	if(sorted)
	std::cout<<"parellel yes"<<"\n";
	
	
		double seq_elapsed_secs = double(seq_end - seq_begin)/CLOCKS_PER_SEC;
		std::cout << "Elapsed Time for seqential radix sort: "  <<seq_elapsed_secs<<"\n";
		
	
	    double par_elapsed_secs = double(par_end - par_begin)/CLOCKS_PER_SEC;
        std::cout << "Elapsed Time for Parallel radix sort: "  <<par_elapsed_secs<<"\n";
	
	return 0;
}
