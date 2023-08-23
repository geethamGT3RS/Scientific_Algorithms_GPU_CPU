#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include<time.h>

#define N  50000000

void array_min_max(const int *array, int size, int *max) {
    
    *max = array[0];
    for (int i = 0; i < size; i++) 
	{
         if (array[i] > *max) {
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

int main()
{
	int *seq_array = (int *)malloc(N * sizeof(int));
	int max;
	
	srand(time(NULL));
		for (int i = 0; i<N; i++)
			seq_array[i] = rand() % 1000;
		
		array_min_max(seq_array, N, &max);
		
		// seqential implementation
	clock_t seq_begin = clock();
	countsort(seq_array , max);
	clock_t seq_end = clock();
	
	
		double seq_elapsed_secs = double(seq_end - seq_begin)/CLOCKS_PER_SEC;
		std::cout << "Elapsed Time for seqential count sort: "  <<seq_elapsed_secs<<"\n";
		
		free(seq_array);
		
		return 0;
}
