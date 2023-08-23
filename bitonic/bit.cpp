#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include<iostream>
#define N 512*512

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
	std::cout << "dataset size"<< N << std::endl;
	int *seq_values = (int *)malloc(N * sizeof(int));
	 
	 for (int i = 0; i < N; i++)
	 { 
       seq_values[i] = rand() % 1000;
     }
	 
	 clock_t seq_begin = clock();
	seq_bitonic_sort(seq_values);
	clock_t seq_end = clock();
	
	
	/*if(arraySortedOrNot(seq_values,N))
	      std::cout<<"sequential yes"<<"\n";*/
	  
	  
	  double seq_elapsed_secs = double(seq_end - seq_begin)/CLOCKS_PER_SEC;
		std::cout << "Elapsed Time for seqential bitonic sort: "  <<seq_elapsed_secs<<"\n";
		
		free(seq_values);
		return 0;
}
