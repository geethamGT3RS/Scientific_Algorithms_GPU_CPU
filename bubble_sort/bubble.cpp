
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <ctime>    

#define N  50000

int arraySortedOrNot(int arr[], int n)
{
    
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
	int *arr;
	
    arr= (int*)malloc( N * sizeof(int) );
	
	srand(time(0));

    for(int i = 0; i < N; i++){
         arr[i] = (rand() % 1000) + 1;
    }   
	
    clock_t seq_begin = clock();
	
	int flag{0}; // adaptive 
	for(int i =N-1 ; i>0 ; i--)
	{
		flag = 0;
		
		for(int j =0 ; j<i;j++)
		{
			if(arr[j+1] < arr[j])//comparing each adajacent element
			{
				int temp = arr[j+1];
				arr[j+1] = arr[j];
				arr[j] = temp;
				flag =1;
			}
		}
		if(flag ==0)
			break; 
	}
	
    clock_t seq_end = clock();
    std::cout<<"dataset size"<< N << std::endl;

   /* std::cout << "Sequential: " << "\n";
     for(int k = 0; k < N; k++)
         std::cout << arr[k] << " ";
		 
		 std::cout<<"\n";*/
	/*	 
	if(arraySortedOrNot(seq_h_output_array,N))
	std::cout<<"sequential yes"<<"\n";*/

	double seq_elapsed_secs = double(seq_end - seq_begin)/CLOCKS_PER_SEC;
	std::cout << "Elapsed Time for seqential Bubblesort: "  <<seq_elapsed_secs<<"\n";
		 
		 free(arr);
		 return 0;
}
