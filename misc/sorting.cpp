#include<iostream>
#include<math.h>
using namespace std;

class Node{
	public:
	int data;
	Node *next;
};


class Sort{
	
	public:
	void bubble(int A[],int n);
	void insertion(int A[] , int n);
	void selection(int A[] , int n);
	void merge(int A[] , int l , int h);
	void Imergesort(int A[] , int n);
	void Rmergesort(int A[] , int l , int h);
	void countsort(int A[] , int n);
	void Binsort(int A[] , int n);
	void Radixsort(int A[] , int n );
};

void insert( Node **p , int data)
{
  
  Node *q ,*r; 
   
   if(*p == NULL)
   {
	   q= new Node;
       q->data = data;
       q->next = NULL;
	   *p = q;
   }
   
   else
   {
	   q = *p;
	   while(q!=NULL)
	   {
           r = q;		  
		  q = q->next;
	   }
	   q = new Node;
	   q->data = data;
	   q->next = NULL;
	   r->next = q;

   }  
	   
}



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
	

void swap( int *a , int *b)
{
	int *temp;
	temp = a;
	a = b;
	b = temp;
}

void Sort :: bubble(int A[],int n)
{
	int flag{0}; // adaptive 
	for(int i =n-1 ; i>0 ; i--)
	{
		flag = 0;
		for(int j =0 ; j<i;j++)
		{
			if(A[j]>A[j+1])
			{
				swap(A[j],A[j+1]);//comparing each adajacent element
				flag =1;
			}
			
		}
		if(flag ==0)
			break; 
	}
	
	
	for(int i = 0 ; i<n ; i++)
	{
	cout<<A[i]<<" ";
	}
}


void Sort :: insertion(int A[] , int n)
{
	for(int i = 1; i<n ; i++)
	{
		for(int j =i-1; j>=0 ;j--)
		{
             if(A[j]>A[i])
			 {
				 swap(A[j],A[i]);//finding its position to insert and shifiting all 
				 i--;    // all other elements 
			 }
	    }
	}
	
	/*for(int i = 0 ; i<n ; i++)
	{
	cout<<A[i]<<" ";
	}*/
}

void Sort :: selection(int A[] , int n)
{
	for(int i = 0 ; i<n-1 ; i++)
    {
		int k = i;
		for(int j = i+1 ; j<n ; j++)
		{
			if(A[k]>A[j])
				k =j;
		}
		swap(A[k],A[i]);
	}
	
		for(int i = 0 ; i<n ; i++)
	{
	cout<<A[i]<<" ";
	}
}


int partition( int A[] , int i , int j)
{
	int pivot = A[i];
	int temp = i;
	
	while(i<j)
	{
		while(A[i]<=pivot)
			i++;
		while(A[j]>pivot)
			j--;
		
		if(i<j)
			swap(A[i],A[j]);
	}
	
	swap(A[temp] , A[j]);
	return j;
}

void quick(int A[] , int i , int j)
{
		int k{0};
		
	if(i<j)
	{
		k = partition(A,i,j);
		quick(A,i,k);
		quick(A,k+1,j);
	}
}

void Sort :: merge(int A[] , int l , int h)
{
	int mid = (h+l)/2;
	
	int  i{l} , j{mid+1},k{l};
	
	int B[h+1];
	
	while(i<=mid && j<=h)
	{
		if(A[i] <A[j])
			B[k++] = A[i++];
		else
			B[k++] = A[j++];
	}
	
	while(i<=mid)
		B[k++] = A[i++];
	while(j<=h)
		B[k++] = A[j++];
	
	for(i = l ; i<=h ; i++)
		A[i] = B[i];
}

void Sort :: Imergesort(int A[] , int n)
{
	int p{0};
	
	for(p = 2 ; p<=n ; p*=2)
	{
		for(int i = 0 ; i+p-1<n ; i+=p)
		{
			int l = i;
			int h = i+p-1;
			merge(A,l,h);
		}
	}
	
	 if(p/2 < n)
	 merge(A,0,n-1);
}

void Sort :: Rmergesort(int A[] , int l , int h)
{
	if(l<h)
	{
		int mid = (l+h)/2;
		Rmergesort(A,l,mid);
		Rmergesort(A,mid+1,h);
		merge(A,l,h);
	}
}

void Sort :: countsort(int A[] , int n)
{
	int max_element = max(A,n);
	
	int count[max_element+1] = {0};
	
	for(int i =0 ; i<n ; i++)
	{
	  count[A[i]] +=1;
	}
	
	for(int i =0 , j = 0; i<max_element+1;)
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

void Sort :: Binsort(int A[] , int n)
{
	int max_element = max(A,n);
	
	Node *bin[max_element+1]= {NULL};
	
	
	for(int i =0 ; i<n ; i++)
	{
	   insert(&bin[A[i]] , A[i]);
	}
		
	
	for(int i =0 , j = 0; i<max_element+1;)
	{
	  if(bin[i]!=NULL)
	  {
		  A[j] = bin[i]->data;
		  j++;
		  //del(&bin[i]);
		  bin[i] = bin[i]->next;
	  }
	  if(bin[i] == NULL)
	  {
		  i++;
	  }
	
	}
}

void Sort :: Radixsort(int A[] , int n )
{

	int max_element = max(A,n);
	int key = 1;
	
 for(int k = 0 ; pow(k,10)<max_element ; k++)
 {
	Node *radix[10] = {NULL};
	
	for(int l =0 ; l<k ; l++)
	{
		key = key*10;
	}
	
	for(int i =0 ; i<n ; i++)
	{
	   insert(&radix[(A[i]/key)%10] , A[i]);
	}
	
	for(int i =0 , j = 0; i<10;)
	{
	  if(radix[i]!=NULL)
	  {
		  A[j] = radix[i]->data;
		  j++;
		  //del(&bin[i]);
		  radix[i] = radix[i]->next;
	  }
	  if(radix[i] == NULL)
	  {
		  i++;
	  }
	
	}
 }
 
}

void  ShellSort(int A[] , int n)
{
	int gap = n/2;
	while(gap>0)
	{
        for(int i = 0 , j = i+gap ; j<n; i++ , j++)
		{
				if(A[i]>A[j])
				{
					swap(A[i],A[j]);
					
					int k = i-gap;
					int l = i;
					while(k>=0)
					{
						if(A[k]>A[l])
						swap(A[k],A[l]);	
						
						l = k;
						k-=gap;
					}
		  
	             }
		}
		 gap = gap/2;
    }
}
	
int main()
{
	Sort s;
	int A[1024];
	
	 for (int i = 0; i<1024; ++i)
	{
      A[i] = rand() % 100;
    }
	
	//s.bubble(A,5);
	s.insertion(A,1024);
	//s.selection(A,5);
	//s.Radixsort(A,10);
	//quick(A, 1 , 2);
	//ShellSort(A,11);
	
	for(int i = 0 ; i<1024 ; i++)
	{
	cout<<A[i]<<" ";
	}
	
	
	return 0;
}
	