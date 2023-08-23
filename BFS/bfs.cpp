#include<iostream>
using namespace std;

#define N 25000
int** random_adjacency_matrix(int n)
{
    int** matrix = new int*[n];
    for(int i=0; i<n; i++){
        matrix[i] = new int[n];
    }

  
    srand(time(NULL));
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            matrix[i][j] = rand() % 2;
        }
    }

    
    for(int i=0; i<n; i++){
        matrix[i][i] = 0;
    }

    
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            matrix[j][i] = matrix[i][j];
        }
    }

    return matrix;
}
class Node
{
  public:
	int data;
	Node *next;
};

class queue
{
	private:
	Node *top;
	
	public:
	queue(){top = NULL;}
	
	void enqueue(int key);
	int dequeue();
	bool isempty();
};
void queue :: enqueue(int key)
{
	Node *p;
	
	
	if(top == NULL)
	{
		p = new Node;
		top = p;
		top->data = key;
		top->next = NULL;
	}
	else
	{
		 Node *q;
		 p = top;
		while(p!=NULL)
		{
			q = p;
			p= p->next;
		}
		
		p = new Node;
		p->data = key;
		q->next = p;
	}
}
	
int queue :: dequeue()
{
   int key = top->data;
   top = top->next;
  return key;
}

bool queue :: isempty()
{
	if(top == NULL)
		return true;
	
	return false;
}
void BFS(int **A , int index, int n )
{
	int u{0};
	//cout<<index<<" ";
	
	queue q;
	
	q.enqueue(index);
	
    int visited[n] ={0};
	visited[index] = 1;
    
     
	while(!q.isempty())
 {
	 u = q.dequeue();

     for(int v = 1 ; v<n ; v++)
	 {
        if(A[u][v] == 1 && visited[v] == 0)
		{
               //cout<<v<< " ";
                visited[v] = 1;
                 q.enqueue(v);
		}
	 }
 }
}


int main()
{
	std::cout << "data size" << N << endl;
	 int **A = random_adjacency_matrix(N);
       
	   clock_t seq_begin = clock();
	   BFS(A,0,N);
	   clock_t seq_end = clock();

	   
	   double seq_elapsed_secs = double(seq_end - seq_begin)/CLOCKS_PER_SEC;
		std::cout << "Elapsed Time for seqential BFS: "  <<seq_elapsed_secs<<"\n";
	   
	   for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete[] A;
	   
	   return 0;
}
