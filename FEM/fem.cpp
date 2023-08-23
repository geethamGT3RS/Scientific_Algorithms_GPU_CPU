#include <iostream>
#include <vector>

int main()
{
    double phi = 4e8;
    double b = 0.01;
    double k = 373;
    double T0 = 30;
    double T1 = 100;

    int m = 500;
    int n = 500;

    double dx = 0.2;
    double dt = 0.02;

    int Nbi = 10;

    std::vector<std::vector<double>> theta(m, std::vector<double>(n, 0.0));

    // BC
			for(int i = 0 ; i < m ; i++)
			for(int j = 0 ; j <n ; j ++)
		{
			if (j == 0)
			{
				theta[i][j] = 0.0;
			}
			else if (i == m - 1)
        {
            theta[i][j] = 1.0;
        }
		}
clock_t seq_begin = clock();
    for (int j = 0; j < n - 1; j++)
    {
        for (int i = m - 2; i >= 1; i--)
        {
            theta[i][j + 1] = theta[i][j] + (dt / (dx * dx)) * (theta[i + 1][j] - 2 * theta[i][j] + theta[i - 1][j]) +
                              dt * (phi * (b * b)) / (k * (T1 - T0));
        }

        theta[0][j + 1] = theta[1][j + 1] * ((1 + (dx / 4) * Nbi) / (1 - (dx / 4) * Nbi));
    }
clock_t seq_end = clock();
	
    std::vector<std::vector<double>> T(m, std::vector<double>(n, 0.0));
    /*for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            T[i][j] = T0 + theta[i][j] * (T1 - T0);
			std::cout<<T[i][j]<<" ";
        }
		std::cout<<std::endl;
    }*/
		double seq_elapsed_secs = double(seq_end - seq_begin)/CLOCKS_PER_SEC;
		std::cout << "Elapsed Time for seqential FEM: "  <<seq_elapsed_secs<<"\n";
    return 0;
}
