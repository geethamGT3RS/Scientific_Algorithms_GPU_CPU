#include <iostream>
#include <cmath>
#include <vector>

int main() {
    
        int m = 4096; // x axis
        int n = 4096; // y axis

        double b = 0.5;
        double L = 1;

        double beta = L / b;

        double dx = 1.0 / (m - 1);
        double dy = beta / (n - 1);

        double w = 0.75;
        std::vector<std::vector<double>> theta(m, std::vector<double>(n, 0.0));

        std::vector<std::vector<double>> theta_guess(m, std::vector<double>(n, 3.0 / 7.0));

        for (int i = 0; i < m; i++) {
            theta[i][0] = 0.0;
            theta[i][n - 1] = 1.0;
            theta_guess[i][0] = 0.0;
            theta_guess[i][n - 1] = 1.0;
        }

        std::vector<std::vector<double>> theta_new_guess(m, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> error(m, std::vector<double>(n, 0.0));

        double sum = 0.0;
        int iterations = 0;
//        double tolerance = 0.1;

       // while (sum > tolerance || iterations <= 2) {
          while(iterations <=20){
            iterations++;
            sum = 0.0;

            for (int i = 1; i < m - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    theta[i][j] = (1.0 / (2.0 * (1.0 + std::pow(dx / dy, 2)))) *
                                  (theta_guess[i + 1][j] + theta_guess[i - 1][j] +
                                   std::pow(dx / dy, 2) * (theta_guess[i][j + 1] + theta_guess[i][j - 1]));
                }
            }

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    theta_new_guess[i][j] = theta_guess[i][j] + w * (theta[i][j] - theta_guess[i][j]);
                    error[i][j] = std::abs(theta[i][j] - theta_guess[i][j]);
                    sum += error[i][j];
                }
            }

            theta_guess = theta_new_guess;
        }

        std::vector<std::vector<double>> T(m, std::vector<double>(n, 0.0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T[i][j] = theta[i][j] * 70 + 30;
            }
        }

        // Print the result
       /* for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << T[i][j] << " ";
            }
            std::cout << std::endl;
        }*/
    

    return 0;
}
