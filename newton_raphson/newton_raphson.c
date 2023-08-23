#include <stdio.h>
#include <math.h>

float f(float x) {
    // Define the function for which to find the root
    return x * x - 4.0;
}

float fPrime(float x) {
    // Define the derivative of the function
    return 2.0 * x;
}

float newtonRaphson(float x0, int maxIterations) {
    float x = x0;
    for (int i = 0; i < maxIterations; ++i) {
        float delta = f(x) / fPrime(x);
        x -= delta;
        if (fabsf(delta) < 1e-6) {
            break;
        }
    }
    return x;
}

int main() {
    float result;
    float x0 = 2.0; // Initial guess
    int maxIterations = 100;

    result = newtonRaphson(x0, maxIterations);

    printf("Approximate root: %.6f\n", result);

    return 0;
}

