#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int gcd(int a, int b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

int modularExponentiation(int base, unsigned int exponent, int modulus) {
    int result = 1;
    base = base % modulus;
    while (exponent > 0) {
        if (exponent % 2 == 1)
            result = (result * base) % modulus;
        exponent = exponent >> 1;
        base = (base * base) % modulus;
    }
    return result;
}

void rsaEncrypt(int *message, int publicKey, int modulus, int *encryptedMessage, int messageSize) {
    for (int i = 0; i < messageSize; i++) {
        encryptedMessage[i] = modularExponentiation(message[i], publicKey, modulus);
    }
}

int main() {
    int p=3, q=7; // Prime numbers
    int publicKey, modulus; // RSA keys
    int messageSize=5; // Size of the message
    int *message, *encryptedMessage; // Arrays

    // Generate the public key and modulus
    int phi = (p - 1) * (q - 1);
    publicKey = 2; // Starting with a common value of public key (e)
    while (publicKey < phi) {
        if (gcd(publicKey, phi) == 1)
            break;
        publicKey++;
    }
    modulus = p * q;

    // Allocate memory for the arrays
    message = (int*)malloc(messageSize * sizeof(int));
    encryptedMessage = (int*)malloc(messageSize * sizeof(int));

    // Initialize the message array
    for (int i = 0; i < messageSize; i++) {
        scanf("%d",&message[i]);
    }

    // Perform RSA encryption
    rsaEncrypt(message, publicKey, modulus, encryptedMessage, messageSize);

    // Print the encrypted message
    printf("Encrypted message: ");
    for (int i = 0; i < messageSize; i++) {
        printf("%d ", encryptedMessage[i]);
    }
    printf("\n");

    // Free memory
    free(message);
    free(encryptedMessage);

    return 0;
}

