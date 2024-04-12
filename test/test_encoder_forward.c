#include <stdio.h>
#include <stdlib.h>

// define the vocab size and the dimension of each word
#define V 1000
#define C 64

// define the maximum sequence length and batch size
#define max_T 128
//#define B 1;


// test function
void encoder_forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C_) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* wte_ix =  wte + ix * C;
            float* wpe_t  = wpe + t * C;

            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}


int main() {
    // Randomly generated input sequence (array of integers) representing token indices in the vocabulary
    int inp[max_T];
    for (int i = 0; i < max_T; i++) {
        inp[i] = rand() % V; // Assume the vocabulary size is V
    }

    // Initialize the output array and the word token embedding and positional embedding arrays
    float out[max_T * C];
    float wte[V * C];
    float wpe[max_T * C];

    // Randomly initialize the word token embeddings and positional embeddings
    srand(time(NULL)); // Set the random seed
    for (int i = 0; i < V * C; i++) {
        wte[i] = ((float)rand()) / RAND_MAX * 2 - 1; // Random values in the range [-1, 1]
    }

    // Call the encoder_forward function 
    encoder_forward(out, inp, wte, wpe, 1, max_T, C);

    // Print the first 10 elements of the output array
    printf("Output after encoder_forward:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f \n", out[i]);
    }
    printf("\n");
    return 0;

}
