#include <stdio.h>
#include <stdlib.h>

// define the vocab size and the dimension of each word
#define V 1000
// #define C 64

// define the maximum sequence length and batch size
#define max_T 128
//#define B 1;


// test function
// wte: word token embedding, of shape (vocab_size, C); wpe: word position embedding, of shape(T, C)
// B: batch size; T: number of words; C: dimension of each word
// inp of shape (B, T, C); out of shape(B, T, C)
void encoder_forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C) {
    for (int b = 0; b < B; b++) { // for each batch, which contains a word sequence
        for (int t = 0; t < T; t++) { // for each sequence, which contains T words
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* wte_ix =  wte + ix * C; // 定位到对应的词嵌入向量
            float* wpe_t  = wpe + t * C; // Position the positional embedding corresponding to the position

            for (int i = 0; i < C; i++) { // for each word, we add token embedding and position embedding
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}


int main() {
    int C = 64;
    int B = 1;

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
