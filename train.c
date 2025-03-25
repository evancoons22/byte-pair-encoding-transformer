#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

#define DIMENSION_EMBEDDING 256
#define DIMENSION_HIDDEN 256 // same as above?  hidden = embedding? 
#define DIMENSION_KEYS 64
#define DIMENSION_VALUES 64
#define DIMENSION_MODEL (Embedding size?)
#define NUM_HEADS 4
#define NN_SIZE 1024

#define LINEAR_1

typedef struct { 
    float* items; 
    size_t count;
    size_t capacity;
} Activations;

typedef struct { 
    float* items; 
    size_t count;
    size_t capacity;
} Weights;



void mha_forward();
void mmha_forward();
void nn_forward();
void linear_forward();
void add_and_norm();
void linear();

#include <math.h>
#include <stdlib.h>

// Compute positional encoding for a given maximum sequence length and model dimension.
// 'pe' should be a pre-allocated array of size max_seq_len * d_model.
void compute_positional_encoding(float *pe, int max_seq_len, int d_model) {
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            // calculate the "frequency" for this dimension
            float freq = 1.0 / pow(10000, (float)i / d_model);
            float angle = pos * freq;
            // even indices get sine
            pe[pos * d_model + i] = sin(angle);
            // Make sure we don't go out of bounds
            if (i + 1 < d_model) {
                pe[pos * d_model + i + 1] = cos(angle);
            }
        }
    }
}


void softmax(float input[], float output[], size_t size) {
    float sum = 0; 
    for (size_t i = 0; i < size; i++) { 
        sum += exp(input[i]); 
    } 
    for (size_t i = 0; i < size; i++) { 
        output[i] = exp(input[i]) / sum;
    } 
} 

int main() { 
    printf("test\n");

} 
