#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>


#define DIMENSION_EMBEDDING 256
#define DIMENSION_HIDDEN 256  // usually the same as the embedding (model) dimension
#define NUM_HEADS 4
#define DIMENSION_KEYS (DIMENSION_HIDDEN / NUM_HEADS)  // 256 / 4 = 64
#define DIMENSION_VALUES (DIMENSION_HIDDEN / NUM_HEADS) // 256 / 4 = 64
#define NN_SIZE 1024  // 4 * DIMENSION_HIDDEN

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

void softmax_matrix(float input[], float output[], size_t N, size_t M) {
    // Apply softmax row-wise (each row sums to 1)
    for (size_t i = 0; i < N; i++) {
        // Find max value in the row for numerical stability
        float max_val = input[i * M];
        for (size_t j = 1; j < M; j++) {
            if (input[i * M + j] > max_val) {
                max_val = input[i * M + j];
            }
        }
        float sum = 0.0f;
        for (size_t j = 0; j < M; j++) {
            sum += exp(input[i * M + j] - max_val);
        }
        for (size_t j = 0; j < M; j++) {
            output[i * M + j] = exp(input[i * M + j] - max_val) / sum;
        }
    }
}

void transpose(float X[], int N, int M) {
    float *temp = malloc(N * M * sizeof(float));
    for (int i = 0; i < N * M; i++) { temp[i] = X[i]; }
    // Transpose from temp back to X
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            X[j * N + i] = temp[i * M + j];
        }
    }
    free(temp);
}

void matmul(float A[], float B[], int N, int D, int M, float result[]) { 
    // A is N x D, B is D x M, result is N x M
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result[i * M + j] = 0;
            for (int k = 0; k < D; k++) {
                result[i * M + j] += A[i * D + k] * B[k * M + j];
            }
        }
    }
}

// size is the input size
// Q, K, V have already been projected here to size x d_k
void attention(float Q[], float K[], float V[], int size) { 
    // Q * K^T
    transpose(K, size, DIMENSION_KEYS);
    float *QKT = malloc(size * size * sizeof(float));
    matmul(Q, K, size, DIMENSION_KEYS, size, QKT);
    // scale
    float factor = sqrt(DIMENSION_KEYS);
    for (int i = 0; i < size; i ++ ) { 
        for (int j = 0; j < size; j ++ ) { 
            QKT[i * size + j] /= factor;
        } 
    } 
    //softmax
    float *QKT_out = malloc(size * size * sizeof(float));
    softmax_matrix(QKT, QKT_out, size, size);
    float *result = malloc(size * DIMENSION_VALUES * sizeof(float));
    matmul(QKT_out, V, size, size, DIMENSION_VALUES, result);


    free(QKT);
    free(QKT_out);
    free(result);

} 

void masked_attention(float Q[], float K[], float V[], int size) { 
    // Q * K^T
    transpose(K, size, DIMENSION_KEYS);
    float *QKT = malloc(size * size * sizeof(float));
    matmul(Q, K, size, DIMENSION_KEYS, size, QKT);
    // scale
    float factor = sqrt(DIMENSION_KEYS);
    for (int i = 0; i < size; i ++ ) { 
        for (int j = 0; j < size; j ++ ) { 
            QKT[i * size + j] /= factor;
        } 
    } 
    
    // Apply causal mask: set upper triangular part (future tokens) to negative infinity
    // This ensures tokens can only attend to previous tokens and themselves
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            QKT[i * size + j] = -INFINITY;
        }
    }
    
    //softmax
    float *QKT_out = malloc(size * size * sizeof(float));
    softmax_matrix(QKT, QKT_out, size, size);
    float *result = malloc(size * DIMENSION_VALUES * sizeof(float));
    matmul(QKT_out, V, size, size, DIMENSION_VALUES, result);


    free(QKT);
    free(QKT_out);
    free(result);

} 


void nn_forward(float weights[], float activations[], float z[]) { 
    // alignment in memory matters here: 
    // weights index: layer i, to node k from node j
    // input will be considered the first of the activations
    // output will be considered the last of the activation

    // input layer is size 256
    // NN_SIZE 1024
    //     *   *
    //     *   *
    // *   *   *   *
    // *   *   *   *
    //     *   *
    //     *   *

    // first layer
    int layer = 1;
    for(size_t k = 0; k < NN_SIZE; k ++) {
        for(size_t j = 0; j < DIMENSION_VALUES; j ++) { 
            z[layer * 256 + k] += activations[j] * weights[k * DIMENSION_VALUES + j]; // 0 * 256 + j 
        } 
        activations[layer * 256 + k] = z[layer * 256 + k]; // some type of activation here
    } 

    // second layer
    layer = 2;
    for(size_t k = 0; k < NN_SIZE; k ++) {
        for(size_t j = 0; j < NN_SIZE; j ++) { 
            z[256 + 1024 + k] += activations[256 + NN_SIZE + j] * weights[256*1024 + k * NN_SIZE + j];
        } 
        activations[256 + 1024 + k] = z[256 + 1024 + k]; // some type of activation here
    } 

    // third layer
    layer = 3;
    for(size_t k = 0; k < DIMENSION_VALUES; k ++) {
        for(size_t j = 0; j < NN_SIZE; j ++) { 
            z[256 + 1024 * 2 + k] += activations[256 + 2 * NN_SIZE + j] * weights[256*1024 + 1024*1024 + k * NN_SIZE + j]; 
        } 
        activations[256 + 1024 * 2 + k] = z[256 + 1024 * 2 + k]; // some type of activation here
    } 



} 


int main() { 
    // TODO: load the byte pair encoding from the previous program.
    // TODO: accept input from the user in the command line and then run the forward part
    float* pe = malloc(10000 * DIMENSION_EMBEDDING * sizeof(float));
    compute_positional_encoding(pe, 10000, DIMENSION_EMBEDDING);
    // TODO: convert pe to matrix
    

    // dummy 100 token input
    int size = 100;

    // TODO: learned projection matrices W_q, W_k, W_v
    // dummy scaled attention, 
    float Q[size * DIMENSION_KEYS];
    float K[size * DIMENSION_KEYS];
    float V[size * DIMENSION_VALUES];
    attention(Q, K, V, size);

    printf("trained\n");
    // TODO: much much more

    free(pe);

} 
