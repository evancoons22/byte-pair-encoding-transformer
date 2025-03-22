#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>


#define FORWARD_1 256
#define FORWARD_2 256
#define FORWARD_3 256
#define FORWARD_4 256

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
