#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define DIMENSION_EMBEDDING 256
#define DIMENSION_HIDDEN 256  // usually the same as the embedding (model) dimension
#define NUM_HEADS 4
#define DIMENSION_KEYS (DIMENSION_HIDDEN / NUM_HEADS)  // 256 / 4 = 64
#define DIMENSION_VALUES (DIMENSION_HIDDEN / NUM_HEADS) // 256 / 4 = 64
#define NN_SIZE 1024  // 4 * DIMENSION_HIDDEN
#define VOCAB_SIZE 6400

#define DA_INIT_CAP 256
#define da_append(da, item)                                                          \
    do {                                                                                 \
        if ((da)->count >= (da)->capacity) {                                             \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                                \
                                                                                         \
        (da)->items[(da)->count++] = (item);                                             \
    } while (0)

typedef struct { 
    uint32_t* items;
    size_t count;
    size_t capacity;
} Tokens; 

typedef struct { 
    uint32_t value;
    uint32_t l;
    uint32_t r;
} KV;

typedef struct { 
    KV* items;
    size_t count;
    size_t capacity;
} Map;

Map load_map(const char *filename) {
    Map map = {0};
    FILE *f = fopen(filename, "rb");
    if (!f) return map;

    fread(&map.count, sizeof(size_t), 1, f);
    map.items = malloc(map.count * sizeof(KV));  // malloc is fine even if it's fixed
    fread(map.items, sizeof(KV), map.count, f);

    map.capacity = map.count;  // not needed, but harmless

    fclose(f);
    return map;
}

// Compute positional encoding for a given maximum sequence length and model dimension.
// 'pe' should be a pre-allocated array of seq_len max_seq_len * d_model.
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


void softmax(float input[], float output[], size_t seq_len) {
    float sum = 0; 
    for (size_t i = 0; i < seq_len; i++) { 
        sum += exp(input[i]); 
    } 
    for (size_t i = 0; i < seq_len; i++) { 
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

// seq_len is the input seq_len
// Q, K, V have already been projected here to seq_len x d_k
void attention(float Q[], float K[], float V[], int seq_len, float* result) { 
    // Q * K^T
    // copy K into new memory to avoid corruption due to transpose
    float K_temp[seq_len*DIMENSION_KEYS];
    for (int i = 0; i < seq_len*DIMENSION_KEYS; i++) { K_temp[i] = K[i]; }

    transpose(K_temp, seq_len, DIMENSION_KEYS);
    float *QKT = malloc(seq_len * seq_len * sizeof(float));
    matmul(Q, K_temp, seq_len, DIMENSION_KEYS, seq_len, QKT);
    // scale
    float factor = sqrt(DIMENSION_KEYS);
    for (int i = 0; i < seq_len; i ++ ) { 
        for (int j = 0; j < seq_len; j ++ ) { 
            QKT[i * seq_len + j] /= factor;
        } 
    } 
    //softmax
    float *QKT_out = malloc(seq_len * seq_len * sizeof(float));
    softmax_matrix(QKT, QKT_out, seq_len, seq_len);
    matmul(QKT_out, V, seq_len, seq_len, DIMENSION_VALUES, result);

    free(QKT);
    free(QKT_out);
} 

void masked_attention(float Q[], float K[], float V[], int seq_len, float* result) { 
    // Q * K^T
    transpose(K, seq_len, DIMENSION_KEYS);
    float *QKT = malloc(seq_len * seq_len * sizeof(float));
    matmul(Q, K, seq_len, DIMENSION_KEYS, seq_len, QKT);
    // scale
    float factor = sqrt(DIMENSION_KEYS);
    for (int i = 0; i < seq_len; i ++ ) { 
        for (int j = 0; j < seq_len; j ++ ) { 
            QKT[i * seq_len + j] /= factor;
        } 
    } 
    
    // Apply causal mask: set upper triangular part (future tokens) to negative infinity
    // This ensures tokens can only attend to previous tokens and themselves
    for (int i = 0; i < seq_len; i++) {
        for (int j = i + 1; j < seq_len; j++) {
            QKT[i * seq_len + j] = -INFINITY;
        }
    }
    
    //softmax
    float *QKT_out = malloc(seq_len * seq_len * sizeof(float));
    softmax_matrix(QKT, QKT_out, seq_len, seq_len);
    matmul(QKT_out, V, seq_len, seq_len, DIMENSION_VALUES, result);

    free(QKT);
    free(QKT_out);
} 

//rows of Q, K, V are the embeddings
//void attention_projection (const float* input, const float* W_Q, const float* W_K, const float* W_V, float* Q, float* K, float* V, int d_model) { 
//
//}

void add_and_norm(float input[], float output[], float result[], int seq_len, int dim) { 
    // input: matrix of shape [seq_len × dim] - original input to the sublayer
    // output: matrix of shape [seq_len × dim] - output from the sublayer
    // result: matrix of shape [seq_len × dim] - where to store the final result
    // seq_len: number of tokens in the sequence
    // dim: dimension of each token's embedding (DIMENSION_HIDDEN/DIMENSION_EMBEDDING)
    
    // First, add the residual connection (skip connection)
    // Add the original input to the sublayer output for each position and feature
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < dim; j++) {
            result[i * dim + j] = input[i * dim + j] + output[i * dim + j];
        }
    }
    
    // Layer normalization - applied separately for each token
    // For each token (row in the matrix)
    for (int i = 0; i < seq_len; i++) {
        // 1. Calculate mean across the feature dimension for this token
        float mean = 0.0f;
        for (int j = 0; j < dim; j++) {
            mean += result[i * dim + j];
        }
        mean /= dim;
        
        // 2. Calculate variance across the feature dimension for this token
        float variance = 0.0f;
        for (int j = 0; j < dim; j++) {
            float diff = result[i * dim + j] - mean;
            variance += diff * diff;
        }
        variance /= dim;
        
        // 3. Normalize each feature of this token (add small epsilon for numerical stability)
        float epsilon = 1e-5f;
        for (int j = 0; j < dim; j++) {
            result[i * dim + j] = (result[i * dim + j] - mean) / sqrtf(variance + epsilon);
        }
        
        // 4. Apply scale and shift parameters (gamma and beta)
        // In a real implementation, these would be learned parameters per feature
        // For simplicity, we'll use gamma=1 and beta=0 (identity transformation)
        // If you want to make these learnable, you would need to add them as parameters
        // float gamma[dim]; // Learned per feature
        // float beta[dim];  // Learned per feature
        // for (int j = 0; j < dim; j++) {
        //     result[i * dim + j] = gamma[j] * result[i * dim + j] + beta[j];
        // }
    }
}

void nn_forward(float weights[], float activations[], float z[]) { 
    // alignment in memory matters here: 
    // weights index: layer i, to node k from node j
    // input will be considered the first of the activations
    // output will be considered the last of the activation

    // input layer is seq_len 256
    // NN_SIZE 1024
    //     *
    //     *
    // *   *   *
    // *   *   *
    //     *
    //     *

    // first layer
   // int layer = 1;
   // for(size_t k = 0; k < NN_SIZE; k ++) {
   //     for(size_t j = 0; j < DIMENSION_VALUES; j ++) { 
   //         z[layer * 256 + k] += activations[j] * weights[k * DIMENSION_VALUES + j]; // 0 * 256 + j 
   //     } 
   //     activations[layer * 256 + k] = z[layer * 256 + k]; // some type of activation here
   // } 

    // second layer
   // layer = 3;
   // for(size_t k = 0; k < DIMENSION_VALUES; k ++) {
   //     for(size_t j = 0; j < NN_SIZE; j ++) { 
   //         z[256 + 1024 * 2 + k] += activations[256 + 2 * NN_SIZE + j] * weights[256*1024 + 1024*1024 + k * NN_SIZE + j]; 
   //     } 
   //     activations[256 + 1024 * 2 + k] = z[256 + 1024 * 2 + k]; // some type of activation here
   // } 

    
    // ------------------------------
    // something like this will replace this nn_forward function
    // ------------------------------
    
    // 1024 is the inner size of the nn
    // this replaces the above layer. 
    float* weights_head = weights;
    float* activations_head = activations;
    float* z_head = z;
    matmul(weights_head, activations_head, 1024, 256, 1, z_head);

    // need relu activations
    weights_head = weights + 1024*256;
    activations_head = activations + 1024;
    z_head = z + 1024;
    matmul(weights, activations, 256, 1024, 1, z_head);

} 

void convert_to_embeddings(float *E, float* embeddings, float* input, int seq_len){ 
    // Create learned embedding matrix E
    // Initialize E with random values (in a real scenario, this would be loaded from a trained model)
    for (int i = 0; i < VOCAB_SIZE * DIMENSION_EMBEDDING; i++) {
        E[i] = ((float)rand() / RAND_MAX) * 0.1f;  // Small random values
    }
    // Allocate memory for embeddings
    
    // Convert tokens to embeddings (skipping actual one-hot encoding for efficiency)
    // In practice, this is equivalent to selecting the corresponding row from E for each token
    for (int i = 0; i < seq_len; i++) {
        int token_id = (int)input[i];
        if (token_id >= VOCAB_SIZE) token_id = 0;  // Handle out-of-vocabulary tokens
        
        // Copy the embedding for this token (equivalent to multiplying one-hot by E)
        for (int j = 0; j < DIMENSION_EMBEDDING; j++) {
            embeddings[i * DIMENSION_EMBEDDING + j] = E[token_id * DIMENSION_EMBEDDING + j];
        }
    }
} 

void init_rand(float *X, int seq_len) { 
    for (int i = 0; i < seq_len; i++) {
        X[i] = ((float)rand() / RAND_MAX) * 0.1f;  // Small random values
    }
} 

void ffn_block(float* input, float* weights1, float* weights2, float* output, int seq_len) {
    // Process each token in the sequence independently
    // input is [seq_len x DIMENSION_HIDDEN]
    // weights1 is [DIMENSION_HIDDEN x NN_SIZE]
    // weights2 is [NN_SIZE x DIMENSION_HIDDEN]
    
    // Allocate memory for one token's intermediate values
    float* token_intermediate = malloc(NN_SIZE * sizeof(float));
    float* token_output = malloc(DIMENSION_HIDDEN * sizeof(float));
    
    // Process each token independently
    for (int pos = 0; pos < seq_len; pos++) {
        // Get the current token's input vector
        float* token_input = &input[pos * DIMENSION_HIDDEN];
        
        // First layer: token_input -> token_intermediate
        matmul(token_input, weights1, 1, DIMENSION_HIDDEN, NN_SIZE, token_intermediate);
        
        // Apply ReLU activation
        for (int i = 0; i < NN_SIZE; i++) {
            token_intermediate[i] = token_intermediate[i] > 0 ? token_intermediate[i] : 0;
        }
        
        // Second layer: token_intermediate -> token_output
        matmul(token_intermediate, weights2, 1, NN_SIZE, DIMENSION_HIDDEN, token_output);
        
        // Store the result in the output array
        for (int i = 0; i < DIMENSION_HIDDEN; i++) {
            output[pos * DIMENSION_HIDDEN + i] = token_output[i];
        }
    }
    
    // Add & norm layer for the entire sequence
    add_and_norm(input, output, output, seq_len, DIMENSION_HIDDEN);

    // Clean up
    free(token_intermediate);
    free(token_output);
}

// Helper function to check if a token is a byte (0-255)
static inline int is_byte_token(uint32_t token) {
    return token <= 255;
}

// Helper function to find a token in the map
static inline KV* find_token_in_map(uint32_t token, Map map) {
    for (size_t i = 0; i < map.count; i++) {
        if (map.items[i].value == token) {
            return &map.items[i];
        }
    }
    return NULL;
}

// Convert a single token back to bytes, writing to the output buffer
// Returns number of bytes written
size_t token_to_bytes(uint32_t token, char* output, size_t max_len, Map map) {
    // Base case: if it's a byte token, write it directly
    if (is_byte_token(token)) {
        if (max_len < 1) return 0;
        output[0] = (char)token;
        return 1;
    }
    
    // Find this token in the map
    KV* kv = find_token_in_map(token, map);
    if (!kv) {
        fprintf(stderr, "Error: Token %u not found in map\n", token);
        return 0;  // Token not found
    }
    
    // Recursively decode left and right components
    size_t bytes_written = 0;
    
    // Decode left component
    size_t left_bytes = token_to_bytes(kv->l, output, max_len, map);
    if (left_bytes == 0) return 0;  // Error or buffer full
    bytes_written += left_bytes;
    
    // Decode right component
    if (bytes_written >= max_len) return bytes_written;  // Buffer full
    size_t right_bytes = token_to_bytes(kv->r, output + left_bytes, max_len - left_bytes, map);
    if (right_bytes == 0) return bytes_written;  // Error or buffer full
    bytes_written += right_bytes;
    
    return bytes_written;
}

// Convert array of tokens back to bytes
size_t tokens_to_bytes(const uint32_t* tokens, size_t token_count, char* output, size_t max_len, Map map) {
    if (!tokens || !output || token_count == 0 || max_len == 0) return 0;
    
    size_t total_written = 0;
    
    for (size_t i = 0; i < token_count; i++) {
        // Check remaining buffer space
        if (total_written >= max_len) break;
        
        size_t bytes_written = token_to_bytes(tokens[i], 
                                            output + total_written, 
                                            max_len - total_written, 
                                            map);
        
        if (bytes_written == 0) break;  // Error occurred
        total_written += bytes_written;
    }
    
    return total_written;
}

// This function doesn't make sense to me, we need to fix it. Take in one token and use the defined map (which is a graph), to convert it back to a string of characters (could be one or more characters). Here is the map struct: 

Tokens convert_to_tokens(const char* text, Map map) {
    // takes a string of text, converts to list of uint32_t tokens
    Tokens tokens = {0};
    size_t text_len = strlen(text);
    
    // Start with byte-level tokens
    for (size_t i = 0; i < text_len; i++) {
        uint32_t byte_token = (uint32_t)(unsigned char)text[i];
        da_append(&tokens, byte_token);
    }
    
    // Merge tokens according to the BPE map
    int merged;
    do {
        merged = 0;
        for (size_t i = 0; i < tokens.count - 1; i++) {
            // Try to find a merge rule for adjacent tokens
            for (size_t j = 0; j < map.count; j++) {
                if (map.items[j].l == tokens.items[i] && i + 1 < tokens.count && map.items[j].r == tokens.items[i + 1]) {
                    // Found a merge rule - apply it
                    tokens.items[i] = map.items[j].value;
                    // Remove the right token by shifting everything left
                    for (size_t k = i + 1; k < tokens.count - 1; k++) {
                        tokens.items[k] = tokens.items[k + 1];
                    }
                    tokens.count--;
                    merged = 1;
                    break;
                }
            }
            if (merged) break;  // Start over since we modified the tokens
        }
    } while (merged);

    return tokens;
}

int main() { 
    // TODO: load the byte pair encoding from the previous program.
    // TODO: accept input from the user in the command line and then run the forward part
    
    Map map = load_map("bpe");
    printf("map loaded\n");
    printf("vocab size: %ld\n", map.count);

    // Maximum sequence length for which we pre-compute positional encodings
    
    char* example = "What is the second letter of the alphabet?";
    Tokens tokens = convert_to_tokens(example, map);
    
    int max_seq_len = 10000;
    float* pe = malloc(max_seq_len * DIMENSION_EMBEDDING * sizeof(float));
    compute_positional_encoding(pe, max_seq_len, DIMENSION_EMBEDDING);

    
    // Use the actual sequence length from our example tokens
    int seq_len = tokens.count;
    
    // learned embedding projection
    float* E = malloc(VOCAB_SIZE * DIMENSION_EMBEDDING * sizeof(float));
    // embedding vector that will be input into model
    float* embeddings = malloc(seq_len * DIMENSION_EMBEDDING * sizeof(float));
    
    // Convert our actual tokens to embeddings
    float* token_floats = malloc(seq_len * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        token_floats[i] = (float)tokens.items[i];
    }
    convert_to_embeddings(E, embeddings, token_floats, seq_len);
    free(token_floats);
    
    // Add positional encodings to the embeddings
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < DIMENSION_EMBEDDING; j++) {
            // Add the positional encoding to the token embedding
            // This ensures the model can distinguish between tokens at different positions
            embeddings[i * DIMENSION_EMBEDDING + j] += pe[i * DIMENSION_EMBEDDING + j];
        }
    }


    // dummy scaled attention, 
    float *Q = malloc(seq_len* DIMENSION_KEYS * NUM_HEADS * sizeof(float));
    float *K = malloc(seq_len * DIMENSION_KEYS * NUM_HEADS * sizeof(float));
    float *V = malloc(seq_len * DIMENSION_VALUES * NUM_HEADS * sizeof(float));

    float* W_Q = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS * sizeof(float));
    float* W_K = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS * sizeof(float));
    float* W_V = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_VALUES * sizeof(float));
    float* W_O = malloc(DIMENSION_HIDDEN * DIMENSION_HIDDEN * sizeof(float));

    init_rand(W_Q, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS);
    init_rand(W_K, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS);
    init_rand(W_V, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_VALUES);
    init_rand(W_O, DIMENSION_HIDDEN * DIMENSION_HIDDEN);

    // NUM_HEADS * DIMENSION_VALUES = DIMENSION_EMBEDDING
    // hence, we add an norm with the input the is seq_len DIMENSION_EMBEDDING
    float *result = malloc(NUM_HEADS * seq_len * DIMENSION_VALUES * sizeof(float));
    // Initialize result array to zeros

    // Apply attention for each head
    for (int head = 0; head < NUM_HEADS; head++) {
        float *Q_head = Q + (head * seq_len * DIMENSION_KEYS);
        float *K_head = K + (head * seq_len * DIMENSION_KEYS);
        float *V_head = V + (head * seq_len * DIMENSION_VALUES);
        float *result_head = result + (head * seq_len * DIMENSION_VALUES);
        
        // Project embeddings to Q, K, V for this head using weight matrices
        float *W_Q_head = W_Q + (head * DIMENSION_HIDDEN * DIMENSION_KEYS);
        float *W_K_head = W_K + (head * DIMENSION_HIDDEN * DIMENSION_KEYS);
        float *W_V_head = W_V + (head * DIMENSION_HIDDEN * DIMENSION_VALUES);
        
        // Project embeddings to Q, K, V matrices
        matmul(embeddings, W_Q_head, seq_len, DIMENSION_HIDDEN, DIMENSION_KEYS, Q_head);
        matmul(embeddings, W_K_head, seq_len, DIMENSION_HIDDEN, DIMENSION_KEYS, K_head);
        matmul(embeddings, W_V_head, seq_len, DIMENSION_HIDDEN, DIMENSION_VALUES, V_head);
        
        attention(Q_head, K_head, V_head, seq_len, result_head);
    }


    // First add & norm after attention
    float* result_aan1 = malloc(DIMENSION_EMBEDDING * seq_len * sizeof(float)); 
    add_and_norm(embeddings, result, result_aan1, seq_len, DIMENSION_EMBEDDING);

    // Feed-forward neural network block
    float* ffn_weights1 = malloc(DIMENSION_HIDDEN * NN_SIZE * sizeof(float));
    float* ffn_weights2 = malloc(NN_SIZE * DIMENSION_HIDDEN * sizeof(float));
    init_rand(ffn_weights1, DIMENSION_HIDDEN * NN_SIZE);
    init_rand(ffn_weights2, NN_SIZE * DIMENSION_HIDDEN);
    
    float* result_aan2 = malloc(DIMENSION_EMBEDDING * seq_len * sizeof(float));
    ffn_block(result_aan1, ffn_weights1, ffn_weights2, result_aan2, seq_len);

    // Decoder side implementation
    // 1. Masked Multi-head Attention
    float *decoder_Q = malloc(seq_len * DIMENSION_KEYS * NUM_HEADS * sizeof(float));
    float *decoder_K = malloc(seq_len * DIMENSION_KEYS * NUM_HEADS * sizeof(float));
    float *decoder_V = malloc(seq_len * DIMENSION_VALUES * NUM_HEADS * sizeof(float));
    
    float* decoder_W_Q = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS * sizeof(float));
    float* decoder_W_K = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS * sizeof(float));
    float* decoder_W_V = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_VALUES * sizeof(float));
    float* decoder_W_O = malloc(DIMENSION_HIDDEN * DIMENSION_HIDDEN * sizeof(float));
    
    init_rand(decoder_W_Q, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS);
    init_rand(decoder_W_K, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS);
    init_rand(decoder_W_V, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_VALUES);
    init_rand(decoder_W_O, DIMENSION_HIDDEN * DIMENSION_HIDDEN);

    float *masked_mha_result = malloc(NUM_HEADS * seq_len * DIMENSION_VALUES * sizeof(float));

    // Apply masked attention for each head
    for (int head = 0; head < NUM_HEADS; head++) {
        float *Q_head = decoder_Q + (head * seq_len * DIMENSION_KEYS);
        float *K_head = decoder_K + (head * seq_len * DIMENSION_KEYS);
        float *V_head = decoder_V + (head * seq_len * DIMENSION_VALUES);
        float *result_head = masked_mha_result + (head * seq_len * DIMENSION_VALUES);
        
        float *W_Q_head = decoder_W_Q + (head * DIMENSION_HIDDEN * DIMENSION_KEYS);
        float *W_K_head = decoder_W_K + (head * DIMENSION_HIDDEN * DIMENSION_KEYS);
        float *W_V_head = decoder_W_V + (head * DIMENSION_HIDDEN * DIMENSION_VALUES);
        
        matmul(embeddings, W_Q_head, seq_len, DIMENSION_HIDDEN, DIMENSION_KEYS, Q_head);
        matmul(embeddings, W_K_head, seq_len, DIMENSION_HIDDEN, DIMENSION_KEYS, K_head);
        matmul(embeddings, W_V_head, seq_len, DIMENSION_HIDDEN, DIMENSION_VALUES, V_head);
        
        masked_attention(Q_head, K_head, V_head, seq_len, result_head);
    }

    // Add & norm after masked attention
    float* masked_aan = malloc(DIMENSION_EMBEDDING * seq_len * sizeof(float));
    add_and_norm(embeddings, masked_mha_result, masked_aan, seq_len, DIMENSION_EMBEDDING);

    // 2. Cross attention with encoder outputs (result_aan2)
    float *cross_Q = malloc(seq_len * DIMENSION_KEYS * NUM_HEADS * sizeof(float));
    float *cross_K = malloc(seq_len * DIMENSION_KEYS * NUM_HEADS * sizeof(float));
    float *cross_V = malloc(seq_len * DIMENSION_VALUES * NUM_HEADS * sizeof(float));
    
    float* cross_W_Q = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS * sizeof(float));
    float* cross_W_K = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS * sizeof(float));
    float* cross_W_V = malloc(NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_VALUES * sizeof(float));
    float* cross_W_O = malloc(DIMENSION_HIDDEN * DIMENSION_HIDDEN * sizeof(float));
    
    init_rand(cross_W_Q, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS);
    init_rand(cross_W_K, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_KEYS);
    init_rand(cross_W_V, NUM_HEADS * DIMENSION_HIDDEN * DIMENSION_VALUES);
    init_rand(cross_W_O, DIMENSION_HIDDEN * DIMENSION_HIDDEN);

    float *cross_result = malloc(NUM_HEADS * seq_len * DIMENSION_VALUES * sizeof(float));

    // Apply cross attention for each head
    for (int head = 0; head < NUM_HEADS; head++) {
        float *Q_head = cross_Q + (head * seq_len * DIMENSION_KEYS);
        float *K_head = cross_K + (head * seq_len * DIMENSION_KEYS);
        float *V_head = cross_V + (head * seq_len * DIMENSION_VALUES);
        float *result_head = cross_result + (head * seq_len * DIMENSION_VALUES);
        
        float *W_Q_head = cross_W_Q + (head * DIMENSION_HIDDEN * DIMENSION_KEYS);
        float *W_K_head = cross_W_K + (head * DIMENSION_HIDDEN * DIMENSION_KEYS);
        float *W_V_head = cross_W_V + (head * DIMENSION_HIDDEN * DIMENSION_VALUES);
        
        // Q comes from decoder, K and V come from encoder
        matmul(masked_aan, W_Q_head, seq_len, DIMENSION_HIDDEN, DIMENSION_KEYS, Q_head);
        matmul(result_aan2, W_K_head, seq_len, DIMENSION_HIDDEN, DIMENSION_KEYS, K_head);
        matmul(result_aan2, W_V_head, seq_len, DIMENSION_HIDDEN, DIMENSION_VALUES, V_head);
        
        attention(Q_head, K_head, V_head, seq_len, result_head);
    }

    // Add & norm after cross attention
    float* cross_aan = malloc(DIMENSION_EMBEDDING * seq_len * sizeof(float));
    add_and_norm(masked_aan, cross_result, cross_aan, seq_len, DIMENSION_EMBEDDING);

    // 3. Feed-forward neural network block
    float* decoder_ffn_weights1 = malloc(DIMENSION_HIDDEN * NN_SIZE * sizeof(float));
    float* decoder_ffn_weights2 = malloc(NN_SIZE * DIMENSION_HIDDEN * sizeof(float));
    init_rand(decoder_ffn_weights1, DIMENSION_HIDDEN * NN_SIZE);
    init_rand(decoder_ffn_weights2, NN_SIZE * DIMENSION_HIDDEN);
    
    float* ffn_output = malloc(DIMENSION_EMBEDDING * seq_len * sizeof(float));
    ffn_block(cross_aan, decoder_ffn_weights1, decoder_ffn_weights2, ffn_output, seq_len);

    // 4. Final linear layer and softmax
    float* output_weights = malloc(DIMENSION_HIDDEN * VOCAB_SIZE * sizeof(float));
    init_rand(output_weights, DIMENSION_HIDDEN * VOCAB_SIZE);
    
    float* logits = malloc(seq_len * VOCAB_SIZE * sizeof(float));
    float* probabilities = malloc(seq_len * VOCAB_SIZE * sizeof(float));
    
    // Linear transformation for each token
    matmul(ffn_output, output_weights, seq_len, DIMENSION_HIDDEN, VOCAB_SIZE, logits);
    
    // Apply softmax to get probabilities
    // Apply softmax to get probabilities
    softmax_matrix(logits, probabilities, seq_len, VOCAB_SIZE);
    
    // For each position, find the token with highest probability
    uint32_t* output_tokens = malloc(seq_len * sizeof(uint32_t));
    float max_prob = 0.0;
    int max_idx = 0;
    for (int j = 1; j < VOCAB_SIZE; j++) {
        if (probabilities[j] > max_prob) {
            max_prob = probabilities[j];
            max_idx = j;
        }
    }
    output_tokens[0] = (uint32_t)max_idx;

    // printf("output: %d \n", output_tokens[0]);
    printf("output token: %d \n", max_idx);

    size_t max_len = 16;
    // Convert tokens back to text
    char* output_buffer = malloc(128); // Allocate enough space for worst case
    // size_t bytes_written = tokens_to_bytes(output_tokens, seq_len, output_buffer, seq_len * 4, map);
    size_t bytes_written = token_to_bytes(max_idx, output_buffer, max_len, map);

    printf("bytes written: %ld\n", bytes_written);
    output_buffer[bytes_written] = '\0'; // Null terminate the string
    printf("decoded token: \n");                                     
    for (size_t i = 0; i < bytes_written; i++) {
        printf("%d", output_buffer[i]);
    } 
    printf("\n");                                     

    // Free the new allocations
    free(output_tokens);
    free(output_buffer);



    // Free decoder-side allocations
    free(decoder_Q);
    free(decoder_K);
    free(decoder_V);
    free(decoder_W_Q);
    free(decoder_W_K);
    free(decoder_W_V);
    free(decoder_W_O);
    free(masked_mha_result);
    free(masked_aan);
    free(cross_Q);
    free(cross_K);
    free(cross_V);
    free(cross_W_Q);
    free(cross_W_K);
    free(cross_W_V);
    free(cross_W_O);
    free(cross_result);
    free(cross_aan);
    free(decoder_ffn_weights1);
    free(decoder_ffn_weights2);
    free(ffn_output);
    free(output_weights);
    free(logits);
    free(probabilities);
  // printf("print vectors of result\n");
  // for(int i = 0; i < 10; i ++) { 
  //     for(int j = 0; j < 10; j ++) { 
  //    // printf("%f\n", result[NUM_HEADS * DIMENSION_VALUES * seq_len + i]);
  //     printf("%f\n", result_aan2[DIMENSION_EMBEDDING*i + j]);
  //     // printf("%f\n", result[i]);
  //     } 
  //     printf("\n");
  // } 

    fprintf(stdout, "attention finished\n");

    free(pe);
    free(E);
    free(embeddings);
    free(W_Q);
    free(W_K);
    free(W_V);
    free(W_O);
    free(Q);
    free(K);
    free(V);
    free(result);
    free(result_aan1);
    free(result_aan2);
    free(ffn_weights1);
    free(ffn_weights2);

} 
