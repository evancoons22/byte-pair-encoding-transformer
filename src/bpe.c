#include <stdio.h> 
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <errno.h>

#define HASHMAP_IMPLEMENTATION
#include "hashmap.h"

#define HEAP_IMPLEMENTATION
#include "bpe_heap.h"


//thank you https://github.com/tsoding

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
	uint32_t text[2];
    int count;
} Count;

typedef struct { 
    Count* items;
    size_t count;
    size_t capacity;
} Counts;

typedef struct { 
    char* items;
    size_t count;
    size_t capacity;
} Text;

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

void save_map(const char *filename, const Map *map) {
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "build/%s", filename);
    
    // Validate the map before saving
    for (size_t i = 0; i < map->count; i++) {
        if (map->items[i].value < 256 || 
            map->items[i].l > UINT32_MAX || 
            map->items[i].r > UINT32_MAX) {
            fprintf(stderr, "Invalid token values detected at index %zu: value=%u, l=%u, r=%u\n", 
                    i, map->items[i].value, map->items[i].l, map->items[i].r);
            return;
        }
    }

    FILE *f = fopen(filepath, "wb");
    if (!f) return;

    fwrite(&map->count, sizeof(size_t), 1, f);
    fwrite(map->items, sizeof(KV), map->count, f);

    fclose(f);
}

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

int contains(Counts* arr, uint32_t x[]) { 
    for (size_t i = 0; i < arr->count; i ++)  { 
        if (arr->items[i].text[0] == x[0] && arr->items[i].text[1] == x[1]) { 
            return i;
        } 
    } 
    return -1;
} 

int compare(const void* a, const void* b) {
    const Count* pairA = (const Count*)a;
    const Count* pairB = (const Count*)b;

    if (pairA->count < pairB->count) return 1;
    if (pairA->count > pairB->count) return -1;
    return 0;
}

    //print counter
void print_counter(Counts* counter) { 
    // for (size_t i = 0; i < counter->count; i ++)  { 
    for (size_t i = 0; i < 10; i ++)  { 
        printf("%c%c: %d\n", counter->items[i].text[0], counter->items[i].text[1], counter->items[i].count);
    } 
} 

void build_counter(Counts* counter, Tokens* tokens) { 
    for (size_t i = 0; i < tokens->count - 1; i ++)  { 
        Count p = {
            .text = { tokens->items[i],tokens->items[i+1] },
            .count = 1
        };
        int v = contains(counter, p.text);
        if (v < 0) { 
            da_append(counter, p);
        } else {  
            counter->items[v].count += 1;
        } 
    } 

} 

int compress(Tokens* input, Tokens* output, Map* map, uint32_t count) { 
   // if (count % 1 == 0) { 
   //  printf("on iteration %d, current size = %ld\n", count, input->count);
   // } 
    Counts counter = {0};
    build_counter(&counter, input);
    qsort(counter.items, counter.count, sizeof(Count), compare);

    // uint32_t* top = counter.items[0];
    Count top = counter.items[0];

    if (counter.count == 0) {
        fprintf(stderr, "No pairs found\n");
        return 1;
    }
    if (counter.items[0].count <=1 )  return 1;  

    uint32_t new_token_id = 256 + count;

    // add new key value to the map
    KV kv = {
        .value = new_token_id,
        .l = top.text[0],
        .r = top.text[1]
    }; 
    da_append(map, kv);

    size_t i = 0;
    while (i < input->count) {
        if (i + 1 < input->count && input->items[i] == top.text[0] && input->items[i + 1] == top.text[1]) {
            da_append(output, new_token_id);
            i += 2; 
        } else {
            da_append(output, input->items[i]);
            i += 1;
        }
    }
    
    return 0;
} 

// go through file, find the most often token (a pair) 
// for example a b b c
// replace token b b, so a z c
// decrement a b 1, b c 1
// increment a z 1, z c 1

void build_heap_counter(BPEHeap* heap, Tokens* tokens) {
    for (size_t i = 0; i < tokens->count - 1; i++) { 
        if (tokens->items[i] >= 256 || tokens->items[i+1] >= 256) {
            // fprintf(stderr, "Warning: Token value out of ASCII range (>=256) at position %zu: %u, %u\n", i, tokens->items[i], tokens->items[i+1]);
            // just skip if token is not ascii
            continue;
        }
        // adding item to the heap
        bpe_heap_bump(heap, tokens->items[i], tokens->items[i+1]);
    }
} 

int compress2(Tokens* input, Tokens* output, Map* map, BPEHeap* heap, uint32_t count) { 
   if (count % 100 == 0) { 
    printf("on iteration %d, text length = %ld\n", count, input->count);
   } 

    // get the top item
    uint32_t a = 0, b = 0; 
    uint64_t value = 0;
    int pop_result = bpe_heap_pop_max(heap, &a, &b, &value);

    if (!pop_result) {
        fprintf(stderr, "No pairs found\n");
        return 1;
    }
    if (value <=1 )  return 1;  

    uint32_t new_token_id = 256 + count;

    // add new key value to the map
    KV kv = {
        .value = new_token_id,
        .l = a,
        .r = b
    }; 
    da_append(map, kv);

    size_t i = 0;
    // assume sequence found aa a b bb
    // then we will have aa Z bb
    // add these two to the map
    while (i < input->count) {
        if (i + 1 < input->count && input->items[i] == a && input->items[i + 1] == b) {
            // Only decrement pairs if they exist (not at boundaries)
            if (i > 0) {
                bpe_heap_decrement(heap, input->items[i-1], a);
            }
            if (i + 2 < input->count) {
                bpe_heap_decrement(heap, b, input->items[i+2]);
            }

            // increment these a Z, Z b
            bpe_heap_bump(heap, a, new_token_id);
            bpe_heap_bump(heap, new_token_id, b);

            // add to the new token stream
            da_append(output, new_token_id);
            i += 2; 
        } else {
            da_append(output, input->items[i]);
            i += 1;
        }
    }
    
    return 0;
} 


// Function to load text from a file
char* load_text_from_file(const char* filename, size_t* text_len) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size < 0) {
        fprintf(stderr, "Error determining size of file '%s': %s\n", filename, strerror(errno));
        fclose(file);
        return NULL;
    }

    // Allocate buffer with extra byte for null terminator
    char* buffer = (char*)malloc(file_size + 1);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed for file '%s'\n", filename);
        fclose(file);
        return NULL;
    }

    // Read file content
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read < (size_t)file_size && ferror(file)) {
        fprintf(stderr, "Error reading file '%s': %s\n", filename, strerror(errno));
        free(buffer);
        fclose(file);
        return NULL;
    }

    // Null-terminate the string
    buffer[bytes_read] = '\0';
    *text_len = bytes_read;
    
    fclose(file);
    return buffer;
}

void print_to_file(FILE* file, Map* map) { 
    printf("Printing encoding to file\n");
    if (file == NULL) { 
        printf("error opening file\n");
    } 
    fprintf(file, "map = {\n");
    for (size_t i = 0; i < map->count; i++) {
        KV kv = map->items[i];
        // printf("%d => %d, %d\n", kv.value, kv.l, kv.r);
        // fprintf(file, "\t%d => %d, %d\n", kv.value, kv.l, kv.r);
        fprintf(file, "\t%u => %u, %u\n", kv.value, kv.l, kv.r);
    }
    fprintf(file, "}");
} 

void render_tokens(Tokens* tokens) { 
    for (size_t i = 0; i < tokens->count; i++) {
        if (tokens->items[i] < 256) {
            printf("%c", (char)tokens->items[i]);
        } else {
            printf("[%u]", tokens->items[i]);
        }
    }
    printf("\n");
} 

void run_version_1() { 
    // Load text from file
    size_t text_len = 0;
    char* text_test = load_text_from_file("examples/dostoevsky.txt", &text_len); 
    // char* text_test = load_text_from_file("examples/dostoevsky_long.txt", &text_len);
    // char* text_test = load_text_from_file("examples/shakespeare.txt", &text_len); 
    
    if (text_test == NULL) {
        fprintf(stderr, "Failed to load text from test.txt. Exiting.\n");
    }
    
    printf("Loaded %zu bytes from file\n", text_len);

    // define tokens
    Tokens tokens = {0}; 
    for (size_t i = 0; i < text_len; i ++) { 
        da_append(&tokens, text_test[i]);
    } 

    Tokens output_tokens = {0};
    Tokens temp_tokens = {0};
    Map map = {0}; 
    uint32_t iteration = 0;

    printf("Initial text size: %ld\n", tokens.count);
    

    while (compress(&tokens, &output_tokens, &map, iteration) != 1) {
        iteration++;
        // printf("Iteration %u - Compressed size: %ld\n", iteration, output_tokens.count);
        
        // Swap the buffers properly
        temp_tokens = tokens;      
        tokens = output_tokens;    
        output_tokens.items = temp_tokens.items;
        output_tokens.capacity = temp_tokens.capacity;
        output_tokens.count = 0;   // Reset count but keep the allocated memory
    }

    printf("Iterations : %u\n", iteration);
    printf("Final text size: %ld\n", tokens.count);
    printf("Final vocab size: %ld\n", 256 + map.count);

    //render_tokens(&tokens);


    FILE* file = fopen("mapping.txt", "w"); 
    print_to_file(file, &map);
    
    // Free allocated memory
    free(tokens.items);
    free(output_tokens.items);
    free(text_test);  // Don't forget to free the loaded text buffer

} 

void run_version_2() { 
    BPEHeap heap = BPE_HEAP_INITIALIZER;
    
    // read in the file
    size_t text_len = 0;
    // char* text_test = load_text_from_file("examples/dostoevsky.txt", &text_len); 
    char* text_test = load_text_from_file("examples/dostoevsky_long.txt", &text_len);
    // char* text_test = load_text_from_file("examples/shakespeare.txt", &text_len); 
    
    if (text_test == NULL) {
        fprintf(stderr, "Failed to load text from test.txt. Exiting.\n");
    }
    printf("Loaded %zu bytes from test.txt\n", text_len);
    // define tokens
    Map map = {0}; 
    Tokens tokens = {0}; 
    Tokens output_tokens = {0}; 
    Tokens temp_tokens = {0};
    for (size_t i = 0; i < text_len; i ++) { 
        da_append(&tokens, text_test[i]);
    } 

    printf("Initial text size: %ld\n", tokens.count);

    // count the actual items
    build_heap_counter(&heap, &tokens);


    uint32_t iteration = 0;
    while (compress2(&tokens, &output_tokens, &map, &heap, iteration) != 1) {
        iteration++;
        // printf("Iteration %u - Compressed size: %ld\n", iteration, output_tokens.count);
        // Swap the buffers properly
        temp_tokens = tokens;      
        tokens = output_tokens;    
        output_tokens.items = temp_tokens.items;
        output_tokens.capacity = temp_tokens.capacity;
        output_tokens.count = 0;   // Reset count but keep the allocated memory
    }

    FILE* file = fopen("mapping.txt", "w"); 
    print_to_file(file, &map);
    save_map("bpe_mem", &map);


    printf("Iterations : %u\n", iteration);
    printf("Final text size: %ld\n", tokens.count);
    printf("Final vocab size: %ld\n", 256 + map.count);

    bpe_heap_free(&heap);
} 


int main() { 
   // printf("version 1: \n");
   // run_version_1();
   // printf("\n\n");
   run_version_2();

    return 0;

} 
