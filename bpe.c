#include <stdio.h> 
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <errno.h>

#define HASHMAP_IMPLEMENTATION
#include "hashmap.h"


//thank you tsoding
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

int contains(Counts* arr, uint32_t x[]) { 
    for (size_t i = 0; i < arr->count; i ++)  { 
        if (arr->items[i].text[0] == x[0] && arr->items[i+1].text[1] == x[1]) { 
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
    //if (count % 1024 == 0) { 
    // printf("on iteration %d, current size = %ld\n", count, input->count);
    //} 
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
        fprintf(file, "\t%d => %d, %d\n", kv.value, kv.l, kv.r);
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

int main() { 
    // Load text from file
    size_t text_len = 0;
    char* text_test = load_text_from_file("dostoevsky.txt", &text_len); // just took from claude thanks
    // char* text_test = load_text_from_file("dostoevsky_long.txt", &text_len); // just took from claude thanks
    
    if (text_test == NULL) {
        fprintf(stderr, "Failed to load text from test.txt. Exiting.\n");
        return 1;
    }
    
    printf("Loaded %zu bytes from test.txt\n", text_len);

    // define tokens
    Tokens tokens = {0}; 
    for (size_t i = 0; i < text_len; i ++) { 
        da_append(&tokens, text_test[i]);
    } 

    Tokens output_tokens = {0};
    Tokens temp_tokens = {0};
    Map map = {0}; 
    uint32_t iteration = 0;

    printf("Initial text size: %lld\n", tokens.count);
    

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
    printf("Final text size: %lld\n", tokens.count);
    printf("Final vocab size: %lld\n", 256 + map.count);


    FILE* file = fopen("mapping.txt", "w"); 
    print_to_file(file, &map);
    
    // Free allocated memory
    free(tokens.items);
    free(output_tokens.items);
    free(text_test);  // Don't forget to free the loaded text buffer

    return 0;

} 
