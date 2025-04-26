#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifndef DA_INIT_CAP
#define DA_INIT_CAP 32
#endif

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

Tokens convert_to_tokens(const char* text, Map map) {
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

int main() { 
    Map map = load_map("bpe");
    printf("vocab size: %zu\n", map.count);
    char* text = "convert this text to tokens using bpe map. Defined above by the structs.";
    printf("text original length: %zu\n", strlen(text));
    
    Tokens tokens = convert_to_tokens(text, map);
    
    printf("Converted text to %zu tokens:\n", tokens.count);
    for (size_t i = 0; i < tokens.count; i++) {
        printf("%u ", tokens.items[i]);
    }
    printf("\n");
    
    // Cleanup
    free(tokens.items);
    free(map.items);
    return 0;
} 
