#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <errno.h>
#include "uthash.h"

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

// this is for the markov chain... vocab size of about 6000
typedef struct {
    uint32_t to;      
    uint32_t cnt;     // frequency
} Trans;

// dynamic array of Trans
typedef struct {
    Trans *items;
    size_t count;
    size_t capacity;
} MarkovState;

/* simple map-pair hash: key=(l<<32)|r, value=new_token */
typedef struct {
    uint64_t key;
    uint32_t value;
    UT_hash_handle hh;
} Merge;

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

Map load_map(const char *filename) {
    Map map = {0};
    char path[256];
    snprintf(path, sizeof(path), "build/%s", filename);
    FILE *f = fopen(path, "rb");
    if (!f) return map;

    fread(&map.count, sizeof(size_t), 1, f);
    map.items = malloc(map.count * sizeof(KV));  // malloc is fine even if it's fixed
    fread(map.items, sizeof(KV), map.count, f);

    map.capacity = map.count;  // not needed, but harmless

    fclose(f);
    return map;
}


Tokens convert_to_tokens(const char *text, Map map) {
    size_t text_len = strlen(text);
    Tokens tokens = { .count = 0, .capacity = text_len, 
                      .items = malloc(text_len * sizeof *tokens.items) };

    /* 1) byte-level tokens - only include ASCII characters */
    for (size_t i = 0; i < text_len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c < 128) {  // Only include ASCII characters (0-127)
            da_append(&tokens, (uint32_t)c);
        }
    }

    /* 2) build the merge-rule hash */
    Merge *mh = NULL;
    for (size_t j = 0; j < map.count; j++) {
        uint64_t k = ((uint64_t)map.items[j].l << 32) | map.items[j].r;
        Merge *m = malloc(sizeof *m);
        m->key   = k;
        m->value = map.items[j].value;
        HASH_ADD(hh, mh, key, sizeof m->key, m);
    }

    /* 3) merge loop in a single pass until no more merges */
    int merged;
    do {
        merged = 0;
        size_t i = 0;
        while (i + 1 < tokens.count) {
            uint64_t k = ((uint64_t)tokens.items[i] << 32) | tokens.items[i+1];
            Merge *m;
            HASH_FIND(hh, mh, &k, sizeof k, m);
            if (m) {
                tokens.items[i] = m->value;
                /* shift tail left by one */
                memmove(&tokens.items[i+1],
                        &tokens.items[i+2],
                        (tokens.count - i - 2) * sizeof *tokens.items);
                tokens.count--;
                merged = 1;
                if (i) i--;
            } else {
                i++;
            }
        }
    } while (merged);

    /* cleanup hash */
    Merge *current, *tmp;
    HASH_ITER(hh, mh, current, tmp) {
        HASH_DEL(mh, current);
        free(current);
    }

    return tokens;
}

uint32_t find_next_token(uint32_t t, MarkovState *model) {
    MarkovState *st = &model[t];
    assert(st->count > 0 && "No outgoing transitions for this token");

    // sum frequencies and pick a random offset
    uint64_t total = 0;
    for (size_t i = 0; i < st->count; i++)
        total += st->items[i].cnt;
    uint64_t r = (uint64_t)rand() % total;

    // then just walk along until hitting next item
    // items with high count will be hit more frequently
    for (size_t i = 0; i < st->count; i++) {
        if (r < st->items[i].cnt)
            return st->items[i].to;
        r -= st->items[i].cnt;
    }

    // fallback choose last (shouldn't happen)
    return st->items[st->count - 1].to;
}

void print_string_from_token(uint32_t t, Map map) {
    if (t < 256) {
        // print if ASCII char
        putchar((char)t);
        return;
    }
    // look for the rule that produced t
    for (size_t j = 0; j < map.count; j++) {
        if (map.items[j].value == t) {
            print_string_from_token(map.items[j].l, map);
            print_string_from_token(map.items[j].r, map);
            return;
        }
    }
    // print placeholder if nothing
    printf("[0x%X]", t);
}

/* record one s → t transition */
void add_transition(MarkovState *model, size_t V, uint32_t s, uint32_t t) {
    if (s >= V) { 
        printf("state found that is larger than vocab: s = %d, V = %ld ", s, V);
    } 
    //assert(s < V);
    MarkovState *st = &model[s];
    for (size_t i = 0; i < st->count; i++) {
        if (st->items[i].to == t) {
            st->items[i].cnt++;
            return;
        }
    }
    Trans new_tr = { .to = t, .cnt = 1 };
    da_append(st, new_tr);
}

void build_markov_chain(Tokens *tokens, MarkovState *model, size_t V) {
    for (size_t i = 0; i + 1 < tokens->count; i++) {
        uint32_t cur = tokens->items[i];
        uint32_t nxt = tokens->items[i+1];
        add_transition(model, V, cur, nxt);
    }
}

// save model to binary
void save_markov_chain(const char *fname, MarkovState *model, size_t V) {
    // Create build directory if it doesn't exist
    if (system("mkdir -p build") != 0) {
        fprintf(stderr, "Error creating build directory\n");
        return;
    }

    char path[256];
    snprintf(path, sizeof(path), "build/%s", fname);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error opening file '%s' for writing: %s\n", path, strerror(errno));
        return;
    }

    fwrite(&V, sizeof V, 1, f);
    for (size_t i = 0; i < V; i++) {
        uint32_t c = model[i].count;
        fwrite(&c, sizeof c, 1, f);
        if (c) fwrite(model[i].items, sizeof *model[i].items, c, f);
    }
    fclose(f);
}

int main() { 
    // load map
    Map map = load_map("bpe_mem");
    int V = map.count + 256;  // Add 256 for ASCII characters
    printf("vocab size: %d\n", V);

    // load text
    size_t text_len = 0;
    char* text = load_text_from_file("examples/dostoevsky_long.txt", &text_len); 
    if (!text) {
        fprintf(stderr, "Failed to load text file\n");
        return 1;
    }
    printf("total initial text size: %ld\n", text_len);

    // convert the text to tokens
    printf("converting text to tokens...\n");
    Tokens tokens = convert_to_tokens(text, map);
    printf("tokenized text size: %ld\n", tokens.count);

    // Initialize model - array of V MarkovState buckets
    MarkovState *model = calloc(V, sizeof *model);
    if (!model) {
        fprintf(stderr, "Failed to allocate model\n");
        free(text);
        free(tokens.items);
        return 1;
    }

    printf("building markov chain...\n");
    build_markov_chain(&tokens, model, V);
    // save markov chaing
    printf("saving markov chain to \"build/markov_mem\"\n");
    save_markov_chain("markov_mem", model, V);
    
    // Clean up
    free(text);
    free(tokens.items);
    
    // Free the model
    for (int i = 0; i < V; i++) {
        free(model[i].items);
    }
    free(model);

    return 0;
} 
