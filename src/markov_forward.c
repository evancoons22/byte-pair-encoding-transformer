#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include "uthash.h"    // for convert_to_tokens

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

/* user‐provided */
typedef struct { uint32_t value, l, r; } KV;
typedef struct { KV *items; size_t count, capacity; } Map;
typedef struct { uint32_t *items; size_t count, capacity; } Tokens;
Map load_map(const char *fname);
Tokens convert_to_tokens(const char *text, Map map);

/* Markov chain structs */
typedef struct { uint32_t to, cnt; } Trans;
typedef struct { Trans *items; size_t count, capacity; } MarkovState;

typedef struct { uint64_t key; uint32_t value; UT_hash_handle hh; } Merge;

Tokens convert_to_tokens(const char *text, Map map) {
    size_t text_len = strlen(text);
    Tokens tokens = { .count = 0, .capacity = text_len, 
                      .items = malloc(text_len * sizeof *tokens.items) };

    /* 1) byte-level tokens */
    for (size_t i = 0; i < text_len; i++)
        da_append(&tokens, (uint32_t)(unsigned char)text[i]);

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

/* load model from binary */
MarkovState* load_markov_chain(const char *fname, size_t *outV) {
    char path[4096];
    snprintf(path, sizeof(path), "build/%s", fname);
    FILE *f = fopen(path, "rb"); assert(f);
    size_t V; fread(&V, sizeof V, 1, f);
    MarkovState *model = calloc(V, sizeof *model);
    for (size_t i = 0; i < V; i++) {
        uint32_t c; fread(&c, sizeof c, 1, f);
        if (c) {
            model[i].count = model[i].capacity = c;
            model[i].items = malloc(c * sizeof *model[i].items);
            fread(model[i].items, sizeof *model[i].items, c, f);
        }
    }
    fclose(f);
    *outV = V;
    return model;
}

Map load_map(const char *filename) {
    Map map = {0};
    char path[4096];
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

// get next token
uint32_t find_next_token(uint32_t t, MarkovState *model) {
    MarkovState *st = &model[t]; assert(st->count);
    uint64_t sum = 0;
    for (size_t i = 0; i < st->count; i++) sum += st->items[i].cnt;
    uint64_t r = rand() % sum;
    for (size_t i = 0; i < st->count; i++) {
        if (r < st->items[i].cnt) return st->items[i].to;
        r -= st->items[i].cnt;
    }
    return st->items[st->count-1].to;
}

// print the next token to text using the map
void print_string_from_token(uint32_t t, Map map) {
    if (t < 256) { putchar((char)t); return; }
    for (size_t j = 0; j < map.count; j++) {
        if (map.items[j].value == t) {
            print_string_from_token(map.items[j].l, map);
            print_string_from_token(map.items[j].r, map);
            return;
        }
    }
    printf("[0x%X]", t);
}

int main(void) {
    // load bpe and markov chain
    Map map = load_map("bpe_mem");
    size_t V;
    MarkovState *model = load_markov_chain("markov_mem", &V);

    // input
    char buf[4096];
    printf("Enter seed text> ");
    if (!fgets(buf, sizeof buf, stdin)) return 0;
    buf[strcspn(buf, "\n")] = 0;

    // convert to tokens and use only the last (markov chain)
    Tokens tokens = convert_to_tokens(buf, map);
    if (!tokens.count) return 0;
    uint32_t cur = tokens.items[tokens.count - 1];

    // 100 next tokens
    for (int i = 0; i < 100; i++) {
        uint32_t nxt = find_next_token(cur, model);
        print_string_from_token(nxt, map);
        cur = nxt;
    }
    putchar('\n');
    return 0;
}

