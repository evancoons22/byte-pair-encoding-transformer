#ifndef BPE_HEAP_H
#define BPE_HEAP_H

#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include "uthash.h"

#ifdef HEAP_IMPLEMENTATION

typedef struct BPEEntry {
    uint64_t key;
    uint64_t count;
    size_t   idx;
    UT_hash_handle hh;
} BPEEntry;

typedef struct {
    BPEEntry **heap;
    size_t     size, cap;
    BPEEntry  *map;
} BPEHeap;

#define BPE_HEAP_INITIALIZER {NULL,0,0,NULL}

static inline void bpe_heap_init(BPEHeap *h) {
    h->heap = NULL; h->size = h->cap = 0; h->map = NULL;
}

static void swap_entry(BPEEntry **a, BPEEntry **b) {
    BPEEntry *t = *a; *a = *b; *b = t;
    size_t i = (*a)->idx; (*a)->idx = (*b)->idx; (*b)->idx = i;
}

static void sift_up(BPEHeap *h, size_t i) {
    while(i>0) {
        size_t p = (i-1)/2;
        if(h->heap[p]->count >= h->heap[i]->count) break;
        swap_entry(&h->heap[p], &h->heap[i]);
        i = p;
    }
}

static void sift_down(BPEHeap *h, size_t i) {
    for(;;) {
        size_t l = 2*i+1, r = l+1, m = i;
        if(l<h->size && h->heap[l]->count > h->heap[m]->count) m = l;
        if(r<h->size && h->heap[r]->count > h->heap[m]->count) m = r;
        if(m==i) break;
        swap_entry(&h->heap[i], &h->heap[m]);
        i = m;
    }
}

static BPEEntry *get_entry(BPEHeap *h, uint32_t a, uint32_t b) {
    uint64_t key = ((uint64_t)a<<32)|b;
    BPEEntry *e;
    HASH_FIND(hh, h->map, &key, sizeof(key), e);
    if(!e) {
        e = calloc(1, sizeof *e);
        e->key = key;
        e->count = 0;
        e->idx = h->size;
        HASH_ADD_KEYPTR(hh, h->map, &e->key, sizeof(e->key), e);
        if(h->size == h->cap) {
            h->cap = h->cap ? h->cap*2 : 1024;
            h->heap = realloc(h->heap, h->cap * sizeof *h->heap);
        }
        h->heap[h->size++] = e;
    }
    return e;
}

static inline void bpe_heap_bump(BPEHeap *h, uint32_t a, uint32_t b) {
    BPEEntry *e = get_entry(h,a,b);
    e->count++;
    sift_up(h, e->idx);
}

static inline int bpe_heap_top(BPEHeap *h, uint32_t *a, uint32_t *b, uint64_t *cnt) {
    if(!h->size) return 0;
    BPEEntry *e = h->heap[0];
    *a = e->key>>32; *b = (uint32_t)e->key; *cnt = e->count;
    return 1;
}

static inline int bpe_heap_pop_max(BPEHeap *h, uint32_t *a, uint32_t *b, uint64_t *cnt) {
    if(!h->size) return 0;
    BPEEntry *top = h->heap[0];
    *a = top->key>>32; *b = (uint32_t)top->key; *cnt = top->count;
    HASH_DEL(h->map, top);
    if(--h->size) {
        h->heap[0] = h->heap[h->size];
        h->heap[0]->idx = 0;
        sift_down(h, 0);
    }
    free(top);
    return 1;
}

static inline int bpe_heap_decrement(BPEHeap *h, uint32_t a, uint32_t b) {
    uint64_t key = ((uint64_t)a<<32) | b;
    BPEEntry *e;
    HASH_FIND(hh, h->map, &key, sizeof(key), e);
    if (!e) return 0;

    if (--e->count == 0) {
        size_t idx = e->idx;
        HASH_DEL(h->map, e);
        free(e);
        if (--h->size > idx) {
            h->heap[idx] = h->heap[h->size];
            h->heap[idx]->idx = idx;
            sift_down(h, idx);
            sift_up(h, idx);
        }
    } else {
        sift_down(h, e->idx);
    }
    return 1;
}


static inline void bpe_heap_free(BPEHeap *h) {
    BPEEntry *e, *tmp;
    HASH_ITER(hh, h->map, e, tmp) {
        HASH_DEL(h->map, e);
        free(e);
    }
    free(h->heap);
    h->heap = NULL; h->size = h->cap = 0;
}

#endif // HEAP_IMPLEMENTATION
#endif // BPE_HEAP_H

