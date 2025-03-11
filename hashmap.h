#ifndef HASHMAP_H
#define HASHMAP_H

#include <stdio.h> 
#include <stdlib.h> 

#define SIZE 31

typedef struct Node { 
    uint32_t key;
    uint32_t value;
    struct Node* next;
} Node;

typedef struct {
    Node **buckets;
} HashMap;

uint32_t int_hash(uint32_t key);
HashMap* hashmap_create();
void hashmap_insert(HashMap* hashmap, uint32_t key, uint32_t value);
int hashmap_get(HashMap* hashmap, uint32_t key, int* found);
int hashmap_remove(HashMap* hashmap, uint32_t key, int* removed);
void freeHashMap(HashMap *map);

#ifdef HASHMAP_IMPLEMENTATION

uint32_t int_hash(uint32_t key) {
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key;
}


HashMap* hashmap_create() { 
    HashMap* map = malloc(sizeof(HashMap)); 
    map->buckets = calloc(SIZE, sizeof(Node*));
    return map;
} 

void hashmap_insert(HashMap* hashmap, uint32_t key, uint32_t value) { 
    // first get the hash 
    unsigned int idx = int_hash(key) % SIZE;
    Node* node = hashmap->buckets[idx];
    while(node) { 
        if (node->key == key) { 
            node->value = value; 
            return;
        } 
        node = node->next;
    } 
    Node* new_node = malloc(sizeof(Node));
    new_node->key = key;
    new_node->value = value;
    // whatever is in the bucket right now, put new_node in the first space and move the rest over
    new_node->next = hashmap->buckets[idx];
    hashmap->buckets[idx] = new_node;

} 

int hashmap_get(HashMap* hashmap, uint32_t key, int* found) { 
    unsigned int idx = int_hash(key) % SIZE;
    Node* node = hashmap->buckets[idx];
    while(node) { 
        if (node->key == key) {  
            *found = 1;
            return node->value;
        } else { 
            node = node->next;
        } 
    } 
    *found = 0;
    return 0;
} 

int hashmap_remove(HashMap* hashmap, const uint32_t key, int* removed) {
    unsigned int idx = int_hash(key) % SIZE;
    Node* node = hashmap->buckets[idx];
    Node* prev = NULL;
    *removed = 0;
    while (node) {
        if (node->key == key) {
            if (prev) {
                prev->next = node->next;
            } else {
                hashmap->buckets[idx] = node->next;
            }
            free(node);
            *removed = 1;
            return 0;  
        }
        prev = node;
        node = node->next;
    }
    return 0;
}

void freeHashMap(HashMap *map) {
    for (int i = 0; i < SIZE; i++) {
        Node *node = map->buckets[i];
        while (node) {
            Node *temp = node;
            node = node->next;
            free(temp);
        }
    }
    free(map->buckets);
    free(map);
}


#endif //HASHMAP_IMPLEMENTATION
#endif //HASHMAP_H
