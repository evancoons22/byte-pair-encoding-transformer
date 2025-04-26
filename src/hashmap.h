#ifndef HASHMAP_H
#define HASHMAP_H

#include <stdio.h> 
#include <stdlib.h> 
#include <stdint.h>

#define SIZE 31

// Key structure to hold a pair of uint32_t values
typedef struct {
    uint32_t first;
    uint32_t second;
} KeyPair;

typedef struct Node { 
    KeyPair key;      // Changed from uint32_t to KeyPair
    uint32_t value;
    struct Node* next;
} Node;

typedef struct {
    Node **buckets;
} HashMap;

uint32_t pair_hash(KeyPair key);
HashMap* hashmap_create();
void hashmap_insert(HashMap* hashmap, KeyPair key, uint32_t value);
int hashmap_get(HashMap* hashmap, KeyPair key, int* found);
int hashmap_remove(HashMap* hashmap, KeyPair key, int* removed);
void freeHashMap(HashMap *map);

#ifdef HASHMAP_IMPLEMENTATION

// Hash function for a pair of uint32_t values
uint32_t pair_hash(KeyPair key) {
    // Combine the two values to create a single hash
    uint64_t combined = ((uint64_t)key.first << 32) | key.second;
    
    // Apply a hash function to the combined value
    uint32_t hash = (uint32_t)(combined ^ (combined >> 32));
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;
    
    return hash;
}

// Helper function to check if two KeyPairs are equal
int key_equals(KeyPair a, KeyPair b) {
    return (a.first == b.first) && (a.second == b.second);
}

HashMap* hashmap_create() { 
    HashMap* map = malloc(sizeof(HashMap)); 
    map->buckets = calloc(SIZE, sizeof(Node*));
    return map;
} 

void hashmap_insert(HashMap* hashmap, KeyPair key, uint32_t value) { 
    // Get the hash and bucket index
    unsigned int idx = pair_hash(key) % SIZE;
    Node* node = hashmap->buckets[idx];
    
    // Check if key already exists
    while(node) { 
        if (key_equals(node->key, key)) { 
            node->value = value; 
            return;
        } 
        node = node->next;
    } 
    
    // Create a new node
    Node* new_node = malloc(sizeof(Node));
    new_node->key = key;
    new_node->value = value;
    
    // Insert at the beginning of the bucket
    new_node->next = hashmap->buckets[idx];
    hashmap->buckets[idx] = new_node;
} 

int hashmap_get(HashMap* hashmap, KeyPair key, int* found) { 
    unsigned int idx = pair_hash(key) % SIZE;
    Node* node = hashmap->buckets[idx];
    
    while(node) { 
        if (key_equals(node->key, key)) {  
            *found = 1;
            return node->value;
        } else { 
            node = node->next;
        } 
    } 
    
    *found = 0;
    return 0;
} 

int hashmap_remove(HashMap* hashmap, KeyPair key, int* removed) {
    unsigned int idx = pair_hash(key) % SIZE;
    Node* node = hashmap->buckets[idx];
    Node* prev = NULL;
    *removed = 0;
    
    while (node) {
        if (key_equals(node->key, key)) {
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
