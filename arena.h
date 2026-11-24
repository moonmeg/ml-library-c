#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Arena{
    size_t size;
    size_t capacity;
    uint8_t *data;
} Arena;

static inline Arena arena_init(size_t capacity){
    Arena a = {0, capacity, malloc (capacity)};
    
    if(a.data == NULL){ a.capacity = 0; }

    return a;
}

static inline void* arena_alloc(Arena *a, size_t size){
    size = (size + 7) & ~(size_t)7;

    if(a-> size + size > a-> capacity){
        return NULL;
    }
    void *ptr = &a->data[a-> size];
    a->size += size;

    return ptr;
}

static inline void arena_free(Arena *a){
    free(a->data);
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}
#endif