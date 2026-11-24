#ifndef ML_H
#define ML_H

#include <stddef.h>
#include <stdint.h>
#include "arena.h"
#include <math.h>

typedef uint32_t u32; //alias 

typedef struct{
    u32 rows;
    u32 cols;
    float *data; //pointer to actual array of nos.
} Matrix;


//The function: it returns a pointer to matrix.
Matrix* mat_create(Arena* arena, u32 rows, u32 cols);

void mat_copy(Matrix *dest, Matrix *src);
void mat_clear(Matrix *m);
void mat_fill(Matrix *m, float val);
void mat_add(Matrix *a, Matrix *b);
void mat_sub(Matrix *a, Matrix *b);
void mat_scale(Matrix *m, float s);
void mat_mul(Matrix *dest, Matrix *a, Matrix *b);
void mat_relu(Matrix *m);
void mat_sigmoid(Matrix *m);
void mat_rand(Matrix *m, float low, float high);
void mat_print(Matrix *m, const char *name);

#endif