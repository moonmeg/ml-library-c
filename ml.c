#include "ml.h"
#include "arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


Matrix* mat_create(Arena* arena, u32 rows, u32 cols){
    //1. Allocate the matrix ("box") itself inside the arena.
    Matrix *m = arena_alloc(arena, sizeof(Matrix));

    //2. Set the dimensions
    m->rows = rows;
    m->cols = cols;

    //3. Allocate actual array of floats inside arena
    m->data = arena_alloc(arena, sizeof(float) * rows * cols);

    return m;
}

//Allows to save a state of matrix
void mat_copy(Matrix *dest, Matrix *src){
    for(u32 i = 0; i < src->rows * src->cols; ++i){
        dest->data[i] = src->data[i];
    }
}

//Clears the matrix to all zeros.
void mat_clear(Matrix *m){
    for(u32 i = 0; i < m->rows * m->cols; ++i){
        m->data[i] = 0.0f;
    }
}

void mat_fill(Matrix *m, float val){
    for(u32 i = 0; i < m->rows * m-> cols; ++i){
        m->data[i] = val;
    }
}

//Matrix Addition
void mat_add(Matrix *a, Matrix *b){
    for(u32 i = 0; i < a->rows * a-> cols; ++i){
        //a->rows*a->cols - total num of elements in the matrix
        a->data[i] += b->data[i];
    }
}

//Matrix subtraction
void mat_sub(Matrix *a, Matrix *b){
    for(u32 i = 0; i < a->rows * a-> cols; ++i){
        a->data[i] -= b->data[i];
    }
}

//Matrix Scaling
void mat_scale(Matrix *m, float s){
    for(u32 i = 0; i < m->rows * m->cols; ++i){
        m->data[i] *= s;
    }
}

//Matrix Multiplication
void mat_mul(Matrix *dest, Matrix *a, Matrix *b){
    for(u32 i = 0; i < a->rows; ++i){
        for(u32 j = 0; j < b->cols; ++j){
            float sum = 0.0f;
            for(u32 k = 0; k < a->cols; ++k){
                //Mat mul: row of A * column of B
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            //Store result in destination(dest)
            dest->data[i * dest->cols + j] = sum;
        }
    }
}

//Matrix relu - Rectified Linear Unit
void mat_relu(Matrix *m){
    for(u32 i = 0; i < m->rows * m->cols; ++i){
        if(m->data[i] < 0){
            m->data[i] = 0;
        }
    }
}

//Matrix Sigmoid
void mat_sigmoid(Matrix *m){
    for(u32 i = 0; i < m->rows * m->cols; ++i){
        //sigmoid = 1 / (1 + e^-x)
        m->data[i] = 1.0f / (1.0f + expf(- m->data[i])); 
    }
}

void mat_print(Matrix *m, const char *name){
    printf("%s = [\n", name);
    for(u32 i = 0; i < m->rows; ++i){
        printf(" ");
        for(u32 j = 0; j < m->cols; ++j){
            printf("%f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_rand(Matrix *m, float low, float high){
    for(u32 i = 0; i < m->rows * m-> cols; ++i){
        //generate float num b/w 0 and 1
        float r = (float) rand()/ (float) RAND_MAX;
        m->data[i] = low + r * (high - low);
    }
}