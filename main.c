/* * PROJECT: C-ML-Library (from scratch)
 * -----------------------------------
 * - [X] Custom Arena Memory Allocator (for O(1) Matrix allocation)
 * - [X] Matrix Math Engine (Multiplication, Addition, Scaling)
 * - [X] Sigmoid Activation Logic (Non-linear mapping)
 * - [X] Numerical Gradient Descent (Finite Difference method)
 * - [X] Successfully solved the XOR Problem (Accuracy > 98%)
 * * This project demonstrates low-level memory management and 
 * fundamental implementation of backpropagation-less machine learning.
 */

#include "ml.h"
#include "arena.h"
#include <stdio.h>
#include <time.h> //for time()

typedef struct{
    Matrix *w1; //Weights for 1st layer(2x2)
    Matrix *b1; //biases for 1st layer(1x2)
    Matrix *w2; //Weights for 2nd layer(2x1)
    Matrix *b2; //Biases for 2nd layer(1x1)

    //storage for immediate calcs
    Matrix *a1; //output for 1st layer
    Matrix *a2; //final output
} Xor;

//Prototypes
Xor xor_init(Arena *arena);
void forward_xor(Xor m, Matrix *input);
float xor_cost(Xor m, Matrix *ti, Matrix *to);
void xor_learn(Xor m, Xor g, float eps, float rate, Matrix *ti, Matrix *to);
float xor_predict(Xor m, float x1, float x2, Matrix *ti);
void xor_visualize(Xor m, Matrix *ti);

//4 possible inputs XOR (00, 01, 10, 11)
float xor_input_data[] = {
    0, 0,
    0, 1,
    1, 0, 
    1, 1
};
//4 correct answers (0, 1, 1, 0)
float xor_target_data[] = {
    0,
    1, 
    1, 
    0
};

int main(){
    srand(time(NULL)); //seed the randomizer at the start of the program
    Arena arena = arena_init(1024 * 1024);

    //Initialize the network
    Xor m = xor_init(&arena);
    Xor g = xor_init(&arena);

    Matrix *ti = mat_create(&arena, 1, 2);
    Matrix *to = mat_create(&arena, 1, 1);

    float eps = 1e-1f;
    float rate = 1e-1f;

    printf("Initial Cost: %f\n", xor_cost(m, ti, to));
    printf("Training in progress...\n");

    //Training Loop (100,000 iterations)
    for(size_t i = 0; i < 100000; ++i){
        xor_learn(m, g, eps, rate, ti, to);
        //Print progress every 10,000 steps
        if(i % 10000 == 0){
            printf("Iteration %zu: Cost = %f\n", i, xor_cost(m, ti, to));
        }
    }

    printf("Final Cost: %f\n\n", xor_cost(m, ti, to));

    // Show results
    printf("\n----Result-----\n");
    for(size_t i = 0; i < 4; ++i){
        float x1 = xor_input_data[i * 2 + 0];
        float x2 = xor_input_data[i * 2 + 1];

        float out = xor_predict(m, x1, x2, ti);

        printf("Input: [%.0f, %.0f] -> Predicted: %f (Target: %.0f)\n", x1, x2, out, xor_target_data[i]);
    }

    // Decision map
    xor_visualize(m, ti);

    arena_free(&arena);
    

    return 0;
}

//Initialization Function
Xor xor_init(Arena *arena){
    Xor m;

    //Layer 1: 2 inputs -> 2 hidden neurons
    m.w1 = mat_create(arena, 2, 2);
    m.b1 = mat_create(arena, 1, 2);
    m.a1 = mat_create(arena, 1, 2);

    //Layer 2: 2 hidden neurons -> 1 output
    m.w2 = mat_create(arena, 2, 1);
    m.b2 = mat_create(arena, 1, 1);
    m.a2 = mat_create(arena, 1, 1);

    //Fill weights w/ random values for network to start guessing
    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);

    return m;
}

//Forward Pass function
void forward_xor(Xor m, Matrix *input){
    //1st layer: input * w1 + b1
    mat_mul(m.a1, input, m.w1);
    mat_add(m.a1, m.b1);
    mat_sigmoid(m.a1); //Squashing the result

    //Second layer: a1 * w2 + b2
    mat_mul(m.a2, m.a1, m.w2);
    mat_add(m.a2, m.b2);
    mat_sigmoid(m.a2); //Squashing the result b/w 0 and 1
}

//Cost Function - Updated
float xor_cost(Xor m, Matrix *ti, Matrix *to){
    float result = 0.0f;
    //size_t n = 4; //training samples

    for(size_t i = 0; i < 4; ++i){
        //1. Load input from data array
        ti->data[0] = xor_input_data[i * 2 + 0];
        ti->data[1] = xor_input_data[i * 2 + 1];

        //2. Load the expected target
        float y = xor_target_data[i];

        //3. Feed the input thru the brain
        forward_xor(m, ti);

        //4. Calculate error: (Guess - target)^2
        float d = m.a2->data[0] - y;
        result += d * d;
    }
    //Return average error - Mean Squared Error
    return result / 4.0f;
}
//Updated function
void xor_learn(Xor m, Xor g, float eps, float rate, Matrix *ti, Matrix *to){
    float c = xor_cost(m, ti, to); //current cost

    //**** CALCULATE GRADIENTS ****

    //Wiggle 1: w1
    for(u32 i = 0; i < m.w1->rows * m.w1->cols; ++i){
        float saved = m.w1->data[i];

        m.w1->data[i] +=eps;
        float nc = xor_cost(m, ti, to); //new cost
        m.w1->data[i] = saved;
        g.w1->data[i] = (nc - c) / eps; //The "Slope"
    }

    //Wiggle 2: b1
    for(u32 i = 0; i < m.b1->rows * m.b1->cols; ++i){
        float saved = m.b1->data[i];

        m.b1->data[i] += eps;
        float nc = xor_cost(m, ti, to); //New cost
        m.b1->data[i] = saved;
        g.b1->data[i] = (nc - c) / eps;
    }

    //Wiggle 3: w2
    for(u32 i = 0; i < m.w2->rows * m.w2->cols; ++i){
        float saved = m.w2->data[i];

        m.w2->data[i] += eps;
        float nc = xor_cost(m, ti, to);
        m.w2->data[i] = saved;
        g.w2->data[i] = (nc - c)/ eps;
    }

    //Wiggle 4: b2
    for(u32 i = 0; i < m.b2->rows * m.b2->cols; ++i){
        float saved = m.b2->data[i];

        m.b2->data[i] += eps;
        float nc = xor_cost(m, ti, to);
        m.b2->data[i] = saved;
        g.b2->data[i] = (nc - c) / eps;
    }
    //---- GRADIENT DESCENT -----
    //weight = weight - (learning rate * gradient)

    //Wiggle w1
    for(u32 i = 0; i < m.w1->rows * m.w1->cols; ++i){
        m.w1->data[i] -= rate * g.w1->data[i];
    }

    //Wiggle b1
    for(u32 i = 0; i < m.b1->rows * m.b1->cols; ++i){
        m.b1->data[i] -= rate * g.b1->data[i];
    }

    //Wiggle w2
    for(u32 i = 0; i < m.w2->rows * m.w2->cols; ++i){
        m.w2->data[i] -= rate * g.w2->data[i];
    }

    //Wiggle b2
    for(u32 i = 0; i < m.b2->rows * m.b2->cols; ++i){
        m.b2->data[i] -= rate * g.b2->data[i];
    }
}

//Predict Function
float xor_predict(Xor m, float x1, float x2, Matrix *ti){
    ti->data[0] = x1;
    ti->data[1] = x2;

    forward_xor(m, ti);
    return m.a2->data[0];
}

void xor_visualize(Xor m, Matrix *ti){
    printf("\n----Neural Decision Map (High = #, Low =  .) ---\n");
    for(int y = 0; y <= 10; ++y){
        for(int x = 0; x <= 20; ++x){
            float out = xor_predict(m, (float)x/20.0f, (float)y/10.0f, ti);
            if(out > 0.5f){
                printf("#");
            } else{
                printf(".");
            }
        }
        printf("\n");
    }
}