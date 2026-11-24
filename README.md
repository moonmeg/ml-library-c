# ml-library-c

A small neural network library written from scratch in C, with no external dependencies beyond the standard library. It includes a custom arena memory allocator, a matrix math engine, and a 2-layer feedforward network trained on the XOR problem using gradient descent.

This is a work in progress. The goal is to understand how neural networks work underneath frameworks like PyTorch, including memory layout, matrix operations, and how gradients are computed.

## Features

- **Arena allocator** (`arena.h`): one upfront `malloc`, then fast bump-pointer allocation for all matrices. Everything is released with a single `arena_free`. Allocations are aligned to 8 bytes.
- **Matrix engine** (`ml.c`, `ml.h`): create, fill, copy, clear, add, subtract, scale, multiply, print, and randomize.
- **Activation functions**: sigmoid and ReLU.
- **Forward propagation**: input -> (matrix multiply + bias + sigmoid) -> (matrix multiply + bias + sigmoid) -> output.
- **Training with gradient descent**: gradients are estimated numerically using finite differences (nudge each parameter by a small `eps`, measure the change in cost).
- **Cost function**: mean squared error.
- **ASCII decision map**: prints the network's output across the input space.

## What it does not do (yet)

- **No backpropagation.** Gradients come from finite differences, which cost one full cost evaluation per parameter per step. This is simple and good for learning what a gradient is, but slow.
- The network architecture is hardcoded to the XOR example (2 inputs, 2 hidden neurons, 1 output).
- No dimension checks in the matrix functions yet.

## Build and run

Requires `gcc` (or any C compiler).

```bash
gcc main.c ml.c -o my_ml_lib -lm
./my_ml_lib
```

`-lm` links the math library (needed for `expf`).

## Example output

The weights start random, so exact numbers vary from run to run.

```
Initial Cost: 0.337852
Training in progress...
Iteration 0: Cost = 0.336152
Iteration 10000: Cost = 0.182090
Iteration 20000: Cost = 0.140296
Iteration 30000: Cost = 0.003525
Iteration 40000: Cost = 0.001563
Iteration 50000: Cost = 0.000994
Iteration 60000: Cost = 0.000726
Iteration 70000: Cost = 0.000570
Iteration 80000: Cost = 0.000469
Iteration 90000: Cost = 0.000397
Final Cost: 0.000345

----Result-----
Input: [0, 0] -> Predicted: 0.018811 (Target: 0)
Input: [0, 1] -> Predicted: 0.981318 (Target: 1)
Input: [1, 0] -> Predicted: 0.981316 (Target: 1)
Input: [1, 1] -> Predicted: 0.018061 (Target: 0)
```

After training, every prediction is within about 2% of its target. XOR is not linearly separable, so solving it requires the hidden layer.

Occasionally a run can get stuck in a poor local minimum because of the random initialization. If that happens, just run it again.

## Project structure

| File | Purpose |
|---|---|
| `arena.h` | Arena allocator (`arena_init`, `arena_alloc`, `arena_free`) |
| `ml.h` / `ml.c` | Matrix type and math functions, activations |
| `main.c` | XOR network: init, forward pass, cost, training, prediction, visualization |

## How training works

1. Run the forward pass on all four XOR inputs and compute the mean squared error.
2. For each weight and bias, add a small `eps`, recompute the cost, and set the gradient to `(new_cost - old_cost) / eps`.
3. Update each parameter: `param -= rate * gradient`.
4. Repeat for 100,000 iterations.

## Roadmap

- [ ] Replace numerical gradients with backpropagation and compare speed
- [ ] Add dimension assertions to matrix functions
- [ ] Check for failed allocations in `mat_create`
- [ ] Make the number of layers configurable instead of hardcoded
- [ ] Add more datasets beyond XOR
