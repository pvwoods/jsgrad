A simple implementation of an autograd framework for javascript with a Multi Layer Perceptron and 2D Convolutional layer that trains against toy datasets. Based on the [lesson from Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0).

## Features

- Automatic differentiation system
- Neural network building blocks:
  - Multi-Layer Perceptron (MLP)
  - 2D Convolutional Layer (Conv2D)
- Training example with MSE loss
- Comprehensive unit tests

## MLP Example

The MLP example trains a simple neural network on a binary classification task:

```
step 0: 1.576735949332759
step 10: 0.738121537851598
step 20: 0.22560775316053272
step 30: 0.09944237732767153
step 40: 0.06201518984620813
step 50: 0.04451835351766176
step 60: 0.03445270000447273
step 70: 0.027945728570807543
step 80: 0.023411597703849537
step 90: 0.02008185196292793
```

## Conv2D Example

The Conv2D example demonstrates using 2D convolutional kernels on a 4x4 grid input:

```
Testing Conv2D on a 4x4 input:
Conv2D output shape: 2 channels, 9 values per channel

Training Conv2D for 5 steps:
step 0: 0.XXXXXX
step 1: 0.XXXXXX
step 2: 0.XXXXXX
step 3: 0.XXXXXX
step 4: 0.XXXXXX
```

The actual loss values will vary due to random initialization.

## Running Tests

The project includes unit tests for the Value class, MLP, and Conv2D components. To run the tests:

```
npm test
```

or directly with Node.js:

```
node tests.js
```

The tests verify the implementation of the basic operations, forward pass calculations, and gradient computations.