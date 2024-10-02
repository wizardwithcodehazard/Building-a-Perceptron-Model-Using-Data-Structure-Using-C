# Building a Perceptron Model Using Data Structures and Algorithms in C

## Objective

The goal of this project is to create a simple perceptron model using fundamental Data Structures and Algorithms (DSA) concepts. This project aims to help students understand neural networks by applying basic DSA techniques such as arrays, matrices, loops, and arithmetic operations.

## Problem Statement

Develop a perceptron model capable of classifying data points into two categories through supervised learning. The model utilizes DSA concepts to handle input data, update weights, and predict outcomes based on input features.

## Approach

1. **Data Representation**: Use arrays to represent input features and weights.
2. **Weight Update Rule**: Implement the perceptron learning rule to adjust weights during training.
3. **Prediction**: Apply a simple activation function to predict the output based on the weighted sum of inputs.
4. **Training**: Train the model iteratively to minimize classification errors.

## Data Structures Used

- **Arrays**: To store input features, weights, and labels.
- **Matrices**: For handling multiple data points and corresponding weight updates.
- **Loops**: For iterating through the training data and updating weights.

## Activity Breakdown

### 1. Define Input and Weights Structure

```c
#define DATASET_SIZE 100  // Number of data points
#define FEATURES 2        // Number of features for each data point

double X[DATASET_SIZE][FEATURES];  // Input features
int Y[DATASET_SIZE];                // Labels (0 for Iris-setosa, 1 for Iris-versicolor)
float W[FEATURES];                  // Weights for each feature
```

### 2. Initialize Weights

```c
void initializeWeights() {
    for (int i = 0; i < FEATURES; i++) {
        W[i] = (float)rand() / (float)(RAND_MAX) - 0.5;  // Random values between -0.5 and 0.5
    }
}
```

### 3. Activation Function

```c
int activation(float weightedSum) {
    return (weightedSum >= 0) ? 1 : 0;  // Output 1 if the weighted sum is positive, else 0
}
```

### 4. Calculate Weighted Sum

```c
float calculateWeightedSum(double features[]) {
    float weightedSum = 0;
    for (int i = 0; i < FEATURES; i++) {
        weightedSum += W[i] * features[i];
    }
    return weightedSum;
}
```

### 5. Train the Perceptron

```c
void trainPerceptron(double X[][FEATURES], int Y[], int epochs, float learningRate) {
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < DATASET_SIZE; i++) {
            int prediction = activation(calculateWeightedSum(X[i]));
            int error = Y[i] - prediction;

            // Update weights based on the error
            for (int j = 0; j < FEATURES; j++) {
                W[j] += learningRate * error * X[i][j];
            }
        }
    }
}
```

### 6. Test the Model

```c
int testPerceptron(double features[]) {
    return activation(calculateWeightedSum(features));
}
```

## Additional Enhancements

1. **Performance Metrics**: Implement accuracy, precision, and recall to evaluate the performance of the perceptron model on the test data.
2. **Visualize Decision Boundary**:
   
    ![image](https://github.com/user-attachments/assets/c2b8ca21-a4b4-410f-83f7-ab65a3aef034)

4. **Dataset**: Load a simple dataset, such as the Iris dataset, filtering out **Iris-virginica**, to train and test the perceptron.

## Training Output Example

During training, the output of the weights is printed after certain epochs, providing insight into the learning process:

```
Epoch: 0, Weights: [0.190000, -0.030000]
Epoch: 10, Weights: [1.080000, -0.860000]
Epoch: 20, Weights: [1.500000, -1.840000]
Epoch: 30, Weights: [1.920000, -2.670000]
Epoch: 40, Weights: [2.290000, -3.450000]
Epoch: 50, Weights: [2.660000, -4.040000]
Epoch: 60, Weights: [2.990000, -4.460000]
Epoch: 70, Weights: [2.970000, -5.010000]
Epoch: 80, Weights: [3.440000, -5.200000]
Epoch: 90, Weights: [3.530000, -5.630000]
Predicted class for input (5.00, 3.50): 0
```

## Conclusion

This project demonstrates how DSA concepts such as arrays, loops, and matrices can be applied to build a simple perceptron model. Through this project, students will gain a hands-on understanding of how basic neural networks work and how they can be trained to classify data.

By implementing a perceptron using these foundational structures, students bridge the gap between traditional DSA techniques and modern machine learning applications.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Building-a-Perceptron-Model-Using-Data-Structure-Using-C.git
   ```

2. Navigate into the project directory:
   ```bash
   cd Building-a-Perceptron-Model-Using-Data-Structure-Using-C
   ```

3. Compile the C program:
   ```bash
   gcc -o perceptron perceptron.c
   ```

4. Run the executable:
   ```bash
   ./perceptron
   ```

## License

This project is licensed under the GPL License. See the [LICENSE](LICENSE) file for more information.

