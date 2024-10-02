#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Model parameters
#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define FEATURES 2
#define THRESHOLD 0.5
#define DATASET_SIZE 100

// Data structure for Iris data points
typedef struct {
    double features[FEATURES];
    int label; // 0 for Iris-setosa, 1 for Iris-versicolor
} DataPoint;

// Function prototypes
void trainPerceptron(DataPoint* dataset, double weights[], int epochs, int dataset_size);
double predictRaw(double features[], double weights[]);
int predict(double features[], double weights[]);
void updateWeights(double weights[], double features[], int error);
void exportWeights(double weights[], const char* filename);
void loadData(DataPoint* dataset, int dataset_size);
void shuffleDataset(DataPoint* dataset, int dataset_size);
void normalizeInput(double* features);
void getUserInput(double* user_input);
void handleUserPrediction(double* weights, DataPoint* dataset, int* dataset_size);

int main() {
    srand(time(NULL)); // Seed the random number generator
    int dataset_size = DATASET_SIZE;
    DataPoint* dataset = (DataPoint*)malloc(dataset_size * sizeof(DataPoint));
    double weights[FEATURES] = {0.0, 0.0};

    if (dataset == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    loadData(dataset, dataset_size);
    shuffleDataset(dataset, dataset_size);

    trainPerceptron(dataset, weights, EPOCHS, dataset_size);
    exportWeights(weights, "weights.csv");

    // Test with the problematic input
    double test_input[FEATURES] = {4.3, 2.3};
    normalizeInput(test_input);
    double raw_prediction = predictRaw(test_input, weights);
    int prediction = predict(test_input, weights);
    printf("Test prediction for (4.3, 2.3): Raw = %f, Class = %d (%s)\n", 
           raw_prediction, prediction, prediction == 0 ? "Iris-setosa" : "Iris-versicolor");

    handleUserPrediction(weights, dataset, &dataset_size);

    free(dataset);
    return 0;
}

// Load Iris dataset from raw data
void loadData(DataPoint* dataset, int dataset_size) {
    // Raw Iris dataset
    double raw_data[DATASET_SIZE][FEATURES + 1] = {
        // ... (same data as before)
        {5.1, 3.5, 0}, {4.9, 3.0, 0}, {4.7, 3.2, 0}, {4.6, 3.1, 0}, {5.0, 3.6, 0},
        {5.4, 3.9, 0}, {4.6, 3.4, 0}, {5.0, 3.4, 0}, {4.4, 2.9, 0}, {4.9, 3.1, 0},
        {5.4, 3.7, 0}, {4.8, 3.4, 0}, {4.8, 3.0, 0}, {4.3, 3.0, 0}, {5.8, 4.0, 0},
        {5.7, 4.4, 0}, {5.4, 3.9, 0}, {5.1, 3.5, 0}, {5.7, 3.8, 0}, {5.1, 3.8, 0},
        {5.4, 3.4, 0}, {5.1, 3.7, 0}, {4.6, 3.6, 0}, {5.1, 3.3, 0}, {4.8, 3.4, 0},
        {5.0, 3.0, 0}, {5.0, 3.4, 0}, {5.2, 3.5, 0}, {5.2, 3.4, 0}, {4.7, 3.2, 0},
        {4.8, 3.1, 0}, {5.4, 3.4, 0}, {5.2, 4.1, 0}, {5.5, 4.2, 0}, {4.9, 3.1, 0},
        {5.0, 3.2, 0}, {5.5, 3.5, 0}, {4.9, 3.1, 0}, {4.4, 3.0, 0}, {5.1, 3.4, 0},
        {5.0, 3.5, 0}, {4.5, 2.3, 0}, {4.4, 3.2, 0}, {5.0, 3.5, 0}, {5.1, 3.8, 0},
        {4.8, 3.0, 0}, {5.1, 3.8, 0}, {4.6, 3.2, 0}, {5.3, 3.7, 0}, {5.0, 3.3, 0},
        {7.0, 3.2, 1}, {6.4, 3.2, 1}, {6.9, 3.1, 1}, {5.5, 2.3, 1}, {6.5, 2.8, 1},
        {5.7, 2.8, 1}, {6.3, 3.3, 1}, {4.9, 2.4, 1}, {6.6, 2.9, 1}, {5.2, 2.7, 1},
        {5.0, 2.0, 1}, {5.9, 3.0, 1}, {6.0, 2.2, 1}, {6.1, 2.9, 1}, {5.6, 2.9, 1},
        {6.7, 3.1, 1}, {5.6, 3.0, 1}, {5.8, 2.7, 1}, {6.2, 2.2, 1}, {5.6, 2.5, 1},
        {5.9, 3.2, 1}, {6.1, 2.8, 1}, {6.3, 2.5, 1}, {6.1, 2.8, 1}, {6.4, 2.9, 1},
        {6.6, 3.0, 1}, {6.8, 2.8, 1}, {6.7, 3.0, 1}, {6.0, 2.9, 1}, {5.7, 2.6, 1},
        {5.5, 2.4, 1}, {5.5, 2.4, 1}, {5.8, 2.7, 1}, {6.0, 2.7, 1}, {5.4, 3.0, 1},
        {6.0, 3.4, 1}, {6.7, 3.1, 1}, {6.3, 2.3, 1}, {5.6, 3.0, 1}, {5.5, 2.5, 1},
        {5.5, 2.6, 1}, {6.1, 3.0, 1}, {5.8, 2.6, 1}, {5.0, 2.3, 1}, {5.6, 2.7, 1},
        {5.7, 3.0, 1}, {5.7, 2.9, 1}, {6.2, 2.9, 1}, {5.1, 2.5, 1}, {5.7, 2.8, 1}
    };

    for (int i = 0; i < dataset_size; i++) {
        dataset[i].features[0] = fabs(raw_data[i][0]); // Sepal Length
        dataset[i].features[1] = fabs(raw_data[i][1]); // Sepal Width
        dataset[i].label = (int)raw_data[i][2]; // Label (0 or 1)
    }
}

// Shuffle dataset for randomized training
void shuffleDataset(DataPoint* dataset, int dataset_size) {
    for (int i = dataset_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        DataPoint temp = dataset[i];
        dataset[i] = dataset[j];
        dataset[j] = temp;
    }
}

// Train Perceptron model
void trainPerceptron(DataPoint* dataset, double weights[], int epochs, int dataset_size) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        int correct_predictions = 0;
        for (int i = 0; i < dataset_size; i++) {
            double raw_prediction = predictRaw(dataset[i].features, weights);
            int prediction = (raw_prediction >= THRESHOLD) ? 1 : 0;
            int error = dataset[i].label - prediction;
            if (error == 0) {
                correct_predictions++;
            } else {
                updateWeights(weights, dataset[i].features, error);
            }
        }
        if (epoch % 100 == 0) {
            printf("Epoch: %d, Accuracy: %.2f%%, Weights: [%f, %f]\n", 
                   epoch, (double)correct_predictions / dataset_size * 100, weights[0], weights[1]);
        }
    }
}

// Predict raw output using Perceptron model
double predictRaw(double features[], double weights[]) {
    double weighted_sum = 0.0;
    for (int i = 0; i < FEATURES; i++) {
        weighted_sum += features[i] * weights[i];
    }
    return weighted_sum;
}

// Predict class label using Perceptron model
int predict(double features[], double weights[]) {
    return (predictRaw(features, weights) >= THRESHOLD) ? 1 : 0;
}

// Update weights based on prediction error
void updateWeights(double weights[], double features[], int error) {
    for (int i = 0; i < FEATURES; i++) {
        weights[i] += LEARNING_RATE * error * features[i];
    }
}

// Export trained weights to file
void exportWeights(double weights[], const char* filename) {
    FILE* fptr = fopen(filename, "w");
    if (fptr == NULL) {
        perror("Error opening file for writing");
        return;
    }
    fprintf(fptr, "Feature1 Weight,Feature2 Weight\n");
    fprintf(fptr, "%f,%f\n", weights[0], weights[1]);
    fclose(fptr);
}

// Normalize input features
void normalizeInput(double* features) {
    for (int i = 0; i < FEATURES; i++) {
        features[i] = fabs(features[i]); // Ensure all features are non-negative
    }
}

// Get user input for prediction
void getUserInput(double* user_input) {
    printf("Enter Sepal Length: ");
    scanf("%lf", &user_input[0]);
    printf("Enter Sepal Width: ");
    scanf("%lf", &user_input[1]);
    normalizeInput(user_input);
}

// Handle user prediction and retraining
void handleUserPrediction(double* weights, DataPoint* dataset, int* dataset_size) {
    char continue_prediction = 'y';
    while (continue_prediction == 'y' || continue_prediction == 'Y') {
        double user_input[FEATURES];
        getUserInput(user_input);

        double raw_prediction = predictRaw(user_input, weights);
        int user_result = predict(user_input, weights);
        printf("Predicted class for input (%.2f, %.2f): Raw = %f, Class = %d\n", 
               user_input[0], user_input[1], raw_prediction, user_result);
        printf("Class: %s\n", (user_result == 0) ? "Iris-setosa" : "Iris-versicolor");
        
        printf("Would you like to retrain the model using your input? (y/n): ");
        char retrain_choice;
        scanf(" %c", &retrain_choice);

        if (retrain_choice == 'y' || retrain_choice == 'Y') {
            // Add the new data point to the dataset
            DataPoint new_data_point;
            new_data_point.features[0] = user_input[0];
            new_data_point.features[1] = user_input[1];
            new_data_point.label = user_result; // Assume user-provided label is correct

            // Replace the last data point with the new one
            dataset[*dataset_size - 1] = new_data_point; 
            trainPerceptron(dataset, weights, EPOCHS, *dataset_size); // Train again with updated dataset

            // Export weights again
            exportWeights(weights, "weights.csv");
            printf("Model retrained with your input and weights updated.\n");
        }

        printf("Do you want to predict another flower? (y/n): ");
        scanf(" %c", &continue_prediction);
    }
}
