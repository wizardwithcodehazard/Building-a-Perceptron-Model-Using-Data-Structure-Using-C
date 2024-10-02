#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Define constants
#define LEARNING_RATE 0.1
#define EPOCHS 100
#define DATASET_SIZE 101 // Increased to accommodate one extra user input
#define FEATURES 2

// Structure to store a data point (features and label)
typedef struct {
    double features[FEATURES];
    int label; // 0 for Iris-setosa, 1 for Iris-versicolor
} DataPoint;

// Function prototypes
void trainPerceptron(DataPoint dataset[], double weights[], int epochs, int dataset_size);
int predict(double features[], double weights[]);
void updateWeights(double weights[], double features[], int error);
void exportWeights(double weights[]);
void loadData(DataPoint dataset[]);

// Load the dataset
void loadData(DataPoint dataset[]) {
    double raw_data[DATASET_SIZE - 1][FEATURES + 1] = { // Adjusted for zero-indexing
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

    for (int i = 0; i < DATASET_SIZE - 1; i++) {
        dataset[i].features[0] = fabs(raw_data[i][0]); // First feature (Sepal Length)
        dataset[i].features[1] = fabs(raw_data[i][1]); // Second feature (Sepal Width)
        dataset[i].label = (int)raw_data[i][2]; // Label (0 or 1)
    }
}

// Main Perceptron Training Function
void trainPerceptron(DataPoint dataset[], double weights[], int epochs, int dataset_size) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < dataset_size; i++) {
            int prediction = predict(dataset[i].features, weights);
            int error = dataset[i].label - prediction;
            if (error != 0) {
                updateWeights(weights, dataset[i].features, error);
            }
        }
        if (epoch % 10 == 0) {
            printf("Epoch: %d, Weights: [%f, %f]\n", epoch, weights[0], weights[1]);
        }
    }
}

// Prediction Function
int predict(double features[], double weights[]) {
    double weighted_sum = 0.0;
    for (int i = 0; i < FEATURES; i++) {
        weighted_sum += features[i] * weights[i];
    }
    return (weighted_sum >= 0) ? 1 : 0; // Class 1 if weighted sum >= 0, else Class 0
}

// Function to Update Weights
void updateWeights(double weights[], double features[], int error) {
    for (int i = 0; i < FEATURES; i++) {
        weights[i] += LEARNING_RATE * error * features[i];
    }
}

// Function to export weights to a CSV file
void exportWeights(double weights[]) {
    FILE *fptr = fopen("weights.csv", "w");
    if (fptr == NULL) {
        printf("Error opening file for writing!\n");
        return;
    }

    fprintf(fptr, "Feature1 Weight,Feature2 Weight\n");
    fprintf(fptr, "%f,%f\n", weights[0], weights[1]);
    fclose(fptr);
}

// Main function
int main() {
    DataPoint dataset[DATASET_SIZE];
    double weights[FEATURES] = {0.0, 0.0}; // Initialize weights

    // Load the dataset
    loadData(dataset);

    // Train the perceptron model
    trainPerceptron(dataset, weights, EPOCHS, DATASET_SIZE - 1); // Only use original dataset size

    // Export initial weights
    exportWeights(weights);

    // Test the model with a new input
    double new_input[FEATURES] = {5.0, 3.5}; // Example test data
    int result = predict(new_input, weights);
    printf("Predicted class for input (%.2f, %.2f): %d\n", new_input[0], new_input[1], result);
    printf("Class: %s\n", (result == 0) ? "Iris-setosa" : "Iris-versicolor");

    // --- New Section for User Input ---
    char continue_prediction = 'y';

    while (continue_prediction == 'y' || continue_prediction == 'Y') {
        double user_input[FEATURES];

        // Ask user for Sepal Length and Sepal Width
        printf("Enter Sepal Length and Sepal Width for the flower:\n");
        printf("Sepal Length: ");
        scanf("%lf", &user_input[0]);
        printf("Sepal Width: ");
        scanf("%lf", &user_input[1]);

        user_input[0] = fabs(user_input[0]);
        user_input[1] = fabs(user_input[1]);

        // Predict based on user input
        int user_result = predict(user_input, weights);
        printf("Predicted class for input (%.2f, %.2f): %d\n", user_input[0], user_input[1], user_result);
        printf("Class: %s\n", (user_result == 0) ? "Iris-setosa" : "Iris-versicolor");

        // Ask the user if they want to predict another flower
        printf("Do you want to predict another flower? (y/n): ");
        scanf(" %c", &continue_prediction);

        // Optionally retrain with the new user input
        if (continue_prediction == 'y' || continue_prediction == 'Y') {
            printf("Would you like to retrain the model using your input? (y/n): ");
            char retrain_choice;
            scanf(" %c", &retrain_choice);

            if (retrain_choice == 'y' || retrain_choice == 'Y') {
                // Add the new data point to the dataset
                DataPoint new_data_point;
                new_data_point.features[0] = user_input[0];
                new_data_point.features[1] = user_input[1];
                new_data_point.label = user_result; // Assume user-provided label is correct

                // Retrain the model with the updated dataset
                dataset[DATASET_SIZE - 1] = new_data_point; // Add it to the last position
                trainPerceptron(dataset, weights, EPOCHS, DATASET_SIZE); // Train again with updated dataset

                // Export weights again
                exportWeights(weights);
                printf("Model retrained with your input and weights updated.\n");
            }
        }
    }

    printf("Program terminated. Thank you!\n");
    return 0;
}
