#include <stdio.h>
#include <math.h>

#define NUM_DATA_POINTS 60
#define NUM_FEATURES 3

// Helper function for the hyperbolic tangent activation function
double tanh_activation(double x)
{
    return tanh(x);
}

int main()
{
    float vector[60][3];
    FILE *fp = fopen("data.txt", "r");
    for (int i = 0; i < 60; i++)
    {
        fscanf(fp, "%f %f %f", &vector[i][0], &vector[i][1], &vector[i][2]);
    }
    fclose(fp);
    // Weights initialization
    double weights_input[NUM_FEATURES] = {0.50, 0.5, 0.5};
    double weights_n[NUM_FEATURES] = {0.7, 0.6, 0.5};
    double weights_o[NUM_FEATURES] = {1.1, 1.1, 1.1};

    // Temporary hidden layer activations
    double h[NUM_FEATURES] = {0, 0, 0};
    // Output array to store the predictions
    double final[NUM_FEATURES];
    //
    //
    //
    //
    //
    //
    // updating weights for minimum loss
    float learning_rate = 0.001;
    int test = 60;
    float loss = 0;
    for (int k = 0; k < test; k++)
    {
        for (int i = 0; i < 60; i++)
        {
            double mul[NUM_FEATURES];

            // Calculate the element-wise multiplication of input data and weights_input
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                mul[j] = vector[i][j] * weights_input[j];
            }

            double fun[NUM_FEATURES];
            // Calculate the sum of the element-wise multiplication of weights_n and previous hidden layer activations (h) and add it to the previous calculation
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                fun[j] = weights_n[j] * h[j] + mul[j];
            }

            double hidden_activation[NUM_FEATURES];
            // Apply the hyperbolic tangent (tanh) activation function to each element of the "fun" array, obtaining the new hidden layer activations
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                hidden_activation[j] = tanh_activation(fun[j]);
            }

            // Calculate the final output for the data point using element-wise multiplication of weights_o and the updated hidden layer activations
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                final[j] = weights_o[j] * hidden_activation[j];
            }

            // Update the temporary hidden layer activations for the next iteration
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                h[j] = hidden_activation[j];
            }
            float data_loss, loss;
            data_loss = (pow((final[0] - vector[i][0]), 2) + pow(final[1] - vector[i][1], 2) + pow(final[2] - vector[i][2], 2)) / 3;
            loss += data_loss;
            float d_output[3];
            float d_activation[3];
            float d_fun[3];
            for (int j = 0; j < 3; j++)
            {
                d_output[j] = 2 * (final[j] - vector[i][j]);
            }
            for (int j = 0; j < 3; j++)
            {
                d_activation[j] = d_output[j] * weights_o[j];
            }
            for (int j = 0; j < 3; j++)
            {
                d_fun[j] = d_activation[j] * (1 - pow(h[j], 2));
            }
            for (int j = 0; j < 3; j++)
            {
                weights_n[j] -= learning_rate * d_fun[j] * h[j];
            }
            for (int j = 0; j < 3; j++)
            {
                weights_input[j] -= learning_rate * d_fun[j] * vector[i][j];
            }
            for (int j = 0; j < 3; j++)
            {
                weights_o[j] -= learning_rate * d_output[j] * h[j];
            }
        }
        loss /= 60;
    }
    //
    //
    //
    //
    //
    //
    // Perform the forward propagation

    double output[NUM_DATA_POINTS][NUM_FEATURES];
    for (int i = 0; i < NUM_DATA_POINTS; i++)
    {
        double mul[NUM_FEATURES];

        // Calculate the element-wise multiplication of input data and weights_input
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            mul[j] = vector[i][j] * weights_input[j];
        }

        double fun[NUM_FEATURES];
        // Calculate the sum of the element-wise multiplication of weights_n and previous hidden layer activations (h) and add it to the previous calculation
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            fun[j] = weights_n[j] * h[j] + mul[j];
        }

        double hidden_activation[NUM_FEATURES];
        // Apply the hyperbolic tangent (tanh) activation function to each element of the "fun" array, obtaining the new hidden layer activations
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            hidden_activation[j] = tanh_activation(fun[j]);
        }

        // Calculate the final output for the data point using element-wise multiplication of weights_o and the updated hidden layer activations
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            output[i][j] = weights_o[j] * hidden_activation[j];
        }

        // Update the temporary hidden layer activations for the next iteration
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            h[j] = hidden_activation[j];
        }

        // Print the prediction for the current data point
        printf("Prediction %d: [%.3f, %.3f, %.3f]\n", i + 1, output[i][0], output[i][1], output[i][2]);
    }

    // Print the output array
    printf("\nOutput array:\n");
    for (int i = 0; i < NUM_DATA_POINTS; i++)
    {
        printf("[%.3f, %.3f, %.3f]\n", output[i][0], output[i][1], output[i][2]);
    }
    for (int j = 0; j < 3; j++)
    {

        printf("%0.2f %0.2f %0.2f\n", weights_input[j], weights_n[j], weights_o[j]);
    }
    return 0;
}
