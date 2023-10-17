#include <stdio.h>
#include <math.h>
#include<ctype.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_HMC5883_U.h>
#define NUM_FEATURES 3
float output[25][3];
Adafruit_HMC5883_Unified mag = Adafruit_HMC5883_Unified(12345);

void setup() {
  Serial.begin(9600);
}
void fun_print(float output[25][3])
{
        delay(40);
        Serial.print("X: "); Serial.print(output[24][0]); Serial.print("  ");
        Serial.print("Y: "); Serial.print(output[24][1]); Serial.print("  ");
        Serial.print("Z: "); Serial.print(output[24][2]); Serial.print("  ");
        delay(1000);
  
}
float tanh_activation(float x)
{
    return tanh(x);
}
void loop() {
  float vector[25][3]={0};
  sensors_event_t event;
  int i;
  float a[25],b[25],c[25];
  /*for(i=0;i<25;i++)
  {
    mag.getEvent(&event);
    vector[i][0]=event.magnetic.x;
    vector[i][1]=event.magnetic.y;
    vector[i][2]=event.magnetic.z;
    delay(2000);
  }
  for(i=0;i<25;i++)
  {
    Serial.print(vector[i][0]);
    Serial.print(' ');
    Serial.print(vector[i][1]);
    Serial.print(' ');
    Serial.print(vector[i][2]);
    Serial.print('\n');

  }*/
  for(i=0;i<25;i++)
  { 
    mag.getEvent(&event);
    a[i]=event.magnetic.x;
    b[i]=event.magnetic.y;
    c[i]=event.magnetic.z;
    vector[i][0]=a[i];
    vector[i][1]=b[i];
    vector[i][2]=c[i];
    delay(1000);
  }
  for(i=0;i<25;i++)
  {
    Serial.print(vector[i][0]);
    Serial.print(" ");
    Serial.print(vector[i][1]);
    Serial.print(" ");
    Serial.print(vector[i][2]);
    Serial.print("\n");
  }

    float weights_input[3] = {0.5, 0.5, 0.5};
    float weights_n[3] = {0.7, 0.6, 0.5};
    float weights_o[3] = {1.1, 1.1, 1.1};
        // Temporary hidden layer activations
    float h[3] = {0, 0, 0};
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
        for (int i = 0; i < 25; i++)
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
        loss /= 25;
    }
    // Output array to store the predictions
    for (int i = 0; i < 25; i++)
    {
        float mul[3];

        // Calculate the element-wise multiplication of input data and weights_input
        for (int j = 0; j < 3; j++)
        {
            mul[j] = vector[i][j] * weights_input[j];
            delay(10);
        }

        float fun[3];
        // Calculate the sum of the element-wise multiplication of weights_n and previous hidden layer activations (h) and add it to the previous calculation
        for (int j = 0; j < 3; j++)
        {
            fun[j] = weights_n[j] * h[j] + mul[j];
            delay(10);
        }

        float hidden_activation[3];
        // Apply the hyperbolic tangent (tanh) activation function to each element of the "fun" array, obtaining the new hidden layer activations
        for (int j = 0; j < 3; j++)
        {
            hidden_activation[j] = tanh_activation(fun[j]);
            delay(10);
        }

        // Calculate the final output for the data point using element-wise multiplication of weights_o and the updated hidden layer activations
        for (int j = 0; j < 3; j++)
        {
            output[i][j] = weights_o[j] * hidden_activation[j];
            delay(10);
        }

        // Update the temporary hidden layer activations for the next iteration
        for (int j = 0; j < 3; j++)
        {
            h[j] = hidden_activation[j];
            delay(10);
        }
    }
    fun_print(output);

}
