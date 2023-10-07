/*
Very simple implementation of a Neural Network with
a single Neuron,
a single input, and
a single output,
with the goal of learning y=2*x from 5 training samples.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Training data - last column is target
float train[][2] = {
  {0, 0},
  {1, 2},
  {2, 4},
  {3, 6},
  {4, 8},
};
const size_t N_TRAIN = sizeof(train)/sizeof(train[0]);

// Random float [0.0, 1.0]
float rand_float(void) {
  return (float)rand() / (float)RAND_MAX;
}

// Forward pass
float forward(float w, float b, float x) {
  return w * x + b;
}

// Cost function - mean squared error
float mse(float w, float b) {
  float sq_errs = 0;
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x = train[i][0];
    float y = train[i][1];
    float y_hat = forward(w, b, x);
    float d = y - y_hat;
    sq_errs += d * d;
  }
  return sq_errs / N_TRAIN;
}

// Finite difference (poor mans derivative)
const float EPS = 1e-3;
float finite_diff_w(float w, float b) {
  return (mse(w+EPS, b) - mse(w, b)) / EPS;
}
float finite_diff_b(float w, float b) {
  return (mse(w, b+EPS) - mse(w, b)) / EPS;
}

// Run model on data
void evaluate(float w, float b) {
  printf("Model Evaluation:\n");
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x = train[i][0];
    float y = train[i][1];
    float y_hat = forward(w, b, x);
    printf("%f -> %f\n", x, y_hat);
  }
}

// Entry-point
int main(void) {

  // seed rand()
  // srand(0);
  srand(time(0));

  // Create model - init weights
  printf(" --- Init Model ---\n");
  float w = rand_float();
  float b = rand_float();
  evaluate(w, b);
  printf("(0) c=%f\n", mse(w, b));
  printf("   w=%f, b=%f\n", w, b);


  // set training parameters
  const size_t N_EPOCHS = 10*1000;
  const float LEARNING_RATE = 1e-3;

  printf("\n --- Train Model ---\n");
  // Train model - optimize weights
  for (size_t i=1; i<=N_EPOCHS; i++) {
    // compute finite difference
    float dw = finite_diff_w(w, b);
    float db = finite_diff_b(w, b);

    // compute step
    float sw = -dw * LEARNING_RATE;
    float sb = -db * LEARNING_RATE;

    // update weights
    w += sw;
    b += sb;

    // print stuff
    if (i % (N_EPOCHS/10) == 0 || i == 1) {
      printf("(%lu) c=%f\n", i, mse(w, b));
      printf("   dw=%f, sw=%f, w=%f\n",dw, sw, w);
      printf("   db=%f, sb=%f, b=%f\n",db, sb, b);
    }
  }

  printf("\n --- Final Model ---\n");
  printf("w=%f, b=%f\n", w, b);
  printf("c=%f\n", mse(w, b));
  evaluate(w, b);

}
