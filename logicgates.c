/*
Very simple implementation of a Neural Network with
a single Neuron,
two inputs, and
a single output,
with the goal of learning OR, AND and NAND logic gates.
It will also show, that this model is not able to model XOR.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


// Training data - last column is target
typedef float sample_t[3];

sample_t train_or[] = {
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 1},
};
sample_t train_and[] = {
  {0, 0, 0},
  {1, 0, 0},
  {0, 1, 0},
  {1, 1, 1},
};
sample_t train_nand[] = {
  {0, 0, 1},
  {1, 0, 0},
  {0, 1, 0},
  {1, 1, 0},
};
sample_t train_xor[] = {
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 0},
};

sample_t * train = train_xor;
const size_t N_TRAIN = sizeof(*train)/sizeof(*train[0]);

float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float forward(float w1, float w2, float b, float x1, float x2) {
  return sigmoidf(w1 * x1 + w2 * x2 + b);
}

float mse(float w1, float w2, float b) {
  /* Mean Squared Error */
  float sq_errs = 0.0f;
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float y_hat = forward(w1, w2, b, x1, x2);
    float d = y - y_hat;
    sq_errs += d * d;
    // printf("%f, %f, %f, %f, %f, %f\n", x1, x2, y, y_hat, d, sq_errs);
  }
  return sq_errs / N_TRAIN;
}

void evaluate(float w1, float w2, float b) {
  printf("Model Evaluation:\n");
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float y_hat = forward(w1, w2, b, x1, x2);
    printf("%f ^ %f -> %f\n", x1, x2, y_hat);
  }
}

float rand_float(void) { // [0.0 - 1.0]
  return (float)rand() / (float)RAND_MAX;
}

int main(void) {
  // random seed
  srand(time(0));

  // init weights
  printf("\n --- Init Model ---\n");
  float w1 = rand_float();
  float w2 = rand_float();
  float b = rand_float();
  printf("w1=%f, w2=%f, b=%f\n", w1, w2, b);
  evaluate(w1, w2, b);

  // set training parameters
  const size_t N_EPOCHS = 1000*1000;
  const float LEARNING_RATE = 1e-2;
  const float EPS = 1e-3;

  // optimize weights - training loop
  printf("\n --- Train Model ---\n");
  for (size_t i=1; i<=N_EPOCHS; i++) {
    // compute finite differences
    float dw1 = (mse(w1+EPS, w2, b) - mse(w1, w2, b)) / EPS;
    float dw2 = (mse(w1, w2+EPS, b) - mse(w1, w2, b)) / EPS;
    float db = (mse(w1, w2, b+EPS) - mse(w1, w2, b)) / EPS;

    // compute steps
    float sw1 = -dw1 * LEARNING_RATE;
    float sw2 = -dw2 * LEARNING_RATE;
    float sb = -db * LEARNING_RATE;

    // update weights
    w1 += sw1;
    w2 += sw2;
    b += sb;

    // print stuff
    if (i % (N_EPOCHS/10) == 0 || i == 1) {
      printf("(%lu) mse=%f\n", i, mse(w1, w2, b));
      // printf("   dw1=%f, sw1=%f, w1=%f\n",dw1, sw1, w1);
      // printf("   dw2=%f, sw2=%f, w2=%f\n",dw2, sw2, w2);
      // printf("   db=%f, sb=%f, b=%f\n",db, sb, b);
    }

  } // end training loop

  printf("\n --- Final Model ---\n");
  printf("w1=%f, w2=%f, b=%f\n", w1, w2, b);
  printf("c=%f\n", mse(w1, w2, b));
  evaluate(w1, w2, b);

}
