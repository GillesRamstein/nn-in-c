/*
Simple implementation of a Neural Network with
three Neurons,
two inputs, and
a single output,
with the goal of learning the OR, AND, NAND and XOR logic gate.

Using nn.h library.
*/

#define NN_IMPLEMENTATION
#include "nn.h"

#include <time.h>

// Define training data - last column is target
float TRAIN_OR[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 1,
};
float TRAIN_AND[] = {
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    1, 1, 1,
};
float TRAIN_NAND[] = {
    0, 0, 1,
    1, 0, 0,
    0, 1, 0,
    1, 1, 0,
};
float TRAIN_XOR[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0,
};
const size_t X_COLS = 2;
const size_t Y_COLS = 1;
const size_t STRIDE = X_COLS + Y_COLS;
const size_t N_SAMPLES = sizeof(TRAIN_OR) / sizeof(TRAIN_OR[0]) / STRIDE;

int main(void) {
  // set random seed
  srand(time(0));

  // setup training data
  float *training_data = TRAIN_XOR;
  printf("N_TRAIN: %zu\n", N_SAMPLES);

  Matrix x_train = (Matrix){
      .num_rows = N_SAMPLES,
      .num_cols = X_COLS,
      .stride = STRIDE,
      .p_data = training_data,
  };
  MAT_PRINT(x_train);

  Matrix y_train = (Matrix){
      .num_rows = N_SAMPLES,
      .num_cols = Y_COLS,
      .stride = STRIDE,
      .p_data = training_data + X_COLS,
  };
  MAT_PRINT(y_train);

  // define network
  size_t layer_dims[] = {2, 2, 1}; // {dim_in, [dim_h, ...], dim_out}
  Sigma s_hidden = SIGMOID;        // activation for hidden layers
  Sigma s_output = IDENTITY;       // activation for output layer

  // alloc network
  NN nn = nn_create(layer_dims, ARRAY_LEN(layer_dims), s_hidden, s_output);

  // randomize network weights
  nn_rand(nn, 0, 1);

  // setup training parameters
  const TrainParams train_params = {
      .lr = 1e-1,
      .epochs = 10000,
      .batch_size = 2, // only for BGD,
      .gd_type = SGD,
  };

  // train network
  nn_train_loop(nn, x_train, y_train, train_params);

  // eval activations
  for (size_t s = 0; s < x_train.num_rows; ++s) {
    for (size_t i = 0; i < x_train.num_cols; ++i) {
      MAT_AT(nn.activations[0], 0, i) = MAT_AT(x_train, s, i);
    }
    nn_forward(nn);
    printf("%f ^ %f -> %f\n", MAT_AT(nn.activations[0], 0, 0),
           MAT_AT(nn.activations[0], 0, 1),
           MAT_AT(nn.activations[nn.n_layers - 1], 0, 0));
  }

  return 0;
}
