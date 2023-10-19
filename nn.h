/********/
/* nn.h */
/********/

#ifndef NN_H
#define NN_H

#include <math.h>   // expf
#include <stddef.h> // size_t
#include <stdio.h>  // printf
#include <string.h> // strlen

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

// --------------------------------------------------------------

#define ARRAY_LEN(arr) sizeof(arr) / sizeof(arr[0])

float rand_float();
void shuffle_array(size_t *array, size_t n);
float squared_error(float y_pred, float y_true);
float squared_error_derivative(float y_pred, float y_true);
float sigmoid(float x);
float sigmoid_derivative(float x);

typedef enum {
  // Implemented activation functions
  IDENTITY = 0,
  SIGMOID = 1,
} Sigma;

float sigma(float x, Sigma f);
float sigma_derivative(float x, Sigma f);

// --------------------------------------------------------------

typedef struct {
  size_t num_rows;
  size_t num_cols;
  size_t stride;
  float *p_data;
} Matrix;

#define MAT_AT(mat, row, col) mat.p_data[(row) * (mat).stride + (col)]

Matrix mat_alloc(size_t num_rows, size_t num_cols);
void mat_print(Matrix m, const char *name, size_t offset_left);
#define MAT_PRINT(m) mat_print(m, #m, 0)

void mat_fill(Matrix m, float x);
void mat_rand(Matrix m, float min, float max);
Matrix mat_row(Matrix m, size_t row);
void mat_copy(Matrix dst, Matrix m);
// void mat_trp(Matrix m);
void mat_add_num(Matrix m, float x);
void mat_add_mat(Matrix a, Matrix b);
void mat_mul_num(Matrix m, float x);
void mat_mul_mat(Matrix dst, Matrix a, Matrix b);
void mat_sigmoid(Matrix m);

// --------------------------------------------------------------

typedef struct {
  // The input layer (index 0 of the arrays) does not use weights, biases,
  // weight_grads or bias_grads. These elements still get allocated because
  // it makes indexing these arrays by layer more coherent.
  size_t n_layers;
  Matrix *weighted_sums; // array of Vectors; z = w*a_prev + b
  Matrix *activations;   // array of Vectors; a = sigma(z)
  Matrix *weights;       // array of Matrices
  Matrix *weight_grads;  // array of Matrices
  Matrix *biases;        // array of Vectors
  Matrix *bias_grads;    // array of Vectors
  Matrix *errors;        // array of Vectors
  Matrix loss_step;      // Vector
  Matrix loss_batch;     // Vector
  Matrix loss_epoch;     // Vector
  Sigma s_hidden;        // activation function
  Sigma s_output;        // activation function
} NN;

typedef enum {
  EGD = 1, // Epoch (Batch) Gradient Descent
  BGD = 2, // (Mini) Batch Gradient Descent
  SGD = 3, // Stochastic Gradient Descent
} GD_Type;

typedef struct {
  float lr;
  size_t epochs;
  size_t batch_size;
  GD_Type gd_type;
} TrainParams;

#define NN_X_IN(nn) (nn).activations[0]
#define NN_Y_OUT(nn) (nn).activations[(nn).n_layers - 1]

NN nn_create(size_t *layer_dims, size_t n_layers, Sigma s_hidden,
             Sigma s_output);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_PRINT_WEIGHTS(nn) nn_print_weights(nn, #nn)
#define NN_PRINT_ACTS(nn) nn_print_acts(nn, #nn)
#define NN_PRINT_GRADS(nn) nn_print_grads(nn, #nn)
#define NN_PRINT_LOSS(nn, gd_type) nn_print_loss(nn, gd_type)

void nn_rand(NN m, const float min, const float max);
void nn_set_input_layer_activations(NN nn, Matrix x, size_t s);
void nn_forward(NN nn);
void nn_update_losses(NN nn, const Matrix y, const size_t s);
void nn_clear_errors(NN nn);
void nn_set_error_at_output_layer(NN nn, const Matrix y, const size_t s);
void nn_backprop(NN nn);
void nn_update_weights(NN nn, const float lr, size_t n);

#endif // NN_H

// --------------------------------------------------------------
// --------------------------------------------------------------

/********/
/* nn.c */
/********/

#ifdef NN_IMPLEMENTATION

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

void shuffle_array(size_t *array, size_t n) {
  if (n > RAND_MAX / 10) {
    NN_ASSERT(0 && "n must be much smaller than RAND_MAX!");
  }

  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      size_t t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

float squared_error(float y_pred, float y_true) {
  float d = y_pred - y_true;
  return 0.5 * d * d;
}

float squared_error_derivative(float y_pred, float y_true) {
  return y_pred - y_true;
}

float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

float sigmoid_derivative(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float sigma(float x, Sigma f) {
  switch (f) {
  case IDENTITY:
    return x;
  case SIGMOID:
    return sigmoid(x);
  default:
    NN_ASSERT(0 && "Unreachable");
  }
}

float sigma_derivative(float x, Sigma f) {
  switch (f) {
  case IDENTITY:
    return 1.f;
  case SIGMOID:
    return sigmoid_derivative(x);
  default:
    NN_ASSERT(0 && "Unreachable");
  }
}

// --------------------------------------------------------------

Matrix mat_alloc(size_t num_rows, size_t num_cols) {
  Matrix m;
  m.num_rows = num_rows;
  m.num_cols = num_cols;
  m.stride = num_cols;
  m.p_data = NN_MALLOC(sizeof(*m.p_data) * num_rows * num_cols);
  NN_ASSERT(m.p_data != NULL);
  return m;
}

void mat_print(Matrix m, const char *name, size_t offset_left) {
  printf("%*s%s=[", (int)offset_left, "", name);
  for (size_t row = 0; row < m.num_rows; ++row) {
    if (row == 0) {
      printf("[");
    } else {
      printf("\n%*s[", (int)(offset_left + strlen(name) + 2), "");
    }
    for (size_t col = 0; col < m.num_cols; ++col) {
      printf(" %f ", MAT_AT(m, row, col));
    }
    printf("]");
  }
  printf("]\n");
}

void mat_fill(Matrix m, float x) {
  for (size_t row = 0; row < m.num_rows; ++row) {
    for (size_t col = 0; col < m.num_cols; ++col) {
      MAT_AT(m, row, col) = x;
    }
  }
}

void mat_rand(Matrix m, float min, float max) {
  for (size_t row = 0; row < m.num_rows; ++row) {
    for (size_t col = 0; col < m.num_cols; ++col) {
      MAT_AT(m, row, col) = rand_float() * (max - min) + min;
    }
  }
}

Matrix mat_row(Matrix m, size_t row) {
  return (Matrix){
      .num_rows = 1,
      .num_cols = m.num_cols,
      .stride = m.stride,
      .p_data = &MAT_AT(m, row, 0),
  };
}

void mat_copy(Matrix dst, Matrix m) {
  NN_ASSERT(dst.num_rows == m.num_rows);
  NN_ASSERT(dst.num_cols == m.num_cols);
  for (size_t row = 0; row < m.num_rows; ++row) {
    for (size_t col = 0; col < m.num_cols; ++col) {
      MAT_AT(dst, row, col) = MAT_AT(m, row, col);
    }
  }
}

void mat_add_num(Matrix m, float x) {
  for (size_t row = 0; row < m.num_rows; ++row) {
    for (size_t col = 0; col < m.num_cols; ++col) {
      MAT_AT(m, row, col) += x;
    }
  }
}

void mat_add_mat(Matrix a, Matrix b) {
  NN_ASSERT(a.num_cols == b.num_cols);
  NN_ASSERT(a.num_rows == b.num_rows);
  for (size_t row = 0; row < a.num_rows; ++row) {
    for (size_t col = 0; col < a.num_cols; ++col) {
      MAT_AT(a, row, col) += MAT_AT(b, row, col);
    }
  }
}

void mat_mul_num(Matrix m, float x) {
  for (size_t row = 0; row < m.num_rows; ++row) {
    for (size_t col = 0; col < m.num_cols; ++col) {
      MAT_AT(m, row, col) *= x;
    }
  }
}

void mat_mul_mat(Matrix dst, Matrix a, Matrix b) {
  NN_ASSERT(a.num_cols == b.num_rows);
  size_t inner_dim = a.num_cols;

  NN_ASSERT(dst.num_rows == a.num_rows);
  NN_ASSERT(dst.num_cols == b.num_cols);
  for (size_t row = 0; row < dst.num_rows; ++row) {
    for (size_t col = 0; col < dst.num_cols; ++col) {
      MAT_AT(dst, row, col) = 0;
      for (size_t i = 0; i < inner_dim; ++i) {
        MAT_AT(dst, row, col) += (MAT_AT(a, row, i) * MAT_AT(b, i, col));
      }
    }
  }
}

void mat_sigmoid(Matrix m) {
  for (size_t row = 0; row < m.num_rows; ++row) {
    for (size_t col = 0; col < m.num_cols; ++col) {
      MAT_AT(m, row, col) = sigmoid(MAT_AT(m, row, col));
    }
  }
}

// --------------------------------------------------------------

NN nn_create(size_t *layer_dims, size_t n_layers, Sigma s_hidden,
             Sigma s_output) {
  NN_ASSERT(n_layers > 0);

  // init NN struct
  NN nn;
  nn.n_layers = n_layers;
  nn.s_hidden = s_hidden;
  nn.s_output = s_output;

  // malloc arrays to hold matrices
  nn.weighted_sums = malloc(n_layers * sizeof(*nn.weighted_sums));
  NN_ASSERT(nn.weighted_sums != NULL);
  nn.activations = malloc(n_layers * sizeof(*nn.activations));
  NN_ASSERT(nn.activations != NULL);
  nn.weights = malloc(n_layers * sizeof(*nn.weights));
  NN_ASSERT(nn.weights != NULL);
  nn.weight_grads = malloc(n_layers * sizeof(*nn.weight_grads));
  NN_ASSERT(nn.weight_grads != NULL);
  nn.biases = malloc(n_layers * sizeof(*nn.biases));
  NN_ASSERT(nn.biases != NULL);
  nn.bias_grads = malloc(n_layers * sizeof(*nn.bias_grads));
  NN_ASSERT(nn.bias_grads != NULL);
  nn.errors = malloc(n_layers * sizeof(*nn.errors));
  NN_ASSERT(nn.errors != NULL);

  // malloc matrices in the arrays
  for (size_t i = 0; i < n_layers; ++i) {
    // Vectors
    nn.activations[i] = mat_alloc(1, layer_dims[i]);
    if (i == 0) {
      // input layer does not need any of these matrices/vectors, but same
      // length arrays make indexing by layer much simpler and coherent.
      Matrix dummy = mat_alloc(0, 0);
      nn.weighted_sums[i] = dummy;
      nn.weights[i] = dummy;
      nn.weight_grads[i] = dummy;
      nn.biases[i] = dummy;
      nn.bias_grads[i] = dummy;
      nn.errors[i] = dummy;
    } else {
      // Matrices
      nn.weights[i] = mat_alloc(layer_dims[i - 1], layer_dims[i]);
      nn.weight_grads[i] = mat_alloc(layer_dims[i - 1], layer_dims[i]);
      // Vectors
      nn.weighted_sums[i] = mat_alloc(1, layer_dims[i]);
      nn.biases[i] = mat_alloc(1, layer_dims[i]);
      nn.bias_grads[i] = mat_alloc(1, layer_dims[i]);
      nn.errors[i] = mat_alloc(1, layer_dims[i]);
    }

    // malloc a vector to store the loss
    nn.loss_step = mat_alloc(1, layer_dims[n_layers - 1]);
    nn.loss_batch = mat_alloc(1, layer_dims[n_layers - 1]);
    nn.loss_epoch = mat_alloc(1, layer_dims[n_layers - 1]);
  }

  return nn;
}

void nn_print(NN nn, const char *name) {
  char buf[256];
  printf("%s = [\n", name);
  // snprintf(buf, sizeof(buf), "z%zu", (size_t)0);
  // mat_print(nn.weighted_sums[0], buf, 2);
  // snprintf(buf, sizeof(buf), "a%zu", (size_t)0);
  // mat_print(nn.activations[0], buf, 2);
  // for (size_t i = 1; i < nn.n_layers; ++i) {
  for (size_t i = 0; i < nn.n_layers; ++i) {
    printf("\n");
    snprintf(buf, sizeof(buf), " w%zu", i);
    mat_print(nn.weights[i], buf, 2);
    snprintf(buf, sizeof(buf), " b%zu", i);
    mat_print(nn.biases[i], buf, 2);
    snprintf(buf, sizeof(buf), " z%zu", i);
    mat_print(nn.weighted_sums[i], buf, 2);
    snprintf(buf, sizeof(buf), " a%zu", i);
    mat_print(nn.activations[i], buf, 2);
    snprintf(buf, sizeof(buf), " e%zu", i);
    mat_print(nn.errors[i], buf, 2);
    snprintf(buf, sizeof(buf), "dw%zu", i);
    mat_print(nn.weight_grads[i], buf, 2);
    snprintf(buf, sizeof(buf), "db%zu", i);
    mat_print(nn.bias_grads[i], buf, 2);
  }

  snprintf(buf, sizeof(buf), "ls%zu", nn.n_layers - 1);
  mat_print(nn.loss_step, buf, 2);
  printf("]\n");
}

void nn_print_weights(NN nn, const char *name) {
  char buf[256];
  printf("%s(weights) = [\n", name);
  for (size_t i = 1; i < nn.n_layers; ++i) {
    snprintf(buf, sizeof(buf), "w%zu", i);
    mat_print(nn.weights[i], buf, 2);
    snprintf(buf, sizeof(buf), "b%zu", i);
    mat_print(nn.biases[i], buf, 2);
  }
  printf("]\n");
}

void nn_print_acts(NN nn, const char *name) {
  char buf[256];
  printf("%s(activations) = [\n", name);
  for (size_t i = 0; i < nn.n_layers; ++i) {
    if (i > 0) {
      snprintf(buf, sizeof(buf), "z%zu", i);
      mat_print(nn.weighted_sums[i], buf, 2);
    }
    snprintf(buf, sizeof(buf), "a%zu", i);
    mat_print(nn.activations[i], buf, 2);
  }
  printf("]\n");
}

void nn_print_grads(NN nn, const char *name) {
  char buf[256];
  printf("%s(gradients) = [\n", name);
  for (size_t i = 1; i < nn.n_layers; ++i) {
    snprintf(buf, sizeof(buf), "gw%zu", i);
    mat_print(nn.weight_grads[i], buf, 2);
    snprintf(buf, sizeof(buf), "gb%zu", i);
    mat_print(nn.bias_grads[i], buf, 2);
  }
  printf("]\n");
}

void nn_print_loss(NN nn, const GD_Type gd_type) {
  switch (gd_type) {
  case EGD:
    printf("Epoch Loss: [ ");
    break;
  case BGD:
    printf("Batch Loss: [ ");
    break;
  case SGD:
    printf("Step Loss: [ ");
    break;
  default:
    NN_ASSERT(0);
  }
  size_t output_dim = nn.activations[nn.n_layers - 1].num_rows;
  for (size_t i = 0; i < output_dim; ++i) {
    switch (gd_type) {
    case EGD:
      printf("%f", MAT_AT(nn.loss_epoch, 0, i));
      break;
    case BGD:
      printf("%f", MAT_AT(nn.loss_batch, 0, i));
      break;
    case SGD:
      printf("%f", MAT_AT(nn.loss_step, 0, i));
      break;
    default:
      NN_ASSERT(0);
    }
    if (i < output_dim - 1) {
      printf(", ");
    }
  }
  printf(" ]\n");
}

void nn_rand(NN nn, float min, float max) {
  for (size_t i = 0; i < nn.n_layers; ++i) {
    mat_rand(nn.weights[i], min, max);
    mat_rand(nn.biases[i], min, max);
  }
}

void nn_train_loop(NN nn, Matrix x, Matrix y, TrainParams p) {
  const size_t n_samples = x.num_rows;
  printf("Training Samples in Epoch: %zu\n", n_samples);

  // set n_batches and batch_size
  size_t n_batches;
  size_t batch_size;
  switch (p.gd_type) {
  case SGD:
    printf("Step-GD: Update weights on each step.\n");
    n_batches = n_samples;
    batch_size = 1;
    break;
  case BGD:
    printf("Batch-GD: Update weights on each batch.\n");
    n_batches = (size_t)(n_samples / p.batch_size);
    batch_size = p.batch_size;
    break;
  case EGD:
    printf("Epoch-GD: Update weights on each epoch.\n");
    n_batches = 1;
    batch_size = n_samples;
    break;
  }
  printf("Batch size: %zu -> %zu batches per epoch (%zu samples skipped)\n\n",
         batch_size, n_batches, n_samples % batch_size);

  // create map for accessing samples in a shuffled manner
  size_t sample_map[n_samples];
  for (size_t i = 0; i < n_samples; ++i) {
    sample_map[i] = i;
  }

  // epoch loop
  for (size_t e = 0; e < p.epochs; ++e) {
    mat_fill(nn.loss_epoch, 0);
    shuffle_array(sample_map, n_samples);

    // batch loop
    for (size_t b = 0; b < n_batches; ++b) {
      mat_fill(nn.loss_batch, 0);

      // sample loop
      for (size_t ss = 0; ss < batch_size; ++ss) {
        // get sample index
        size_t s = sample_map[ss+b*batch_size];

        // forward pass a single sample
        nn_set_input_layer_activations(nn, x, s);
        nn_forward(nn);
        nn_update_losses(nn, y, s);

        // backprop errors and compute gradients
        nn_clear_errors(nn);
        nn_set_error_at_output_layer(nn, y, s);
        nn_backprop(nn);

        if (p.gd_type == SGD) {
          nn_update_weights(nn, p.lr, 1);
          // printf("[%zu] ", s);
          // NN_PRINT_LOSS(nn, SGD);
        }

      } // sample loop
      if (p.gd_type == BGD) {
        nn_update_weights(nn, p.lr, batch_size);
        // printf("[%zu] ", b);
        // NN_PRINT_LOSS(nn, BGD);
      }

    } // batch loop
    if (p.gd_type == EGD) {
      nn_update_weights(nn, p.lr, n_samples);
    }
    if ((p.epochs > 100 && e % (int)(p.epochs / 25)) == 0 ||
        e == p.epochs - 1) {
      printf("[%zu] ", e);
      NN_PRINT_LOSS(nn, EGD);
    }

  } // epoch loop
}

void nn_set_input_layer_activations(NN nn, Matrix x, size_t s) {
  for (size_t i = 0; i < x.num_cols; ++i) { // set input layer activations
    MAT_AT(nn.activations[0], 0, i) = MAT_AT(x, s, i);
  }
}

void nn_forward(NN nn) {
  // activations of input layer must be set to sample already!

  // for layer l in [1, 2, ..., L-1]
  for (size_t l = 0; l < nn.n_layers - 1; ++l) {

    // for i in layer l+1 neurons
    for (size_t i = 0; i < nn.activations[l + 1].num_cols; ++i) {
      // set weighted sum to bias: z_i = b_i
      MAT_AT(nn.weighted_sums[l + 1], 0, i) = MAT_AT(nn.biases[l + 1], 0, i);

      // for j in layer l neurons
      for (size_t j = 0; j < nn.activations[l].num_cols; ++j) {
        float w_ji = MAT_AT(nn.weights[l + 1], j, i);
        float a_j = MAT_AT(nn.activations[l], 0, j);
        // add weighted activations z_i+=sum(w_ji*aj)
        MAT_AT(nn.weighted_sums[l + 1], 0, i) += w_ji * a_j;
      }

      // a_i=sigma_i(z_i)
      if (l >= nn.n_layers - 2) {
        // activate ouput layer
        MAT_AT(nn.activations[l + 1], 0, i) =
            sigma(MAT_AT(nn.weighted_sums[l + 1], 0, i), nn.s_output);
      } else {
        // activate hidden layers
        MAT_AT(nn.activations[l + 1], 0, i) =
            sigma(MAT_AT(nn.weighted_sums[l + 1], 0, i), nn.s_hidden);
      }
    }
  }
}

void nn_update_losses(const NN nn, const Matrix y, size_t s) {
  NN_ASSERT(y.num_cols == NN_Y_OUT(nn).num_cols);

  for (size_t j = 0; j < y.num_cols; ++j) {
    float a_L = MAT_AT(NN_Y_OUT(nn), 0, j);
    float y_true = MAT_AT(y, s, j);
    float sq_err = squared_error(a_L, y_true);
    MAT_AT(nn.loss_step, 0, j) = sq_err;
    MAT_AT(nn.loss_batch, 0, j) += sq_err;
    MAT_AT(nn.loss_epoch, 0, j) += sq_err;
  }
}

void nn_clear_errors(NN nn) {
  for (size_t i = 1; i < nn.n_layers; ++i) {
    mat_fill(nn.errors[i], 0.f);
  }
}

void nn_set_error_at_output_layer(const NN nn, const Matrix y, size_t s) {
  NN_ASSERT(y.num_cols == NN_Y_OUT(nn).num_cols);

  /*********************************************
   * e_i[L]=sigma_out'(z_i[L])*(a_i[L]-y_true) *
   *********************************************/
  for (size_t j = 0; j < y.num_cols; ++j) {
    float a_L = MAT_AT(NN_Y_OUT(nn), 0, j);
    float z_L = MAT_AT(nn.weighted_sums[nn.n_layers - 1], 0, j);
    float y_true = MAT_AT(y, s, j);
    float sq_err_prime = squared_error_derivative(a_L, y_true);
    float sigma_prime = sigma_derivative(z_L, nn.s_output);
    MAT_AT(nn.errors[nn.n_layers - 1], 0, j) += sq_err_prime * sigma_prime;
  }
}

void nn_backprop(NN nn) {
  // errors at output layers must set be already!

  // backpropagate error backwards from last to second layer
  /*************************************************
   * e_j[l]=sigma'(z_j[l])*SUM{w_ji[l+1]*e_i[l+1]} *
   *************************************************/
  const size_t L = nn.n_layers - 1;

  // for layer l in [L-1, L-2, ..., 2]
  for (size_t l = L - 1; l > 0; --l) {

    // for j in layer l neurons
    for (size_t j = 0; j < nn.activations[l].num_cols; ++j) {

      // for i in layer l+1 neurons
      for (size_t i = 0; i < nn.activations[l + 1].num_cols; ++i) {
        float e_i = MAT_AT(nn.errors[l + 1], 0, i);
        float w_ji = MAT_AT(nn.weights[l + 1], j, i);
        // sum(w_ji*e_i)
        MAT_AT(nn.errors[l], 0, j) += w_ji * e_i;
      }

      float z_j = MAT_AT(nn.weighted_sums[l], 0, j);
      float ds_z_j = sigma_derivative(z_j, nn.s_hidden);
      // multiply sum(w_ji*e_i) with sigma'(z_j)
      MAT_AT(nn.errors[l], 0, j) *= ds_z_j;
    }
  }

  // compute gradients
  /*******************************
   * dE/dw_ij(k)=e_j(k)*a_i(k-1) *
   *******************************/

  // for layer l in [1, 2, ..., L]
  for (size_t l = 1; l < nn.n_layers; ++l) {

    // for j in layer l neurons
    for (size_t j = 0; j < nn.activations[l].num_cols; ++j) {
      float e_j = MAT_AT(nn.errors[l], 0, j);

      // for i in layer neurons
      for (size_t i = 0; i < nn.activations[l - 1].num_cols; ++i) {
        float a_i = MAT_AT(nn.activations[l - 1], 0, i);
        float dw_ij = a_i * e_j;
        // weight gradients at layer l
        MAT_AT(nn.weight_grads[l], i, j) += dw_ij;
      }

      // bias gradients at layer l
      MAT_AT(nn.bias_grads[l], 0, j) += e_j;
    }
  }
}

void nn_update_weights(NN nn, float lr, size_t n) {
  for (size_t l = 1; l < nn.n_layers; ++l) {
    for (size_t i = 0; i < nn.weights[l].num_rows; ++i) {
      for (size_t j = 0; j < nn.weights[l].num_cols; ++j) {
        MAT_AT(nn.weights[l], i, j) -=
            lr * MAT_AT(nn.weight_grads[l], i, j) / n;
        MAT_AT(nn.weight_grads[l], i, j) = 0.f;
      }
    }
    for (size_t j = 0; j < nn.biases[l].num_cols; ++j) {
      MAT_AT(nn.biases[l], 0, j) -= lr * MAT_AT(nn.bias_grads[l], 0, j) / n;
      MAT_AT(nn.bias_grads[l], 0, j) = 0.f;
    }
  }
}

#endif // NN_IMPLEMENTATION
