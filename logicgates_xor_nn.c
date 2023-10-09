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

const size_t STRIDE = 3;
const size_t N_SAMPLES = sizeof(TRAIN_OR)/sizeof(TRAIN_OR[0])/STRIDE;

typedef struct {
  Matrix x_in; // input h0
  Matrix w1, b1, h1, gw1, gb1;
  Matrix w2, b2, y_out, gw2, gb2;
} ModelXOR;

ModelXOR model_init() {
  ModelXOR m;
  // layer 0 / input
  m.x_in = mat_alloc(1,2);
  mat_fill(m.x_in, 0);
  // layer 1
  m.w1 = mat_alloc(2,2);
  m.gw1 = mat_alloc(2,2);
  m.b1 = mat_alloc(1,2);
  m.gb1 = mat_alloc(1,2);
  m.h1 = mat_alloc(1,2);
  mat_rand(m.w1, 0, 1);
  mat_fill(m.gw1, 0);
  mat_rand(m.b1, 0, 1);
  mat_fill(m.gb1, 0);
  mat_fill(m.h1, 0);
  // layer 2 
  m.w2 = mat_alloc(2,1);
  m.gw2 = mat_alloc(2,1);
  m.b2 = mat_alloc(1,1);
  m.gb2 = mat_alloc(1,1);
  m.y_out = mat_alloc(1,1);
  mat_rand(m.w2, 0, 1);
  mat_fill(m.gw2, 0);
  mat_rand(m.b2, 0, 1);
  mat_fill(m.gb2, 0);
  mat_fill(m.y_out, 0);
  return m;
}

void model_print_w(ModelXOR model) {
  MAT_PRINT(model.w1);
  MAT_PRINT(model.b1);
  MAT_PRINT(model.w2);
  MAT_PRINT(model.b2);
}

void model_print_g(ModelXOR model) {
  MAT_PRINT(model.gw1);
  MAT_PRINT(model.gb1);
  MAT_PRINT(model.gw2);
  MAT_PRINT(model.gb2);
}

void model_print_h(ModelXOR model) {
  MAT_PRINT(model.x_in);
  MAT_PRINT(model.h1);
  MAT_PRINT(model.y_out);
}

void model_forward(ModelXOR m){
  // forward x through layer 1 -> h1
  mat_mul_mat(m.h1, m.x_in, m.w1);
  mat_add_mat(m.h1, m.b1);
  mat_sigmoid(m.h1);
  // forward h1 trough layer 2 -> h2
  mat_mul_mat(m.y_out, m.h1, m.w2);
  mat_add_mat(m.y_out, m.b2);
  mat_sigmoid(m.y_out);
  // output y_hat = *m.h2.p_data
}

float squared_error(float y, float y_hat) {
  float d = y - y_hat;
  return d * d;
}

float cost(const ModelXOR m, const Matrix x, const Matrix y) {
  assert(x.num_rows == y.num_rows); // number of samples
  assert(y.num_cols == m.y_out.num_cols); // target and model output dimension
  size_t n_samples = x.num_rows;

  float sq_err = 0;
  for (size_t i=0; i<n_samples; ++i) {
    Matrix x_i = mat_row(x, i);
    Matrix y_i = mat_row(y, i);
    mat_copy(m.x_in, x_i);
    model_forward(m);
    size_t y_dim = y.num_cols;
    for (size_t j=0; j<y_dim; ++j) {
      sq_err += squared_error(MAT_AT(y_i, 0, j), MAT_AT(m.y_out, 0, j));
    }
  }
  return sq_err / n_samples;
}

void compute_finite_diffs(ModelXOR m, const Matrix x, const Matrix y, const float eps) {
  float saved;
  const float mse = cost(m, x, y);

  for (size_t i=0; i<m.w1.num_rows; ++i) {
    for (size_t j=0; j<m.w1.num_cols; ++j) {
      saved = MAT_AT(m.w1, i, j);
      MAT_AT(m.w1, i, j) += eps;
      MAT_AT(m.gw1, i, j) = (cost(m, x, y)-mse)/eps;
      MAT_AT(m.w1, i, j) = saved;
    }
  }

  for (size_t i=0; i<m.b1.num_rows; ++i) {
    for (size_t j=0; j<m.b1.num_cols; ++j) {
      saved = MAT_AT(m.b1, i, j);
      MAT_AT(m.b1, i, j) += eps;
      MAT_AT(m.gb1, i, j) = (cost(m, x, y)-mse)/eps;
      MAT_AT(m.b1, i, j) = saved;
    }
  }

  for (size_t i=0; i<m.w2.num_rows; ++i) {
    for (size_t j=0; j<m.w2.num_cols; ++j) {
      saved = MAT_AT(m.w2, i, j);
      MAT_AT(m.w2, i, j) += eps;
      MAT_AT(m.gw2, i, j) = (cost(m, x, y)-mse)/eps;
      MAT_AT(m.w2, i, j) = saved;
    }
  }

  for (size_t i=0; i<m.b2.num_rows; ++i) {
    for (size_t j=0; j<m.b2.num_cols; ++j) {
      saved = MAT_AT(m.b2, i, j);
      MAT_AT(m.b2, i, j) += eps;
      MAT_AT(m.gb2, i, j) = (cost(m, x, y)-mse)/eps;
      MAT_AT(m.b2, i, j) = saved;
    }
  }
}

void model_update(ModelXOR m, float lr) {
  for (size_t i=0; i<m.w1.num_rows; ++i) {
    for (size_t j=0; j<m.w1.num_cols; ++j) {
      MAT_AT(m.w1, i, j) -= lr * MAT_AT(m.gw1, i, j);
    }
  }
  
  for (size_t i=0; i<m.b1.num_rows; ++i) {
    for (size_t j=0; j<m.b1.num_cols; ++j) {
      MAT_AT(m.b1, i, j) -= lr * MAT_AT(m.gb1, i, j);
    }
  }
  
  for (size_t i=0; i<m.w2.num_rows; ++i) {
    for (size_t j=0; j<m.w2.num_cols; ++j) {
      MAT_AT(m.w2, i, j) -= lr * MAT_AT(m.gw2, i, j);
    }
  }
  
  for (size_t i=0; i<m.b2.num_rows; ++i) {
    for (size_t j=0; j<m.b2.num_cols; ++j) {
      MAT_AT(m.b2, i, j) -= lr * MAT_AT(m.gb2, i, j);
    }
  }
}

void model_eval(ModelXOR m, Matrix x_eval, Matrix y_eval) {
  size_t n_samples = x_eval.num_rows;
  for (size_t i=0; i<n_samples; ++i) {
    mat_copy(m.x_in, mat_row(x_eval, i));
    model_forward(m);
    printf(
      "%d ^ %d = %f  (%d)\n",
      (int)MAT_AT(x_eval, i, 0),
      (int)MAT_AT(x_eval, i, 1),
      MAT_AT(m.y_out, 0, 0),
      (int)MAT_AT(y_eval, i, 0)
    );
  }
}

int main(void) {
  srand(time(0));

  // prepare data
  printf("\n ----------------------------- ");
  printf("\n --- PREPARE TRAINING DATA --- ");
  printf("\n ----------------------------- \n");
  float * training_data = TRAIN_XOR;
  printf("N_TRAIN: %d\n", (int)N_SAMPLES);

  Matrix x_train = (Matrix){
    .num_rows = N_SAMPLES,
    .num_cols = 2,
    .stride = STRIDE,
    .p_data = training_data,
  };
  MAT_PRINT(x_train);

  Matrix y_train = (Matrix){
    .num_rows = N_SAMPLES,
    .num_cols = 1,
    .stride = STRIDE,
    .p_data = training_data + 2,
  };
  MAT_PRINT(y_train);


  // init model
  ModelXOR m = model_init();


  // eval untrained model
  printf("\n ------------------------- ");
  printf("\n --- INITIALISED MODEL --- ");
  printf("\n ------------------------- \n");
  model_print_w(m);
  printf("---------------------\n");
  model_eval(m, x_train, y_train);

  // training params
  const size_t EPOCHS = 100*1000;
  const float LR = 1e-1;
  const float EPS = 1e-1;

  compute_finite_diffs(m, x_train, y_train, EPS);
  model_update(m, LR);

  // training loop
  printf("\n --------------------- ");
  printf("\n --- TRAINING LOOP --- ");
  printf("\n --------------------- \n");
  for (size_t i=0; i<EPOCHS; ++i) {
    compute_finite_diffs(m, x_train, y_train, EPS);
    model_update(m, LR);
    if (0 || i % (EPOCHS/10)==0) {
      printf("(%zu) mse = %f\n", i, cost(m, x_train, y_train));
    }
  }


  // eval model after training
  printf("\n ---------------------- ");
  printf("\n --- AFTER TRAINING --- ");
  printf("\n ---------------------- \n");
  model_print_w(m);
  printf("---------------------\n");
  model_print_g(m);
  printf("---------------------\n");
  printf("mse = %f\n", cost(m, x_train, y_train));
  model_eval(m, x_train, y_train);
}
