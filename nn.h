/********/
/* nn.h */
/********/

#ifndef NN_H
#define NN_H

#include <stddef.h> // size_t
#include <stdio.h> // printf
#include <math.h> // expf

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

// --------------------------------------------------------------

#define ARRAY_LEN(arr) sizeof(arr)/sizeof(arr[0])

float rand_float();
float sigmoid(float x);

// --------------------------------------------------------------

typedef struct {
  size_t num_rows;
  size_t num_cols;
  size_t stride;
  float *p_data;
} Matrix;

#define MAT_AT(mat, row, col) mat.p_data[(row)*(mat).stride + (col)]

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
  size_t n_layers; // number of layers; [x_in, h(1), h(2), y_out] -> 4 layers
  Matrix *h; // layer states; n_layer matrices
  Matrix *w; // weight matrices; n_layers-1 matrices
  Matrix *gw; // gradient matrices; n_layers-1 matrices
  Matrix *b; // bias vectors; n_layers-1 vectors
  Matrix *gb; // gradient vectors; n_layers-1 vectors
} NN;

NN nn_alloc(size_t *layer_dims, size_t n_layers);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)

#endif // NN_H


// --------------------------------------------------------------
// --------------------------------------------------------------


/********/
/* nn.c */
/********/

#ifdef NN_IMPLEMENTATION

float rand_float(void) {
  return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

// --------------------------------------------------------------

Matrix mat_alloc(size_t num_rows, size_t num_cols) {
  Matrix m;
  m.num_rows = num_rows;
  m.num_cols = num_cols;
  m.stride = num_cols;
  m.p_data = NN_MALLOC(sizeof(*m.p_data)*num_rows*num_cols);
  NN_ASSERT(m.p_data != NULL);
  return m;
}

void mat_print(Matrix m, const char *name, size_t offset_left) {
  printf("%*s%s = [\n", (int) offset_left, "", name);
  for (size_t row=0; row<m.num_rows; ++row) {
    printf("%*s  [", (int) offset_left, "");
    for (size_t col=0; col<m.num_cols; ++col) {
      printf(" %f ", MAT_AT(m, row, col));
    }
    printf("]\n");
  }
    printf("%*s]\n", (int) offset_left, "");
}

void mat_fill(Matrix m, float x) {
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(m, row, col) = x;
    }
  }
}

void mat_rand(Matrix m, float min, float max) {
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(m, row, col) = rand_float() * (max-min) + min;
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
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(dst, row, col) = MAT_AT(m, row, col);
    }
  }
}

void mat_add_num(Matrix m, float x) {
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(m, row, col) += x;
    }
  }
}

void mat_add_mat(Matrix a, Matrix b) {
  NN_ASSERT(a.num_cols == b.num_cols);
  NN_ASSERT(a.num_rows == b.num_rows);
  for (size_t row=0; row<a.num_rows; ++row) {
    for (size_t col=0; col<a.num_cols; ++col) {
      MAT_AT(a, row, col) += MAT_AT(b, row, col);
    }
  }
}

void mat_mul_num(Matrix m, float x) {
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(m, row, col) *= x;
    }
  }
}

void mat_mul_mat(Matrix dst, Matrix a, Matrix b) {
  NN_ASSERT(a.num_cols == b.num_rows);
  size_t inner_dim = a.num_cols;

  NN_ASSERT(dst.num_rows == a.num_rows);
  NN_ASSERT(dst.num_cols == b.num_cols);
  for (size_t row=0; row<dst.num_rows; ++row) {
    for (size_t col=0; col<dst.num_cols; ++col) {
      MAT_AT(dst, row, col) = 0;
      for (size_t i=0; i<inner_dim; ++i) {
        MAT_AT(dst, row, col) += (MAT_AT(a, row, i) * MAT_AT(b, i, col));
      }
    }
  }
}

void mat_sigmoid(Matrix m) {
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(m, row, col)=sigmoidf(MAT_AT(m, row, col));
    }
  }
}

// --------------------------------------------------------------

NN nn_alloc(size_t *layer_dims, size_t n_layers) {
  NN_ASSERT(n_layers > 0);

  printf("%p, %zu\n", layer_dims, n_layers);

  // init NN struct
  NN nn;
  nn.n_layers = n_layers;

  // malloc arrays to store matrices
  nn.h = malloc(n_layers*sizeof(*nn.h));
  NN_ASSERT(nn.h != NULL);
  nn.w = malloc((n_layers-1)*sizeof(*nn.w));
  NN_ASSERT(nn.w != NULL);
  nn.gw = malloc((n_layers-1)*sizeof(*nn.gw));
  NN_ASSERT(nn.gw != NULL);
  nn.b = malloc((n_layers-1)*sizeof(*nn.b));
  NN_ASSERT(nn.b != NULL);
  nn.gb = malloc((n_layers-1)*sizeof(*nn.gb));
  NN_ASSERT(nn.gb != NULL);

  // alloc matrices in the arrays
  for (size_t i = 0; i < n_layers; ++i) {
    nn.h[i] = mat_alloc(1, layer_dims[i]);
    if (i < n_layers-1) {
      nn.w[i] = mat_alloc(nn.h[i].num_cols, layer_dims[i+1]);
      nn.gw[i] = mat_alloc(nn.h[i].num_cols, layer_dims[i+1]);
      nn.b[i] = mat_alloc(1, layer_dims[i+1]);
      nn.gb[i] = mat_alloc(1, layer_dims[i+1]);
    }
  }

  return nn;
}

void nn_print(NN nn, const char *name) {
  char buf[256];
  printf("%s = [\n\n", name);
  for (size_t i = 0; i < nn.n_layers-1; ++i) {
    snprintf(buf, sizeof(buf), "w%zu", i);
    mat_print(nn.w[i], buf, 2);
    snprintf(buf, sizeof(buf), "b%zu", i);
    mat_print(nn.b[i], buf, 2);
    printf("\n");
  }
  printf("]\n");
}

#endif // NN_IMPLEMENTATION

