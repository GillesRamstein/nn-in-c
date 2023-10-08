
/********/
/* nn.h */
/********/

#ifndef NN_H
#define NN_H

#include <stddef.h> // size_t
#include <stdio.h> // printf

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct {
  size_t num_rows;
  size_t num_cols;
  float *p_data;
} Matrix;

#define MAT_AT(mat, row, col) mat.p_data[(row)*(mat).num_cols + (col)]

Matrix mat_alloc(size_t num_rows, size_t num_cols);
void mat_print(Matrix m);
void mat_fill(Matrix m, float val);
float rand_float();
void mat_rand(Matrix m, float min, float max);
void mat_trp(Matrix m);
void mat_add_num(Matrix a, float b);
void mat_add_mat(Matrix a, Matrix b);
void mat_mul_num(Matrix a, float b);
void mat_mul_mat(Matrix dst, Matrix a, Matrix b);

#endif // NN_H



/********/
/* nn.c */
/********/

#ifdef NN_IMPLEMENTATION

Matrix mat_alloc(size_t num_rows, size_t num_cols) {
  Matrix m;
  m.num_rows = num_rows;
  m.num_cols = num_cols;
  m.p_data = NN_MALLOC(sizeof(*m.p_data)*num_rows*num_cols);
  NN_ASSERT(m.p_data != NULL);
  return m;
}

void mat_print(Matrix m) {
  printf("[\n");
  for (size_t row=0; row<m.num_rows; ++row) {
    printf("  ");
    for (size_t col=0; col<m.num_cols; ++col) {
      printf("%f  ", MAT_AT(m, row, col));
    }
    printf("\n");
  }
    printf("]\n");
}

void mat_fill(Matrix m, float val) {
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(m, row, col) = val;
    }
  }
}

float rand_float(void) {
  return (float) rand() / (float) RAND_MAX;
}

void mat_rand(Matrix m, float min, float max) {
  for (size_t row=0; row<m.num_rows; ++row) {
    for (size_t col=0; col<m.num_cols; ++col) {
      MAT_AT(m, row, col) = rand_float() * (max-min) + min;
    }
  }
}

void mat_add_num(Matrix a, float b) {
  for (size_t row=0; row<a.num_rows; ++row) {
    for (size_t col=0; col<a.num_cols; ++col) {
      MAT_AT(a, row, col) += b;
    }
  }
}

void mat_add_mat(Matrix a, Matrix b) {
  assert(a.num_cols == b.num_cols);
  assert(a.num_rows == b.num_rows);
  for (size_t row=0; row<a.num_rows; ++row) {
    for (size_t col=0; col<a.num_cols; ++col) {
      MAT_AT(a, row, col) += MAT_AT(b, row, col);
    }
  }
}

void mat_mul_num(Matrix a, float b) {
  for (size_t row=0; row<a.num_rows; ++row) {
    for (size_t col=0; col<a.num_cols; ++col) {
      MAT_AT(a, row, col) *= b;
    }
  }
}

void mat_mul_mat(Matrix dst, Matrix a, Matrix b) {
  assert(a.num_cols == b.num_rows);
  assert(dst.num_rows == a.num_rows);
  assert(dst.num_cols == b.num_cols);
  printf("%zu\n", a.num_rows);
  for (size_t row=0; row<dst.num_rows; ++row) {
    for (size_t col=0; col<dst.num_cols; ++col) {
      float tmp = 0;
      for (size_t i=0; i<dst.num_cols; ++i) {
        tmp += MAT_AT(a, row, i) * MAT_AT(b, i, col);
      }
      MAT_AT(dst, row, col) = tmp;
    }
  }
}


#endif // NN_IMPLEMENTATION
