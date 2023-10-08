#define NN_IMPLEMENTATION
#include "nn.h"


int main(void) {

  srand(1);

  // alloc 1x1; fill; print
  Matrix m_1_1 = mat_alloc(1, 1);
  mat_fill(m_1_1, -1);
  MAT_PRINT(m_1_1);
  printf("\n"); // ---------------------------------------------

  // alloc 2x3; rand
  Matrix m_2_3 = mat_alloc(2, 3);
  mat_rand(m_2_3, 0, 1);
  MAT_PRINT(m_2_3);
  printf("\n"); // ---------------------------------------------

  // alloc 4x3; add mat; add num; mul num
  Matrix m_4_3 = mat_alloc(4, 3);
  mat_rand(m_4_3, 0, 1);
  MAT_PRINT(m_4_3);
  printf("\n"); // ---------------------------------------------

  mat_add_mat(m_4_3, m_4_3);
  MAT_PRINT(m_4_3);
  printf("\n"); // ---------------------------------------------

  mat_add_num(m_4_3, 17);
  MAT_PRINT(m_4_3);
  printf("\n"); // ---------------------------------------------

  mat_mul_num(m_4_3, 10);
  MAT_PRINT(m_4_3);
  printf("\n"); // ---------------------------------------------

  // mul_mat
  Matrix dst = mat_alloc(3, 4);
  mat_fill(dst, 0);
  MAT_PRINT(dst);
  Matrix a = mat_alloc(3, 2);
  mat_rand(a, 1, 3);
  MAT_PRINT(a);
  Matrix b = mat_alloc(2, 4);
  mat_rand(b, 1, 2);
  MAT_PRINT(b);
  mat_mul_mat(dst, a, b);
  MAT_PRINT(dst);
  printf("\n"); // ---------------------------------------------

  // mul_mat
  Matrix dst1 = mat_alloc(1, 1);
  mat_fill(dst1, 0);
  MAT_PRINT(dst1);
  Matrix a1 = mat_alloc(1, 2);
  mat_rand(a1, 1, 3);
  MAT_PRINT(a1);
  Matrix b1 = mat_alloc(2, 1);
  mat_rand(b1, 1, 2);
  MAT_PRINT(b);
  mat_mul_mat(dst1, a1, b1);
  MAT_PRINT(dst1);
  printf("\n"); // ---------------------------------------------
  
  printf("> finished all tests\n");
  NN_ASSERT(0);

  // TODO: free memory?
}
