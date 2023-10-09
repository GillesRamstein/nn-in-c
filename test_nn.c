#define NN_IMPLEMENTATION
#include "nn.h"

void test_mat_mul_mat_1() {
  /* [[ 7 10 ] [ 15 22 ]] */
  printf("------------------------------\n");
  printf("Mat mul 2x2 * 2x2\n");
  Matrix m1 = mat_alloc(2, 2);
  Matrix m2 = mat_alloc(2, 2);
  float e1[4] = { 1, 2, 3, 4 };
  float e2[4] = { 1, 2, 3, 4 };
  m1.p_data = e1;
  m2.p_data = e2;
  MAT_PRINT(m1);
  MAT_PRINT(m2);
  Matrix m1xm2 = mat_alloc(2, 2);
  mat_mul_mat(m1xm2, m1, m2);
  MAT_PRINT(m1xm2);
  printf("\n");
}

void test_mat_mul_mat_2() {
  /* [[ 6 ] [ 8 ]] */
  printf("Mat mul 2x1 * 1x1\n");
  printf("------------------------------\n");
  Matrix m1 = mat_alloc(2, 1);
  Matrix m2 = mat_alloc(1, 1);
  float e1[2] = { 3, 4 };
  float e2[1] = { 2 };
  m1.p_data = e1;
  m2.p_data = e2;
  MAT_PRINT(m1);
  MAT_PRINT(m2);
  Matrix m1xm2 = mat_alloc(2, 1);
  mat_mul_mat(m1xm2, m1, m2);
  MAT_PRINT(m1xm2);
  printf("\n");
}

void test_mat_mul_mat_3() {
  /* [ 6 8 ] */
  printf("------------------------------\n");
  printf("Mat mul 1x1 * 1x2\n");
  Matrix m1 = mat_alloc(1, 1);
  Matrix m2 = mat_alloc(1, 2);
  float e1[1] = { 2 };
  float e2[2] = { 3, 4 };
  m1.p_data = e1;
  m2.p_data = e2;
  MAT_PRINT(m1);
  MAT_PRINT(m2);
  Matrix m1xm2 = mat_alloc(1, 2);
  mat_mul_mat(m1xm2, m1, m2);
  MAT_PRINT(m1xm2);
  printf("\n");
}

void test_mat_mul_mat_4() {
  /* [[ 11 14 17 20 ] [23 30 37 44 ] [ 35 46 57 68 ]] */
  printf("------------------------------\n");
  printf("Mat mul 3x2 * 2x4\n");
  Matrix m1 = mat_alloc(3, 2);
  Matrix m2 = mat_alloc(2, 4);
  float e1[6] = { 1, 2, 3, 4, 5, 6};
  float e2[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  m1.p_data = e1;
  m2.p_data = e2;
  MAT_PRINT(m1);
  MAT_PRINT(m2);
  Matrix m1xm2 = mat_alloc(3, 4);
  mat_mul_mat(m1xm2, m1, m2);
  MAT_PRINT(m1xm2);
  printf("\n");
}

int main(void) {

  srand(1);

  // alloc 1x1;fill;print
  Matrix m_1_1 = mat_alloc(1, 1);
  mat_fill(m_1_1, -1);
  MAT_PRINT(m_1_1);
  printf("\n");
  
  // alloc 2x3;rand
  Matrix m_2_3 = mat_alloc(2, 3);
  mat_rand(m_2_3, 0, 1);
  MAT_PRINT(m_2_3);
  printf("\n");
  
  // alloc 4x3;add mat;add num;mul num
  Matrix m_4_3 = mat_alloc(4, 3);
  mat_rand(m_4_3, 0, 1);
  MAT_PRINT(m_4_3);
  printf("\n");
  
  mat_add_mat(m_4_3, m_4_3);
  MAT_PRINT(m_4_3);
  printf("\n");
  
  mat_add_num(m_4_3, 17);
  MAT_PRINT(m_4_3);
  printf("\n");
  
  mat_mul_num(m_4_3, 10);
  MAT_PRINT(m_4_3);
  printf("\n");
  
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
  printf("\n");
  
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
  printf("\n");
  
  test_mat_mul_mat_1();
  test_mat_mul_mat_2();
  test_mat_mul_mat_3();
  test_mat_mul_mat_4();

  printf("> finished all tests\n");

  // TODO: free memory?
}
