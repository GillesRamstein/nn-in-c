/*
Simple implementation of a Neural Network with
three Neurons,
two inputs, and
a single output,
with the goal of learning the OR, AND, NAND and XOR logic gate.

h1 = f(i1, i2, w12, w21, b1)
h2 = f(i2, i1, w21, w22, b2)
out = f(h1, h2, w31, w32, b3)
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

typedef struct {
  float w11;
  float w12;
  float b1;
  float w21;
  float w22;
  float b2;
  float w31;
  float w32;
  float b3;
} model_s;

float rand_float(void) { // [0.0 - 1.0]
  return (float)rand() / (float)RAND_MAX;
}

model_s init_model() {
  model_s m = {
    .w11=rand_float(),
    .w12=rand_float(),
    .b1=rand_float(),
    .w21=rand_float(),
    .w22=rand_float(),
    .b2=rand_float(),
    .w31=rand_float(),
    .w32=rand_float(),
    .b3=rand_float(),
  };
  return m;
}

void print_model(model_s m) {
  printf("Model Weights:\n");
  printf(" w11=%f\n", m.w11);
  printf(" w12=%f\n", m.w12);
  printf("  b1=%f\n", m.b1);
  printf(" w21=%f\n", m.w21);
  printf(" w22=%f\n", m.w22);
  printf("  b2=%f\n", m.b2);
  printf(" w31=%f\n", m.w31);
  printf(" w32=%f\n", m.w32);
  printf("  b3=%f\n\n", m.b3);
}

float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float forward(model_s m, float x1, float x2) {
  float h1 = sigmoidf(m.w11 * x1 + m.w12 * x2 + m.b1);
  float h2 = sigmoidf(m.w21 * x1 + m.w22 * x2 + m.b2);
  return sigmoidf(m.w31 * h1 + m.w32 * h2 + m.b3);
}

float mse(model_s model) {
  /* Mean Squared Error */
  float sq_errs = 0.0f;
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y_hat = forward(model, x1, x2);
    float d = y_hat - train[i][2];
    sq_errs += d * d;
  }
  return sq_errs / N_TRAIN;
}

model_s finite_diff_model(model_s m) {
  /*
  Float addition/subtraction is not exact, saving prevents error accumulation
    float a = 1.0;
    a += EPS;
    a -= EPS;
    a == 1.0 -> False
  */

  const float EPS = 1e-3;
  model_s g;
  float tmp;
  float e = mse(m);

  tmp = m.w11;
  m.w11+=EPS;
  g.w11=(mse(m)-e)/EPS;
  m.w11=tmp; 

  tmp = m.w12;
  m.w12+=EPS;
  g.w12=(mse(m)-e)/EPS;
  m.w12=tmp;

  tmp = m.b1;
  m.b1+=EPS;
  g.b1=(mse(m)-e)/EPS;
  m.b1=tmp;

  tmp = m.w21;
  m.w21+=EPS;
  g.w21=(mse(m)-e)/EPS;
  m.w21=tmp;

  tmp = m.w22;
  m.w22+=EPS;
  g.w22=(mse(m)-e)/EPS;
  m.w22=tmp;

  tmp = m.b2;
  m.b2+=EPS;
  g.b2=(mse(m)-e)/EPS;
  m.b2=tmp;

  tmp = m.w31;
  m.w31+=EPS;
  g.w31=(mse(m)-e)/EPS;
  m.w31=tmp;

  tmp = m.w32;
  m.w32+=EPS;
  g.w32=(mse(m)-e)/EPS;
  m.w32=tmp;

  tmp = m.b3;
  m.b3+=EPS;
  g.b3=(mse(m)-e)/EPS;
  m.b3=tmp;

  return g;
}

model_s update_model_weights(model_s m, model_s g, float lr) {
  m.w11-=g.w11;
  m.w12-=g.w12;
  m.b1-=g.b1;
  m.w21-=g.w21;
  m.w22-=g.w22;
  m.b2-=g.b2;
  m.w31-=g.w31;
  m.w32-=g.w32;
  m.b3-=g.b3;
  return m;
}

void evaluate(model_s model) {
  printf("Model Evaluation:\n");
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float y_hat = forward(model, x1, x2);
    printf("%d ^ %d -> %f\n", (int)x1, (int)x2, y_hat);
  }
}

int main(void) {
  // random seed
  srand(time(0));

  // init model weights
  printf("\n --- Init Model ---\n\n");
  model_s model = init_model();
  print_model(model);
  evaluate(model);

  // set training parameters
  const size_t N_EPOCHS = 10*1000;
  const float LEARNING_RATE = 1e-3;

  // optimize weights - training loop
  printf("\n --- Train Model ---\n\n");
  printf("(0) mse=%f\n", mse(model));

  for (size_t i=1; i<=N_EPOCHS; i++) {
    // compute finite differences
    model_s g = finite_diff_model(model);

    // update weights
    model = update_model_weights(model, g, LEARNING_RATE);

    // print error
    if (i % (N_EPOCHS/10) == 0) {
      printf("(%lu) mse=%f\n", i, mse(model));
    }

  } // end training loop

  printf("\n --- Final Model ---\n\n");
  printf("mse=%f\n\n", mse(model));
  print_model(model);
  evaluate(model);

  // Display truth tables of each Neuron when fed the training data
  printf("\n --- Peek Into Neurons ---\n");
  printf(" \nN1: w11=%f, w12=%f, b=%f\n", model.w11, model.w12, model.b1);
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float h1 = sigmoidf(model.w11 * x1 + model.w12 * x2 + model.b1);
    printf("N1: %d ^ %d -> %f\n", (int)x1, (int)x2, h1);
  }
  printf("\nN2: w21=%f, w22=%f, b=%f\n", model.w21, model.w22, model.b1);
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float h2 = sigmoidf(model.w21 * x1 + model.w22 * x2 + model.b2);
    printf("N2: %d ^ %d -> %f\n", (int)x1, (int)x2, h2);
  }
  printf("\nN3: w31=%f, w32=%f, b=%f\n", model.w31, model.w32, model.b1);
  for (size_t i=0; i<=N_TRAIN; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float h3 = sigmoidf(model.w31 * x1 + model.w32 * x2 + model.b3);
    printf("N3: %d ^ %d -> %f\n", (int)x1, (int)x2, h3);
  }
}
