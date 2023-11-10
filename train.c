#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define RENDER_RATE 100

void hsv2rgb(int h, int s, int v, SDL_Color *rgb) {
  rgb->r = rgb->g = rgb->b = v;

  if (s < 0) {
    ;
  } else {
    if (h >= 360) {
      h %= 360;
    }
    if (s >= 255) {
      s %= 255;
    }
    if (v >= 255) {
      v %= 255;
    }
    int f = h % 60;
    h /= 60;
    int p = (2 * v * (255 - s) + 255) / 510;
    if ((h & 1) != 0) {
      int q = (2 * v * (15300 - s * f) + 15300) / 30600;
      switch (h) {
      case 1:
        rgb->r = (int)q;
        rgb->g = (int)v;
        rgb->b = (int)p;
        break;
      case 3:
        rgb->r = (int)p;
        rgb->g = (int)q;
        rgb->b = (int)v;
        break;
      case 5:
        rgb->r = (int)v;
        rgb->g = (int)p;
        rgb->b = (int)q;
        break;
      }
    } else {
      int t = (2 * v * (15300 - (s * (60 - f))) + 15300) / 30600;
      switch (h) {
      case 0:
        rgb->r = (int)v;
        rgb->g = (int)t;
        rgb->b = (int)p;
        break;
      case 2:
        rgb->r = (int)p;
        rgb->g = (int)v;
        rgb->b = (int)t;
        break;
      case 4:
        rgb->r = (int)t;
        rgb->g = (int)p;
        rgb->b = (int)v;
        break;
      }
    }
  }
}

float scaler_sigmoid(float x, float c) { return 2 * c / (1 + exp(-x)) - c; }

float scaler_linear(float x, float xmin, float xmax, float tmin, float tmax) {
  return (x - xmin) / (xmax - xmin) * (tmax - tmin) + tmin;
}

int init_sdl(SDL_Window **window, SDL_Renderer **renderer) {

  if (SDL_INIT_EVERYTHING < 0) {
    fprintf(stderr, "ERROR: SDL_INIT_EVERYTHING");
    return false;
  }

  *window = SDL_CreateWindow("Ur Mom xD", SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH,
                             WINDOW_HEIGHT, SDL_WINDOW_RESIZABLE);

  if (!*window) {
    fprintf(stderr, "ERROR: SDL_CreateWindow");
    return false;
  }

  *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);

  if (!*renderer) {
    fprintf(stderr, "ERROR: SDL_CreateRenderer");
    return false;
  }

  return true;
}

void handle_inputs(SDL_Event *event, int *quit, int *pause) {
  while (SDL_PollEvent(event)) {
    switch (event->type) {
    case SDL_QUIT:
      *quit = true;
      break;
    case SDL_KEYUP:
      break;
    case SDL_KEYDOWN:
      switch (event->key.keysym.sym) {
      case SDLK_ESCAPE:
        *quit = true;
        break;
      case SDLK_p:
        *pause = *pause == true ? false : true;
        break;
      default:
        break;
      }
    default:
      break;
    }
  }
}

void handle_state() {
  // read current model weights
  // TODO: ...
}

size_t array_max(size_t arr[], size_t len) {
  if (!arr || len < 1) {
    return 0;
  }
  size_t result = arr[0];
  for (size_t i = 1; i < len; ++i) {
    result = arr[i] > result ? arr[i] : result;
  }
  return result;
}

size_t array_min(size_t arr[], size_t len) {
  if (!arr || len < 1) {
    return 0;
  }
  size_t result = arr[0];
  for (size_t i = 1; i < len; ++i) {
    result = arr[i] < result ? arr[i] : result;
  }
  return result;
}

float farray_max(float arr[], size_t len) {
  if (!arr || len < 1) {
    return 0;
  }
  float result = arr[0];
  for (size_t i = 1; i < len; ++i) {
    result = arr[i] > result ? arr[i] : result;
  }
  return result;
}

float farray_min(float arr[], size_t len) {
  if (!arr || len < 1) {
    return 0;
  }
  float result = arr[0];
  for (size_t i = 1; i < len; ++i) {
    result = arr[i] < result ? arr[i] : result;
  }
  return result;
}

void handle_rendering(SDL_Renderer *renderer, int w, int h,
                      const char *model_path) {
  SDL_RenderClear(renderer);

  // read model data
  NN nn = nn_load(model_path);
  size_t nodes[nn.n_layers];
  nodes[0] = nn.weights[1].num_rows;
  nodes[1] = nn.weights[1].num_cols;
  for (size_t i = 2; i < nn.n_layers; ++i) {
    nodes[i] = nn.weights[i].num_cols;
  }
  size_t max_nodes = array_max(nodes, nn.n_layers);

  // get max/min values for color scaling
  float weight_max = 0.0;
  float weight_min = 0.0;
  float bias_max = 0.0;
  float bias_min = 0.0;
  for (size_t i = 1; i < nn.n_layers; ++i) {
    for (size_t j = 0; j < nn.weights[i].num_cols; ++j) {
      float b = MAT_AT(nn.biases[i], 0, j);
      bias_min = bias_min > b ? b : bias_min;
      bias_max = bias_max < b ? b : bias_max;
      for (size_t k = 0; k < nn.weights[i].num_rows; ++k) {
        float w = MAT_AT(nn.weights[i], k, j);
        weight_min = weight_min > w ? w : weight_min;
        weight_max = weight_max < w ? w : weight_max;
      }
    }
  }
  // printf("%f, %f, %f, %f\n", weight_max, weight_min, bias_max, bias_min);

  // connections
  SDL_Color rgb_c;
  int x1, y1, x2, y2;
  float weight;
  for (size_t i = 0; i < nn.n_layers - 1; ++i) {
    x1 = w / (nn.n_layers + 1) * (i + 1);
    x2 = w / (nn.n_layers + 1) * (i + 2);
    for (size_t j = 0; j < nodes[i]; ++j) {
      y1 = h / (nodes[i] + 1) * (j + 1);
      for (size_t k = 0; k < nodes[i + 1]; ++k) {
        weight = MAT_AT(nn.weights[i + 1], j, k);
        // printf("%zu, (%zu, %zu) W: %f\n", i, j, k, weight);
        hsv2rgb(scaler_linear(weight, weight_min, weight_max, 90, 360), 254,
                254, &rgb_c);
        y2 = h / (nodes[i + 1] + 1) * (k + 1);
        // aalineRGBA(renderer, x1, y1, x2, y2, rgb_c.r, rgb_c.g, rgb_c.b,
        // 0x88);
        thickLineRGBA(renderer, x1, y1, x2, y2, 2, rgb_c.r, rgb_c.g, rgb_c.b,
                      0x88);
      }
    }
  }

  // nodes
  SDL_Color rgb_n;
  int r = (w > h ? w / nn.n_layers : h / max_nodes) / 9;
  int x, y;
  float bias;
  for (size_t i = 0; i < nn.n_layers; ++i) {
    x = w / (nn.n_layers + 1) * (i + 1);
    for (size_t j = 0; j < nodes[i]; ++j) {
      if (i > 0) {
        bias = MAT_AT(nn.weights[i], 0, j);
      } else {
        bias = 0;
      }
      // printf("%zu, (0, %zu) B: %f\n", i, j, bias);
      hsv2rgb(scaler_linear(bias, bias_min, bias_max, 90, 360), 254, 254,
              &rgb_n);
      y = h / (nodes[i] + 1) * (j + 1);
      aacircleRGBA(renderer, x, y, r, rgb_n.r, rgb_n.g, rgb_n.b, 0xFF);
      filledCircleRGBA(renderer, x, y, r - 1, rgb_n.r, rgb_n.g, rgb_n.b, 0xFF);
    }
  }

  SDL_SetRenderDrawColor(renderer, 0x23, 0x23, 0x23, 155);
  SDL_RenderPresent(renderer);
}

int visualize(const char *model_path) {
  // Setup SDL
  SDL_Window *window = NULL;
  SDL_Renderer *renderer = NULL;
  if (init_sdl(&window, &renderer) == false) {
    return 1;
  }

  int w, h;

  SDL_Event event;
  int quit = false;
  int pause = false;
  while (!quit) {

    // SDL_GetWindowSize(window, &w, &h);
    SDL_GetWindowSizeInPixels(window, &w, &h);

    // process inputs
    handle_inputs(&event, &quit, &pause);

    if (!pause) {
      // update state
      handle_state();

      // render image
      handle_rendering(renderer, w, h, model_path);
    }

    // cap loop rate
    SDL_Delay(RENDER_RATE);
  }

  // Cleanup SDL
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

int main() { return visualize("xor.model"); }
