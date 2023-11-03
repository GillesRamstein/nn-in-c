#include <SDL2/SDL_video.h>
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

void handle_inputs(SDL_Event *event, int *quit) {
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
    result = arr[0] > result ? arr[0] : result;
  }
  return result;
}

void handle_rendering(SDL_Renderer *renderer, int w, int h) {
  SDL_RenderClear(renderer);

  // TODO: read data from model
  size_t n_layers = 6;
  size_t nodes[] = {4, 8, 12, 12, 5, 1};
  size_t max_nodes = array_max(nodes, n_layers);

  // connections
  SDL_Color rgb_c;
  int x1, y1, x2, y2;
  for (size_t i = 0; i < n_layers - 1; ++i) {
    x1 = w / (n_layers + 1) * (i + 1);
    x2 = w / (n_layers + 1) * (i + 2);
    for (size_t j = 0; j < nodes[i]; ++j) {
      y1 = h / (nodes[i] + 1) * (j + 1);
      for (size_t k = 0; k < nodes[i + 1]; ++k) {
        y2 = h / (nodes[i + 1] + 1) * (k + 1);
        // dummy colors TODO: use real weights
        hsv2rgb(
            scaler_linear((x1 + x2) * (y1 + y1), 0, 2 * 1920 * 1080, 0, 360),
            254, 254, &rgb_c);
        aalineRGBA(renderer, x1, y1, x2, y2, rgb_c.r, rgb_c.g, rgb_c.b, 0x88);
        // thickLineRGBA(renderer, x1, y1, x2, y2, 2, rgb_c.r, rgb_c.g, rgb_c.b,
        // 0x88);
      }
    }
  }

  // count nodes for fake node color values
  int n_nodes = 0;
  for (size_t i = 0; i < n_layers; ++i) {
    n_nodes += nodes[i];
  }

  // nodes
  SDL_Color rgb_n;
  int r = (w > h ? w / n_layers : h / max_nodes) / 9;
  int x, y;
  for (size_t i = 0; i < n_layers; ++i) {
    x = w / (n_layers + 1) * (i + 1);
    for (size_t j = 0; j < nodes[i]; ++j) {
      y = h / (nodes[i] + 1) * (j + 1);
      // dummy colors TODO: use real biases
      hsv2rgb(scaler_linear(x * y, 0, 1920 * 1080, 0, 360), 254, 254, &rgb_n);
      // printf("%d %d [%d, %d, %d]\n", x, y, rgb.r, rgb.g, rgb.b);
      // TODO: find antialiased filled circle
      aacircleRGBA(renderer, x, y, r, rgb_n.r, rgb_n.g, rgb_n.b, 0xFF);
      // filledCircleRGBA(renderer, x, y, r - 1, rgb_n.r, rgb_n.g, rgb_n.b,
      // 0xFF);
    }
  }

  SDL_SetRenderDrawColor(renderer, 0x23, 0x23, 0x23, 255);
  SDL_RenderPresent(renderer);
}

int main() {

  /*
  File Structure
  first layer holds numbers of layers: size_t N
  second layer holds number of nodes for each of the N layers: N * size_t n
  next N-1 layers hold the biases for one layer
  next N-1 layers hold the weights for one layer
  */

  size_t layer_dims[] = {2, 5, 1}; // {dim_in, [dim_h, ...], dim_out}
  Sigma s_hidden = LEAKY_RELU;     // activation for hidden layers
  Sigma s_output = SIGMOID;        // activation for output layer

  // alloc network
  NN nn = nn_create(layer_dims, ARRAY_LEN(layer_dims), s_hidden, s_output);

  // write to file
  FILE *fp_w;
  fp_w = fopen("tmp.gil", "w");
  if (!fp_w) {
    fprintf(stderr, "ERROR: fopen write");
    return 1;
  }
  // first line: number of layers
  fprintf(fp_w, "%zu\n\n", nn.n_layers);
  // second line: number of nodes for each layer
  fprintf(fp_w, "%zu ", nn.weights[1].num_rows);
  for (size_t i = 1; i < nn.n_layers; ++i) {
    fprintf(fp_w, "%zu ", nn.weights[i].num_cols);
  }
  fprintf(fp_w, "\n\n");
  // biases for all layers except input layer
  for (size_t i = 1; i < nn.n_layers; ++i) {
    // all biases for one layer on one line
    for (size_t j = 0; j < nn.biases[i].num_cols; ++j) {
      fprintf(fp_w, "%f ", MAT_AT(nn.biases[i], 0, j));
    }
    fprintf(fp_w, "\n\n");
  }
  // weights for all layer except input layer
  for (size_t i = 1; i < nn.n_layers; ++i) {
    // one row per line
    for (size_t k = 0; k < nn.weights[i].num_rows; ++k) {
      for (size_t j = 0; j < nn.weights[i].num_cols; ++j) {
        fprintf(fp_w, "%f ", MAT_AT(nn.biases[i], k, j));
      }
      fprintf(fp_w, "\n");
    }
    fprintf(fp_w, "\n");
  }
  fclose(fp_w);

  // read from file
  FILE *fp_r;
  fp_r = fopen("tmp.gil", "r");
  if (fp_r == NULL) {
    fprintf(stderr, "ERROR: fopen read");
    return 1;
  }

  char *buffc = malloc(sizeof(size_t)+1);
  if (!buffc) {
    fprintf(stderr, "ERROR: alloc buffc");
    return 1;
  }
  // float bufff[10] = { 0 }; // better use malloc here

  // read first line
  if (!fgets(buffc, sizeof(buffc), fp_r)) {
    fprintf(stderr, "ERROR: fgets");
    return 1;
  }
  size_t n_layers = atoi(buffc);
  printf("N_layers: %zu\n", n_layers);

  // read separator line
  if (!fgets(buffc, sizeof(buffc), fp_r)) {
    fprintf(stderr, "ERROR: fgets");
    return 1;
  }

  // read second line
  printf("N_nodes per layer: ");
  buffc = realloc(buffc, sizeof(buffc)*3+1);
  if (!buffc) {
    fprintf(stderr, "ERROR: realloc buffc");
    return 1;
  }
  if (!fgets(buffc, sizeof(buffc), fp_r)) {
    fprintf(stderr, "ERROR: fgets");
    return 1;
  }
  char *tk;
  tk = strtok(buffc, " ");
  size_t tmp;
  while (tk!=NULL) {
    // bufff[i++] = strtof(tk, NULL);
    tmp = atoi(tk);
    if (!tmp) {
      break;
    };
    // tmp = strtof(tk, NULL);
    tk = strtok(NULL, " ");
    printf("%zu ", tmp);
  }
  printf("\n");

  free(buffc);
  fclose(fp_r);

  return 0;

  // Setup SDL
  SDL_Window *window = NULL;
  SDL_Renderer *renderer = NULL;
  if (init_sdl(&window, &renderer) == false) {
    return 1;
  }

  int w, h;

  SDL_Event event;
  int quit = false;
  while (!quit) {

    // SDL_GetWindowSize(window, &w, &h);
    SDL_GetWindowSizeInPixels(window, &w, &h);

    // process inputs
    handle_inputs(&event, &quit);

    // update state
    handle_state();

    // render image
    handle_rendering(renderer, w, h);

    // cap loop rate
    SDL_Delay(RENDER_RATE);
  }

  // Cleanup SDL
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
