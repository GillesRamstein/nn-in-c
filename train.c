#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <stdbool.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

int init_sdl(SDL_Window **window, SDL_Renderer **renderer) {

  if (SDL_INIT_EVERYTHING < 0) {
    fprintf(stderr, "ERROR: SDL_INIT_EVERYTHING");
    return false;
  }

  *window = SDL_CreateWindow(
    "Ur Mom xD",
    SDL_WINDOWPOS_CENTERED,
    SDL_WINDOWPOS_CENTERED,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    SDL_WINDOW_RESIZABLE
  );

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
  ;
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

  // ... render stuff here ...
  size_t n_layers = 6;
  size_t nodes[] = {4, 8, 12, 12, 5, 1};
  size_t max_nodes = array_max(nodes, n_layers);

  // connections
  int x1, y1, x2, y2;
  for (size_t i = 0; i < n_layers - 1; ++i) {
    x1 = w/(n_layers+1) * (i+1);
    x2 = w/(n_layers+1) * (i+2);
    for (size_t j = 0; j < nodes[i]; ++j) {
      y1 = h/(nodes[i]+1) * (j+1);
      for (size_t k = 0; k < nodes[i+1]; ++k) {
        y2 = h/(nodes[i+1]+1) * (k+1);
        aalineRGBA(renderer, x1, y1, x2, y2, 0x75, 0x75, 0x75, 0x88) ;
      }
    }
  }

  // nodes
  int r = (w > h ? w/n_layers : h/max_nodes) / 9;
  int x, y;
  for (size_t i = 0; i < n_layers; ++i) {
    x = w/(n_layers+1) * (i+1);
    for (size_t j = 0; j < nodes[i]; ++j) {
      y = h/(nodes[i]+1) * (j+1);
      // TODO: find antialiased filled circle
      aacircleRGBA(renderer, x, y, r, 0x99, 0x99, 0x99, 0xFF) ;
      aacircleRGBA(renderer, x, y, r-1, 0x99, 0x99, 0x99, 0xFF) ;
      filledCircleRGBA(renderer, x, y, r-1, 0x99, 0x99, 0x99, 0xFF) ;
    }
  }

  SDL_SetRenderDrawColor(renderer, 0x23, 0x23, 0x23, 255);
  SDL_RenderPresent(renderer);
}

int main() {
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
    SDL_Delay(33);
  }


  // Cleanup SDL
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
