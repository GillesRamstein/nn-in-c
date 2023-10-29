CC = gcc
CFLAGS = -Wall -Wextra -Og -g
INCLUDES = -I/usr/include/SDL2/
LIBS = -lSDL2 -lSDL2_ttf -lSDL2_gfx -lm
SRCS = train.c
OBJS = $(SRCS:.c=.o)
MAIN = train

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o build/$(MAIN) $(OBJS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN)
