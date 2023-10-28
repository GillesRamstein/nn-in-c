#!/bin/sh

mkdir -p build

# gcc src/witout_nn/timestwo.c -o build/timestwo -O0 -g
# gcc src/witout_nn/logicgates.c -o build/logicgates -O0 -g -lm
# gcc src/witout_nn/logicgates_xor.c -o build/logicgates_xor -O0 -g -lm
gcc src/logicgates_xor_nn.c -o build/logicgates_xor_nn -O0 -g -Wall -Wextra -lm

# gcc src/test_nn_mat.c -o build/test_nn_mat -O0 -g -Wall -Wextra -lm


if [[ -n $1 ]] && [[ "${1}" = "run" ]]
then
  # build/timestwo
  # build/logicgates
  # build/logicgates_xor
  build/logicgates_xor_nn
fi

if [[ -n $1 ]] && [[ "${1}" = "test" ]]
then
  build/test_nn_mat
fi
