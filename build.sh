#!/bin/sh

mkdir -p build

# gcc witout_nn/timestwo.c -o build/timestwo -O0 -g
# gcc witout_nn/logicgates.c -o build/logicgates -O0 -g -lm
# gcc witout_nn/logicgates_xor.c -o build/logicgates_xor -O0 -g -lm
gcc logicgates_xor_nn.c -o build/logicgates_xor_nn -O0 -g -Wall -Wextra -lm

# gcc test_nn_mat.c -o build/test_nn_mat -O0 -g -Wall -Wextra -lm


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
