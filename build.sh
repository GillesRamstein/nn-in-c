#!/bin/sh

mkdir -p build

# gcc timestwo.c -o build/timestwo -O0 -g
# gcc logicgates.c -o build/logicgates -O0 -g -lm
# gcc logicgates_xor.c -o build/logicgates_xor -O0 -g -lm
gcc test_nn.c -o build/test_nn -O0 -g -Wall -Wextra

if [[ -n $1 ]] && [[ "${1}" = "run" ]]
then
  # build/timestwo
  # build/logicgates
  build/logicgates_xor
fi

if [[ -n $1 ]] && [[ "${1}" = "test" ]]
then
  build/test_nn
fi
