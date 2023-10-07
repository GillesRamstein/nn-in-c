#!/bin/sh

mkdir -p build

# gcc timestwo.c -o build/timestwo -O0 -g
# gcc logicgates.c -o build/logicgates -O0 -g -lm
gcc logicgates_xor.c -o build/logicgates_xor -O0 -g -lm

if [[ -n $1 ]] && [[ "${1}" = "run" ]]
then
  # build/timestwo
  # build/logicgates
  build/logicgates_xor
fi
