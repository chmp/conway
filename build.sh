#clang   -Ofast -g -march=native -std=c99 -Wall -o c_conway conway.c
#objdump -S c_conway > c_conway.S

clang++ \
    -Ofast -g -march=native -std=c++1z -Wall \
    -lboost_thread-mt -lboost_system-mt -lboost_program_options-mt \
    -o conway conway.cpp
objdump -S conway > conway.S