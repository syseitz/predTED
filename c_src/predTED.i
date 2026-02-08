%module predTED

%{
#include "predTED.h"
%}

int predict_TED(const char* struct1, const char* struct2, double* weights, int num_weights, double intercept);
