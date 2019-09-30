#ifndef SVMLEARNING_H
#define SVMLEARNING_H

#include "../global_header.h"
#include "svm.h"

class svmLearning
{
public:
    svmLearning();

    void svmPredict(vector<float> facialMotionVector);
    string getPrediction();

private:
    string _model;
    svm_model* _svmModel;
    string _expressionValue;
};

#endif // SVMLEARNING_H
