#include "svmlearning.h"

svmLearning::svmLearning()
{
    _model = "model/CK_reverse.model";
    _svmModel = svm_load_model(_model.c_str());
}

void svmLearning::svmPredict(vector<float> facialMotionVector)
{
    //scale data between [-1;+1]
    double valueMin = *min_element(facialMotionVector.begin(), facialMotionVector.end());
    double valueMax = *max_element(facialMotionVector.begin(), facialMotionVector.end());

    double scaleMin = -1; //the normalized minimum desired
    double scaleMax = 1; //the normalized maximum desired

    double valueRange = valueMax - valueMin;
    double scaleRange = scaleMax - scaleMin;

    vector<float> test;

    double max = *max_element(facialMotionVector.begin(), facialMotionVector.end());
    if(max >= 5000)
    {
        for(int i = 0; i < facialMotionVector.size(); i++)
        {
            double val = ((scaleRange * (facialMotionVector[i] - valueMin))/ valueRange) + scaleMin;
            test.push_back(val);
        }

        //format data for svm
        int n = test.size();
        struct svm_node *x = (struct svm_node *) malloc((n+1)*sizeof(struct svm_node));
        for(int i = 0; i < test.size(); i++)
        {
            x[i].index = i+1;
            x[i].value = test[i];
        }

        x[n].index = -1; //requested by libSVM

        //predict result
        double retval = svm_predict(_svmModel,x);
        //printf("retval: %f\n",retval);

        int val = retval;

        if(val <= 1)
            _expressionValue = "Happy";
        else if(val <= 2)
            _expressionValue = "Fear";
        else if(val <= 3)
            _expressionValue = "Surprise";
        else if(val <= 4)
            _expressionValue = "Anger";
        else if(val <= 5)
            _expressionValue = "Disgust";
        else if(val <= 6)
            _expressionValue = "Sad";
        else
            _expressionValue = "";
        /*else if(val <= 11)
            cout << "11 - neutre" << endl;
        else if(val <= 12)
            cout << "12 - neutre" << endl;
        else if(val <= 13)
            cout << "13 - neutre" << endl;
        else if(val <= 14)
            cout << "14 - neutre" << endl;
        else if(val <= 15)
            cout << "15 - neutre" << endl;
        else if(val <= 16)
            cout << "16 - neutre" << endl;
        else
            cout << "-" << endl;*/
    }
}

string svmLearning::getPrediction()
{
    return _expressionValue;
}
