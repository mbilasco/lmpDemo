#include "global_header.h"
#include "face_detection/face_detection_main.h"
//#include "landmarks_detection/landmarksdetector.h"
#include "feature_extraction/datasetanalyser.h"
#include "classification/svmlearning.h"
#include "lan/landmarks_detection.h"
#include "normalization/face_normalize.h"

void showFlow(Mat flux);
bool analyseEnable(Mat inMat, Rect head, int border);
std::vector<float> computeCumulateMotion(std::vector< std::vector<float> > vectorCumul);
bool keepHeadMotion(std::vector<float> headMotion, std::vector<std::vector<double> > fullRegionMotion);

int main(int argc, char *argv[])
{  
    /*------------------------------------------------------*/
    /*---------------- DATA INITIALIZATION -----------------*/
    /*------------------------------------------------------*/

    svmLearning _mySVMLearning;
    Landmarks_Detection _myLandmarksDetector;
    DatasetAnalyser _myDatasetAnalyser;
    face_detection_main _myFaceDetector;
    int _NumberCumulatedFrame = 6;
    int scale_factor_analyse = 2;
    int scale_factor_print = 0;
    int scale_transition = 4;

    _myDatasetAnalyser.initialize(3,0.5,0.75,200,4,5,3,12);
    _myLandmarksDetector.init();

    VideoCapture cap(0);

    Mat oldFrame, newFrame,frame, flow, drawFrame;
    std::vector<Point2f> pts;
    Rect headArea;
    Ptr<DenseOpticalFlow> aa;

    long T0, T1;
    double sec1,sec2,sec3,sec4;

    std::vector< std::vector<float> > motionValues;

    cap >> frame;
    frame.copyTo(newFrame);
    frame.copyTo(oldFrame);
    frame.copyTo(drawFrame);

    for(int i = 0; i < scale_factor_print; i++)
    {
        pyrDown(oldFrame,oldFrame,Size((oldFrame.cols + 1) / 2, (oldFrame.rows + 1) / 2));
        pyrDown(newFrame,newFrame,Size((newFrame.cols + 1) / 2, (newFrame.rows + 1) / 2));
        pyrDown(drawFrame,drawFrame,Size((drawFrame.cols + 1) / 2, (drawFrame.rows + 1) / 2));
    }

    for(int i = 0; i < scale_factor_analyse; i++)
    {
        pyrDown(frame,frame,Size((frame.cols + 1) / 2, (frame.rows + 1) / 2));

    }

    /*------------------------------------------------------*/
    /*---------------- ANALYSIS BEGINNING ------------------*/
    /*------------------------------------------------------*/

    for(;;)
    {
        cap >> frame;
        frame.copyTo(newFrame);
        frame.copyTo(drawFrame);

        for(int i = 0; i < scale_factor_print; i++)
        {
            pyrDown(newFrame,newFrame,Size((newFrame.cols + 1) / 2, (newFrame.rows + 1) / 2));
            pyrDown(drawFrame,drawFrame,Size((drawFrame.cols + 1) / 2, (drawFrame.rows + 1) / 2));
        }

        for(int i = 0; i < scale_factor_analyse; i++)
        {
            pyrDown(frame,frame,Size((frame.cols + 1) / 2, (frame.rows + 1) / 2));

        }

        printf("-------------- Step 1 : Face detection ---------------\n");
        /*------------------------------------------------------*/
        /*-------------- Step 1 : Face detection ---------------*/
        /*------------------------------------------------------*/

        T0 = cv::getTickCount();

        headArea = _myFaceDetector.detectFace(frame);

        T1 = cv::getTickCount();
        sec1 = (T1 - T0)/cv::getTickFrequency();

        printf("-------------- Step 2 : Landmarks location ---------------\n");
        /*------------------------------------------------------*/
        /*-------------- Step 2 : Landmarks location -----------*/
        /*------------------------------------------------------*/

        T0 = cv::getTickCount();

        pts = _myLandmarksDetector.computeLandmarks(frame,headArea);

        T1 = cv::getTickCount();
        sec2 = (T1 - T0)/cv::getTickFrequency();

        for(int i = 0; i < pts.size(); i++)
        {
            Point2f p = pts[i];
            p.x = p.x * scale_transition;
            p.y = p.y * scale_transition;
            pts[i] = p;
        }

        Rect head;
        head.x = (headArea.x * scale_transition) - (headArea.width*0.25);
        head.y = (headArea.y * scale_transition) - (headArea.height*0.5);
        head.width = (headArea.width * scale_transition) + (headArea.width*0.5);
        head.height = (headArea.height * scale_transition) + (headArea.height);

        printf("-------------- HEAD DETECTED ---------------\n");
        /**-------------- IF HEAD HAS DETECTED -----------------**/

        if(analyseEnable(drawFrame,head, 0))
        {
        	printf("-------------- Step 3 : OPTICAL FLOW ---------------\n");
            /*------------------------------------------------------*/
            /*-------------- Step 3 : Optical Flow -----------------*/
            /*------------------------------------------------------*/

            Mat oldF, newF, frameF;
            oldFrame(head).copyTo(oldF);
            newFrame(head).copyTo(newF);
            newFrame(head).copyTo(frameF);

            cvtColor(oldF,oldF,CV_BGR2GRAY);
            cvtColor(newF,newF,CV_BGR2GRAY);

            int scale_flow = 1;

            for(int i = 0; i < scale_flow; i++)
            {
                pyrDown(oldF,oldF,Size((oldF.cols + 1) / 2, (oldF.rows + 1) / 2));
                pyrDown(newF,newF,Size((newF.cols + 1) / 2, (newF.rows + 1) / 2));
            }

            T0 = cv::getTickCount();

            aa = optflow::createOptFlow_Farneback();
            //aa = optflow::createOptFlow_DIS();
            //aa = optflow::createOptFlow_DeepFlow();
            //aa = optflow::createOptFlow_SparseToDense();
            aa->calc(oldF,newF,flow);

            T1 = cv::getTickCount();
            sec3 = (T1 - T0)/cv::getTickFrequency();

            for(int i = 0; i < scale_flow; i++)
                pyrUp(flow,flow,Size(flow.cols * 2, flow.rows * 2));

            printf("-------------- Step 4 : Optical flow filtering (LMP) ---------------\n");
            /*------------------------------------------------------------*/
            /*-------------- Step 4 : Optical flow filtering (LMP) -------*/
            /*------------------------------------------------------------*/

            for(int i = 0; i < pts.size(); i++)
            {
                Point2f p = pts[i];
                p.x = p.x - head.x;
                p.y = p.y - head.y;
                pts[i] = p;
            }

            T0 = cv::getTickCount();

            _myDatasetAnalyser.extractMotion(frameF,flow,pts);

            T1 = cv::getTickCount();
            sec4 = (T1 - T0)/cv::getTickFrequency();

            printf("-------------- Step 5 : Classification ---------------\n");
            /*------------------------------------------------------*/
            /*-------------- Step 5 : Classification ---------------*/
            /*------------------------------------------------------*/

            std::vector<float> bbb = _myDatasetAnalyser.getMotionVector();

            if(!keepHeadMotion(_myDatasetAnalyser.getMotionVector(),_myDatasetAnalyser.getRegionMotionVector()))
            {
                std::vector<float> vide(bbb.size(),0);
                motionValues.push_back(vide);
            }
            else
            {
                _myDatasetAnalyser.getFrame().copyTo(drawFrame(head));
                motionValues.push_back(_myDatasetAnalyser.getMotionVector());
            }

            if(motionValues.size() > _NumberCumulatedFrame)
                motionValues.erase(motionValues.begin());

            _mySVMLearning.svmPredict(computeCumulateMotion(motionValues));
        }

        newFrame.copyTo(oldFrame);

        stringstream s1,s2,s3,s4;
        s1 << round(sec1 * 1000);
        s2 << round(sec2 * 1000);
        s3 << round(sec3 * 1000);
        s4 << round(sec4 * 1000);

        putText(drawFrame,string("Face detection : "+ s1.str()), Point(0,15), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,0,0),2,2);
        putText(drawFrame,string("Face alignment : "+ s2.str()), Point(0,30), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,0,0),2,2);
        putText(drawFrame,string("Optical flow : "+ s3.str()), Point(0,45), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,0,0),2,2);
        putText(drawFrame,string("Filtering flow : "+ s4.str()), Point(0,60), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,0,0),2,2);
        putText(drawFrame,string("Prediction : "+ _mySVMLearning.getPrediction()), Point(0,90), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255),2,2);

        imshow("Display LMP",drawFrame);

        if(waitKey(30) >= 0) break;
    }

    return 0;
}








void showFlow(Mat flux)
{
    Mat flow = cv::Mat::zeros(flux.size(), flux.type());
    flux.copyTo(flow);

    Mat flowTab[2];
    Mat angle,magnitude;
    split(flow,flowTab);
    cartToPolar((flowTab[0]),-(flowTab[1]),magnitude,angle);
    Mat hsvTab[3];
    hsvTab[0] = (angle*180/M_PI);
    hsvTab[1] = Mat::ones(flow.rows,flow.cols,CV_32FC1);
    hsvTab[2] = magnitude;

    Mat hsv, bgr;
    merge(hsvTab, 3, hsv);
    cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    imshow("Display flow",bgr);
}

bool analyseEnable(Mat inMat, Rect head, int border)
{
    if(head.x <= (0 + border) || head.x >= (inMat.cols - border))
        return false;
    if(head.y <= (0 + border) || head.y >= (inMat.rows - border))
        return false;
    if((head.y + head.height) >= (inMat.rows - border))
        return false;
    if((head.x + head.width) >= (inMat.cols - border))
        return false;

    return true;
}

std::vector<float> computeCumulateMotion(std::vector< std::vector<float> > vectorCumul)
{
    std::vector<float> cumulateMotion(vectorCumul[0].size(),0);

    for(int i = 0; i < vectorCumul.size(); i++)
    {
        for(int j = 0; j < vectorCumul[0].size(); j++)
        {
            cumulateMotion[j] += vectorCumul[i][j];
        }
    }

    return cumulateMotion;
}

bool keepHeadMotion(std::vector<float> headMotion, std::vector<std::vector<double> > fullRegionMotion)
{
    double max = *max_element(headMotion.begin(), headMotion.end());

    if(max < 5000)
        return false;

    return true;
}




