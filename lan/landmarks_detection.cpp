#include "landmarks_detection.h"

Landmarks_Detection::Landmarks_Detection()
{
}

void Landmarks_Detection::init()
{
    detector = get_frontal_face_detector();

    deserialize("model/shape_predictor_68_face_landmarks.dat") >> sp;
}

std::vector<Point2f> Landmarks_Detection::computeLandmarks(Mat inMat, Rect headRect)
{
    _landmarks.clear();
    _dets.clear();

    array2d<rgb_pixel> img;

    dlib::assign_image(img, dlib::cv_image<bgr_pixel>(inMat));

    _dets.push_back(openCVRectToDlib(headRect));

    if(_dets.size() > 0)
    {
        std::vector<full_object_detection> shapes;
        for (unsigned long j = 0; j < _dets.size(); ++j)
        {
            full_object_detection shape = sp(img, _dets[j]);
            shapes.push_back(shape);
        }

        for(int i = 0; i < shapes[0].num_parts(); i++)
        {
            _landmarks.push_back(Point2i(shapes[0].part(i)(0),shapes[0].part(i)(1)));
        }
    }

    return _landmarks;
}

bool Landmarks_Detection::twoPicturesDetected()
{
    return _twoPicturesLandmarks.size() == 2;
}

std::vector< std::vector<Point2i> > Landmarks_Detection::get_twoLandmarksPoint()
{
    return _twoPicturesLandmarks;
}

void Landmarks_Detection::drawLandmarks(Mat& inMat)
{
    for(int i = 0; i < _landmarks.size(); i++)
        circle(inMat,_landmarks[i],1,Scalar(255,0,0),2);
}

cv::Rect Landmarks_Detection::dlibRectangleToOpenCV(dlib::rectangle r)
{
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

dlib::rectangle Landmarks_Detection::openCVRectToDlib(cv::Rect r)
{
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

Mat Landmarks_Detection::compute_mask(Mat inMat)
{
    int minX = _twoPicturesLandmarks[0][0].x;
    int minY = _twoPicturesLandmarks[0][0].y;
    int maxX = _twoPicturesLandmarks[0][0].x;
    int maxY = _twoPicturesLandmarks[0][0].y;

    for(int j = 0; j < _twoPicturesLandmarks[0].size(); j++)
    {
        for(int i = 0; i < _twoPicturesLandmarks.size(); i++)
        {
            if(minX < _twoPicturesLandmarks[i][j].x)
                minX = _twoPicturesLandmarks[i][j].x;

            if(maxX > _twoPicturesLandmarks[i][j].x)
                maxX = _twoPicturesLandmarks[i][j].x;

            if(minY < _twoPicturesLandmarks[i][j].y)
                minY = _twoPicturesLandmarks[i][j].y;

            if(maxY > _twoPicturesLandmarks[i][j].y)
                maxY = _twoPicturesLandmarks[i][j].y;
        }
    }

    if(minY > inMat.rows)
        minY = inMat.rows - 10;

    Rect _roi = Rect(Point2f(minX,minY),Point2f(maxX,maxY));

    Mat mask = Mat::zeros(inMat.size(),CV_8UC1);

    cv::rectangle(mask,_roi,Scalar(255),-1);

    return mask;
}
