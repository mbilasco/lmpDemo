#ifndef LANDMARKS_DETECTION_H
#define LANDMARKS_DETECTION_H

#include "../global_header.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

using namespace dlib;

class Landmarks_Detection
{
public:
    Landmarks_Detection();

    std::vector<Point2f> computeLandmarks(Mat inMat, Rect headRect);
    void init();
    void drawLandmarks(Mat& inMat);
    cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
    bool twoPicturesDetected();
    Mat compute_mask(Mat inMat);
    std::vector< std::vector<Point2i> > get_twoLandmarksPoint();
    dlib::rectangle openCVRectToDlib(cv::Rect r);

private:
    shape_predictor sp;
    frontal_face_detector detector;
    std::vector<Point2f> _landmarks;
    std::vector<dlib::rectangle> _dets;

    std::vector< std::vector<Point2i> > _twoPicturesLandmarks;
};

#endif // LANDMARKS_DETECTION_H
