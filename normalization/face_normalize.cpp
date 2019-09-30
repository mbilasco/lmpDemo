#include "../normalization/face_normalize.h"

Face_Normalize::Face_Normalize()
{
}

Mat Face_Normalize::normalize_face(Mat inMat, std::vector<Point2f> srcPoint, std::vector<Point2f> dstPoint, int method)
{
    vector<Point2i> srcPoint2, dstPoint2;

    for(int i = 0; i < srcPoint.size(); i++)
    {
        Point2i s = srcPoint[i];
        srcPoint2.push_back(s);
        Point2i d = dstPoint[i];
        dstPoint2.push_back(d);
    }

    ImgWarp_MLS *imgTrans;

    switch(method)
    {
        case 0 : imgTrans = new ImgWarp_MLS_Similarity(); break;
        default : imgTrans = new ImgWarp_MLS_Rigid(); break;
    }

    imgTrans->alpha = 1.0;
    imgTrans->gridSize = 5;

    Mat curImg = imgTrans->setAllAndGenerate(inMat,srcPoint2,dstPoint2,inMat.cols, inMat.rows, 1);

    delete imgTrans;

    //compute_normalizedPoint()

    return curImg;
}

std::vector<Point2i> Face_Normalize::get_normalizedPoint()
{
    return _newPoints;
}

Mat Face_Normalize::crop_face(Mat inMat, std::vector<Point2i> dstPoint)
{
    int minX = dstPoint[0].x;
    int minY = dstPoint[0].y;
    int maxX = dstPoint[0].x;
    int maxY = dstPoint[0].y;

    for(int j = 1; j < dstPoint.size(); j++)
    {
        if(minX < dstPoint[j].x)
            minX = dstPoint[j].x;

        if(maxX > dstPoint[j].x)
            maxX = dstPoint[j].x;

        if(minY < dstPoint[j].y)
            minY = dstPoint[j].y;

        if(maxY > dstPoint[j].y)
            maxY = dstPoint[j].y;
    }

    maxY -= 40;

    if(minY > inMat.rows)
        minY = inMat.rows - 10;

    if(minX > inMat.cols)
        minX = inMat.cols - 10;

    if(maxX < 1)
        maxX = 10;

    if(maxY < 1)
        maxY = 10;

    //cout << minX << ", " << minY << ", " << maxX << ", " << maxY << " - - " << inMat.rows << ", " << inMat.cols << endl;

    //circle(inMat,Point2f(minX,minY),1,Scalar(255,0,0),2);
    //circle(inMat,Point2f(maxX,maxY),1,Scalar(255,0,0),2);

    if(dstPoint[0].x > -1)
        _roi = Rect(Point2f(minX,minY),Point2f(maxX,maxY));

    /*if(dstPoint.size() >= 68)
    {
        _roi.x -= 40;
        _roi.y -= 40;
        _roi.width += 80;
        _roi.height += 80;
    }
    else
    {
        _roi.x -= 40;
        _roi.y -= 40;
        _roi.width += 80;
        _roi.height += 80;
    }

    Mat crop = Mat::zeros(_roi.width, _roi.height, inMat.type());

    inMat(_roi).copyTo(crop);

    resize(crop,crop,Size(300,350));*/

    Mat crop = Mat::zeros(_roi.width, _roi.height, inMat.type());
    inMat(_roi).copyTo(crop);
    resize(crop,crop,Size(200,250));

    return crop;
}

std::vector<Point2i> Face_Normalize::get_positionLandmarks(std::vector<Point2i> dstPoint)
{
    for(int i = 0; i < dstPoint.size(); i++)
    {
        dstPoint[i].x = dstPoint[i].x - _roi.x;
        dstPoint[i].y = dstPoint[i].y - _roi.y;
    }

    return dstPoint;
}
