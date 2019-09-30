#ifndef FACE_NORMALIZE_H
#define FACE_NORMALIZE_H

#include "../global_header.h"

#include "../normalization/lib/imgwarp_mls.h"
#include "../normalization/lib/imgwarp_mls_rigid.h"
#include "../normalization/lib/imgwarp_mls_similarity.h"

class Face_Normalize
{
public:
    Face_Normalize();

    Mat normalize_face(Mat inMat, std::vector<Point2f> srcPoint, std::vector<Point2f> dstPoint, int method);
    std::vector<Point2i> get_normalizedPoint();
    Mat crop_face(Mat inMat, std::vector<Point2i> dstPoint);
    std::vector<Point2i> get_positionLandmarks(std::vector<Point2i> dstPoint);

private:

    std::vector<Point2i> _newPoints;
    Rect _roi;
};

#endif // FACE_NORMALIZE_H
