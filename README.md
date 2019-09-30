== Dataset configurations

Dataset configurations used for experiments were attentively stored in order to be able to reproduce the experiments with the same data settings. Various CSV files (about CASME, SMIC, CK+, MMI and Oulu-Casia) are available here: https://nextcloud.univ-lille.fr/index.php/s/XGJkjdjnqeYmk7w

All dataset where restructured using the CK+ model.
 - subject_number
    - sequence number

Each CSV file is structured as follow :
expression_type;subject_number;sequence_number


== LMP

LMP contains the source code needed to extract LMP descriptors between two frames of a video sequence. The LMP code is embedded in a video demo app that illustrates the usage of the descriptor itself.

The app is structured in 5 steps :
* Face detection - face_detection folder
* Landmarks location - lan folder
* Optical Flow - we use the classical Farnbäck optical flow code available in OpenCV 3.4
* Optical flow filtering (LMP) - feature_extraction folder
* Classification - classification folder


Various models used by the app must be downloaded from their originating sources and stored locally in the "model" folder:
* model/haarcascade_frontalface_alt.xml used by landmarks_detection/landmarksdetector.cpp available here https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
* model/lfpw.model used by landmarks_detection/landmarksdetector.cpp available here https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip
* model/shape_predictor_68_face_landmarks.dat  used by lan/landmarks_detection.cpp available here : https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
* model/CK_reverse.model used by classification/svmlearning.cpp and available for download here

The app requires also the following dependencies:
* OpenCV3.4
* Dlib
* the source code of imgwarp provided by cxcxcxcx https://github.com/cxcxcxcx/imgwarp-opencv/tree/master/src/lib that must be downloaded into the normalisation/lib folder. the includes in the .cpp files need to adapted in order to conform with the new project structure. For example, #include "imgwarp_mls.h" becomes #include "../../normalization/lib/imgwarp_mls.h"

* svm.cpp, svm.h and svm.def available here https://github.com/cjlin1/libsvm - the files should be downloaded and included in the classification folder
