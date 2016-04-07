#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
//#include<opencv2/nonfree/nonfree.hpp>
#include <cstdio>    
#include <cstdlib>    
#include <Windows.h>   

using namespace std;  
using namespace cv::dnn;
using namespace cv;

//GetFace.cpp
Mat GetFaceRegion(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade);
vector<Rect>GetALLFaceRect(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade);

//caffe.cpp
vector<float>GetFaceDNNeigen(Mat src,Ptr<dnn::Importer> importer);
vector<vector<float>>GetAllFaceDNN(Mat src,vector<Rect>AllFaceRect,Ptr<dnn::Importer> importer);
void FaceRecognitionHaarDNN(string VideoPath,string TemplateFacePath);


