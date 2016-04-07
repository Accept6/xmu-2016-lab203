#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include <cstdio>    
#include <cstdlib>    
#include <Windows.h>   

using namespace std;   
using namespace cv;

//GetFace.cpp
IplImage* GetSubImage(IplImage *pImg, CvRect roi);
IplImage *GetFaceRegion(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade);

//lbp.cpp
int getPixel(IplImage *pImg, int i, int j);
void setPixel(IplImage *pImg, int i, int j, int value);
int GetHopCount(int i);
int Get1Count(int i);
vector<float>calcLBPH(IplImage *pImg);
vector<float>GetFaceLBPeigen(IplImage *FaceIplImage,int FaceWidth);
vector<Rect>GetALLFaceRect(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade);
vector<float>GetFaceLBP(IplImage *FaceIplImage,int FaceWidth);
vector<vector<float>>GetAllFaceLBP(Mat src,vector<Rect>AllFaceRect,int FaceWidth);
void FaceRecognitionHaarLBP(string VideoPath,string TemplateFacePath);


