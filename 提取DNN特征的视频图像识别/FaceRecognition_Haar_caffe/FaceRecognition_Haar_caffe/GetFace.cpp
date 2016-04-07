#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
//#include<opencv2/nonfree/nonfree.hpp>
#include <cstdio>    
#include <cstdlib>    
#include <Windows.h>   
#include"caffe.h"

using namespace std;   
using namespace cv;

//获取模板人脸
Mat GetFaceRegion(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade)
{
	IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);
	cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY); 

	CvMemStorage *pcvMemStorage = cvCreateMemStorage(0);    
	cvClearMemStorage(pcvMemStorage); 
	CvSeq *pcvSeqFaces = cvHaarDetectObjects(pGrayImage, pHaarClassCascade, pcvMemStorage); 

	int rMaxWidth=0;
	int iMax=-1;
	Mat src(pSrcImage);
	//取模板图像中最大的人脸
	for(int i = 0; i <pcvSeqFaces->total; i++)
	{    
		CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);
		if(r->width > rMaxWidth)
		{
			rMaxWidth =r->width;
			iMax=i;
		}
	}
	if(iMax>-1)
	{
		CvRect* rMax = (CvRect*)cvGetSeqElem(pcvSeqFaces, iMax);
		Rect r(rMax->x,rMax->y,rMax->width,rMax->height);
		Mat rFace(src,r);
		return rFace;
	}
	cvReleaseMemStorage(&pcvMemStorage); 
	cvReleaseImage(&pGrayImage); 
	return src;
}

vector<Rect>GetALLFaceRect(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade)
{
	IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);
	cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY); 

	CvMemStorage *pcvMemStorage = cvCreateMemStorage(0);    
	cvClearMemStorage(pcvMemStorage); 
	CvSeq *pcvSeqFaces = cvHaarDetectObjects(pGrayImage, pHaarClassCascade, pcvMemStorage); 

	vector<Rect>rect;
	for(int i = 0; i <pcvSeqFaces->total; i++)
	{    
		CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);
		Rect R(r->x,r->y,r->width,r->width);
		rect.push_back(R);
	}   
	cvReleaseMemStorage(&pcvMemStorage); 
	cvReleaseImage(&pGrayImage); 
	return rect;
}