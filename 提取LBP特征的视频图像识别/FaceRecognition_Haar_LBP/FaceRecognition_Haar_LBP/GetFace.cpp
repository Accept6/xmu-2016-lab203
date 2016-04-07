#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include <cstdio>    
#include <cstdlib>    
#include <Windows.h>   
#include"lbp.h"

using namespace std;   
using namespace cv;

//��ȡ��ͼ��
IplImage* GetSubImage(IplImage *pImg, CvRect roi)
{
    // ���� ROI 
    cvSetImageROI(pImg,roi);
    // ������ͼ��
    IplImage *result = cvCreateImage( cvSize(roi.width, roi.height), pImg->depth, pImg->nChannels );
    cvCopy(pImg,result);
    cvResetImageROI(pImg);	
    return result;
}

//��ȡģ������
IplImage *GetFaceRegion(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade)
{
	//IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);
	//cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY); 

	CvMemStorage *pcvMemStorage = cvCreateMemStorage(0);    
	cvClearMemStorage(pcvMemStorage); 
	CvSeq *pcvSeqFaces = cvHaarDetectObjects(pSrcImage, pHaarClassCascade, pcvMemStorage); 

	int rMaxWidth=0;
	int iMax=-1;
	//ȡģ��ͼ������������
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
		IplImage *pFace = GetSubImage(pSrcImage, *rMax);
		return pFace;
	}

	cvReleaseMemStorage(&pcvMemStorage); 
	return pSrcImage;
}