#include"stdafx.h"
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include"caffe.h"
using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<float>GetFaceDNNeigen(Mat src,Ptr<dnn::Importer> importer)
{
	dnn::Net net;
	importer->populateNet(net);
	importer.release();                     //We don't need importer anymore
	
	resize(src, src, Size(224, 224));       //GoogLeNet accepts only 224x224 RGB-images
	dnn::Blob inputBlob = dnn::Blob(src);   //Convert Mat to dnn::Blob image batch
	net.setBlob(".data", inputBlob);        //set the network input
	net.forward();                          //compute output
	dnn::Blob prob = net.getBlob("fc6");   //gather output of "prob" layer

	Mat probMat = prob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
	vector<float>FaceDNN;
	float temp = 0.0;
	for (int i = 0;i<probMat.cols;i++)
	{
		temp = probMat.ptr<float>(0)[i];
		FaceDNN.push_back(temp);
	}
	return FaceDNN;
} 

//计算一张图像中所有人脸的DNN特征
vector<vector<float>>GetAllFaceDNN(Mat src,vector<Rect>AllFaceRect,Ptr<dnn::Importer> importer)
{
	vector<vector<float>>ALLFaceDNN;
	for(int i=0;i<AllFaceRect.size();i++)
	{
		Mat dst(src,AllFaceRect[i]);
		vector<float>DNNeigen=GetFaceDNNeigen(dst,importer);
		ALLFaceDNN.push_back(DNNeigen);
	}
	return ALLFaceDNN;
}

void FaceRecognitionHaarDNN(string VideoPath,string TemplateFacePath) 
{
	String modelTxt = "VGG_FACE_deploy.prototxt";
	String modelBin = "VGG_FACE.caffemodel";

	Ptr<dnn::Importer> importer;
	try                                     //Try to import Caffe GoogleNet model
	{
		importer = dnn::createCaffeImporter(modelTxt, modelBin);
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}
	if (!importer)
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBin << std::endl;
		std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}

	const char *FaceDetectionPath = "haarcascade_frontalface_alt2.xml"; 
	CvHaarClassifierCascade *pHaarClassCascade;    
	pHaarClassCascade = (CvHaarClassifierCascade*)cvLoad(FaceDetectionPath); 

	if(pHaarClassCascade)
	{

		//获取模板人脸DNN特征
		IplImage* Face= cvLoadImage(TemplateFacePath.c_str(), CV_LOAD_IMAGE_ANYCOLOR);
		Mat FaceTemplate= GetFaceRegion(Face,pHaarClassCascade);
		vector<float>FaceTemplateDNN = GetFaceDNNeigen(FaceTemplate,importer);

		//视频
		CvCapture* capture = cvCreateFileCapture( VideoPath.c_str());  
		IplImage* pSrcImage;  
		
		float sumMin=100000000.0,sum=0.0;
		int FaceNear=0;
		static int shift=0;
		if(capture)
		{
			 while(1)  
			{  
				//system("cls");
				pSrcImage = cvQueryFrame(capture);
				if(!pSrcImage) break ;  
				
				//一张图像所有人脸的DNN特征
				vector<Rect>AllFaceRect=GetALLFaceRect(pSrcImage,pHaarClassCascade);
				Mat src(pSrcImage);
				vector<vector<float>>ALLFaceDNN = GetAllFaceDNN(src, AllFaceRect, importer);
			
				//一张图像所有人脸的LBP特征与模板人脸特征匹配
				for(int i=0;i<ALLFaceDNN.size();i++)
				{
					sum=0.0;
					for(int j=0;j<ALLFaceDNN[i].size();j++)
					{
						sum= sum+fabs(ALLFaceDNN[i][j]- FaceTemplateDNN[j]);
					}
					//cout<<sum<<endl;
					if(sum<sumMin)
					{
						sumMin=sum;
						FaceNear=i;
					}
				}

				//找到的人脸化红框
				if(sumMin<15000)
				{
					cv::rectangle(src,AllFaceRect[FaceNear].tl(),AllFaceRect[FaceNear].br(),Scalar(0,0,255),2);//画红色
				}
				  
				namedWindow("src",WINDOW_NORMAL);
				imshow("src",src);
				waitKey(1);
			}  
		}
		cvReleaseCapture(&capture);  
		cvReleaseImage(&pSrcImage); 
	}
}