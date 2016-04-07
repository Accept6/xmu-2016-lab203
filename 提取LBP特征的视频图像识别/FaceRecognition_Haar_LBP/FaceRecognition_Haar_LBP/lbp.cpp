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

int getPixel(IplImage *pImg, int i, int j)
{
	if(pImg->nChannels != 1)
	{
		cout<<"image channels must be 1"<<endl;
		return 0;
	}
	if( i < 0 || i >= pImg->height || j < 0 || j >= pImg->width)
	{
		cout<<" ";
		return 0;
	}
	return pImg->imageData[pImg->widthStep * i + j];
}

void setPixel(IplImage *pImg, int i, int j, int value)
{
	if(pImg->nChannels != 1)
	{
		cout<<"image channels must be 1"<<endl;
		return;
	}
	if( i < 0 || i >= pImg->height || j < 0 || j >= pImg->width)
	{
		cout<<" ";
		return;
	}
	pImg->imageData[pImg->widthStep * i + j] = value;
}

//��ȡi��0,1���������
int GetHopCount(int i)
{
    // ת��Ϊ������
    int a[8] = { 0 };
    int k = 7;
    while (i)
    {
        // ��2ȡ��
        a[k] = i % 2;
        i/=2;
        --k;
    }

    // �����������
    int count = 0;
    for (int k = 0; k<8; ++k)
    {
        if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
        {
            ++count;
        }
    }
    return count;
}

//ȡ1�ĸ���
int Get1Count(int i)
{
	int temp=0;
	int count=0;
    while (i)
    {
        // ��2ȡ��
        temp = i % 2;
        i/=2;
		if(temp==1)
		{
			count++;
		}
    }
	return count;
}

//����LBP����
vector<float>calcLBPH(IplImage *pImg)
{
	int i, j;
	IplImage* pLBP;  //LBP����ͼ��
	int weight[] = {0, 2, 4, 8, 16, 32, 64, 128};   //�˸��ܱ����ص�Ȩֵ
	int iBiStrLen = 8;  //LBP����ĳ���
	if(pImg->width != 96 || pImg->height != 96)
	{
		cout<<"image width and iamge height must be 96!"<<endl;
	}
	if(pImg->nChannels != 1)
	{
		cout<<"The channels of images must be 1!"<<endl;
	}
	pLBP = cvCreateImage( cvSize(pImg->width, pImg->height), IPL_DEPTH_8U, 1);

	//LBPͼ�����ػҶ�ֵ����
	for(i=0; i<pLBP->height; i++)
	{
		for(j=0; j<pLBP->width; j++)
		{
			pLBP->imageData[i * pImg->widthStep + j] = 0;
//			cvmSet(pLBP, i, j, 0);
		}
	}

	//Ϊÿ�����ؼ���LBP����
	int height = pImg->height;
	int width = pImg->width;
	for(i=0; i<height; i++)
	{
		for(j=0; j<width; j++)
		{
			if( getPixel(pImg, i, (j+1)%width) > getPixel(pImg, i, j) )
			{
				setPixel( pLBP, i, j, getPixel(pLBP, i, j) + weight[3] );
			}
			else
			{
				setPixel( pLBP, i, (j+1)%width, getPixel( pLBP, i, (j+1)%width ) + weight[7] );
			}

			if( getPixel( pImg, (i+1)%height, (j+1)%width ) > getPixel(pImg, i, j) )
			{
				setPixel( pLBP, i, j, getPixel(pLBP, i, j) + weight[4] );
			}
			else
			{
				setPixel( pLBP, (i+1)%height, (j+1)%width, getPixel(pLBP, (i+1)%height, (j+1)%width) + weight[0] );
			}

			if( getPixel( pImg, (i+1)%height, j ) > getPixel(pImg, i, j) )
			{
				setPixel( pLBP, i, j, getPixel(pLBP, i, j) + weight[5] );
			}
			else
			{
				setPixel( pLBP, (i+1)%height, j, getPixel(pLBP, (i+1)%height, j) + weight[1] );
			}
			if( getPixel( pImg, (i+1)%height, j==0 ? (width-1) : (j-1) ) > getPixel(pImg, i, j) )
			{
				setPixel( pLBP, i, j, getPixel( pLBP, i, j) + weight[6] );
			}
			else
			{
				setPixel( pLBP,  (i+1)%height, j==0 ? (width-1) : (j-1), getPixel( pLBP, (i+1)%height, j==0 ? (width-1) : (j-1) ) + weight[2] );
			}
		}
	}
	int iBNum;  //ͼ�����ص�LBP����
	int iBitTransNum; //LBP������1��0������0��1���任����
	int iUnifPat;  //����uniformģʽ��LBP����
	int k;
	for(i=0; i<height; i++)
	{
		for(j=0; j<width; j++)
		{
			iUnifPat = 0;
			//iBitTransNum = 0;
			iBNum = getPixel(pLBP, i, j);

			iBitTransNum=GetHopCount(iBNum);
			iUnifPat=Get1Count(iBNum);
			if(iBitTransNum > 2)  //�ж��Ƿ�Ϊuniformģʽ����iBitTransNum����2ʱΪuniformģʽ
			{
				iUnifPat = iBiStrLen+1; //��uniformģʽ����ֵΪiBiStrLen+1
			}
			CvScalar sc;
			sc.val[0] = iUnifPat;
			cvSet2D(pLBP, i, j, sc);
		}
	}
	
	CvScalar scal;
	CvMat *pLBPH = cvCreateMat(160, 1, CV_32FC1);  //����uniformģʽ��LBPͼ��Ŀռ���ǿֱ��ͼ
	//�ռ���ǿֱ��ͼ����
	for(i=0; i<160; i++)
	{
		cvmSet(pLBPH, i, 0, 0);
	}
	k=0;
	int ii, jj;

	//����ռ���ǿֱ��ͼ
	for(i=0; i<=72; i+=24)
	{
		for(j=0; j<=72; j+=24)
		{
			for(ii=0; ii<24; ii++)
			{
				for(jj=0; jj<24; jj++)
				{
					scal = cvGet2D(pLBP, i+ii, j+jj);
					int temp = cvmGet(pLBPH, k+scal.val[0], 0);
					cvmSet( pLBPH, k+scal.val[0], 0, temp+1);
				}
			}
			k+=10;
		}		
	}
	vector<float>pLBPeigen;
	//cout<<pLBPH->cols<<endl;
	for(int i=0;i<pLBPH->rows;i++)
	{
		scal = cvGet2D(pLBPH, i, 0);
		pLBPeigen.push_back(scal.val[0]);
	}
	//cout<<pLBPeigen.size()<<endl;

	cvReleaseImage(&pLBP);
	cvReleaseMat(&pLBPH);
	return pLBPeigen;
}

//��ȡģ��ͼ��������LBP����
vector<float>GetFaceLBPeigen(IplImage *FaceIplImage,int FaceWidth)
{
	//IplImage *pGrayImage = cvCreateImage(cvGetSize(FaceIplImage), IPL_DEPTH_8U, 1);
	IplImage *FaceNormal = cvCreateImage(cvSize(FaceWidth, FaceWidth), 8, 1);  //��Ź�񻯵�����ͼ��
	//cvCvtColor(FaceIplImage, pGrayImage, CV_BGR2GRAY); 
	cvResize(FaceIplImage, FaceNormal, CV_INTER_LINEAR);//����ͼ��ߴ�
	cvEqualizeHist(FaceNormal, FaceNormal);//ֱ��ͼ���⻯
	vector<float>LBPeigen= calcLBPH(FaceNormal); //��������ͼ������
	return LBPeigen;
}

//��ȡһ��ͼ������������
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

//����һ��������LBP����
vector<float>GetFaceLBP(IplImage *FaceIplImage,int FaceWidth)
{
	IplImage *pGrayImage = cvCreateImage(cvGetSize(FaceIplImage), IPL_DEPTH_8U, 1);
	IplImage *FaceNormal = cvCreateImage(cvSize(FaceWidth, FaceWidth), 8, 1);  //��Ź�񻯵�����ͼ��
	cvCvtColor(FaceIplImage, pGrayImage, CV_BGR2GRAY); 
	cvResize(pGrayImage, FaceNormal, CV_INTER_LINEAR);//����ͼ��ߴ�
	cvEqualizeHist(FaceNormal, FaceNormal);//ֱ��ͼ���⻯
	vector<float>LBPeigen= calcLBPH(FaceNormal); //��������ͼ������
	return LBPeigen;
}

//����һ��ͼ��������������LBP����
vector<vector<float>>GetAllFaceLBP(Mat src,vector<Rect>AllFaceRect,int FaceWidth)
{
	vector<vector<float>>ALLFaceLBP;
	for(int i=0;i<AllFaceRect.size();i++)
	{
		Mat dst(src,AllFaceRect[i]);
		IplImage* img=&IplImage(dst);
		vector<float>LBPeigen= GetFaceLBP(img,FaceWidth); //��������ͼ������
		ALLFaceLBP.push_back(LBPeigen);
	}
	return ALLFaceLBP;
}

//Haar�����������ȡ����LBP����ƥ��
void FaceRecognitionHaarLBP(string VideoPath,string TemplateFacePath)  
{	
	const char *FaceDetectionPath = "haarcascade_frontalface_alt2.xml"; 
	CvHaarClassifierCascade *pHaarClassCascade;    
	pHaarClassCascade = (CvHaarClassifierCascade*)cvLoad(FaceDetectionPath); 

	if(pHaarClassCascade)
	{

		//��ȡģ������LBP����
		IplImage* Face= cvLoadImage(TemplateFacePath.c_str(), 0);
		IplImage* FaceTemplate= GetFaceRegion(Face,pHaarClassCascade);
		vector<float>FaceTemplateLBP = GetFaceLBPeigen(FaceTemplate,96);

		//��Ƶ
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
				
				//һ��ͼ������������LBP����
				vector<Rect>AllFaceRect=GetALLFaceRect(pSrcImage,pHaarClassCascade);
				Mat src(pSrcImage);
				vector<vector<float>>ALLFaceLBP = GetAllFaceLBP(src,AllFaceRect,96);
			
				//һ��ͼ������������LBP������ģ����������ƥ��
				for(int i=0;i<ALLFaceLBP.size();i++)
				{
					sum=0.0;
					for(int j=0;j<ALLFaceLBP[i].size();j++)
					{
						sum= sum+fabs(ALLFaceLBP[i][j]- FaceTemplateLBP[j]);
					}
					//cout<<sum<<endl;
					if(sum<sumMin)
					{
						sumMin=sum;
						FaceNear=i;
					}
				}

				//�ҵ������������
				if(sumMin<3000)
				{
					cv::rectangle(src,AllFaceRect[FaceNear].tl(),AllFaceRect[FaceNear].br(),Scalar(0,0,255),2);//����ɫ
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