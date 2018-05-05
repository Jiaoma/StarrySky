#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 19:20:55 2018

@author: ljc
"""

import numpy as np
import cv2
from scipy import interpolate
from scipy import ndimage
from os import listdir
from os.path import join

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['png','PNG','jpeg','JPEG','tif','TIF','jpg','JPG'])


def LoadAllImageFromFolder(rootdir):
    imageNameList=[join(rootdir,x) for x in listdir(rootdir) if is_image_file(x)]
    imageDataList=[cv2.imread(x) for x in imageNameList]
    return imageDataList

def LoadSingleImageFromFolder(rootdir):
    imageNameList=[join(rootdir,x) for x in listdir(rootdir) if is_image_file(x)]
    image_num=len(imageNameList)
    imageData=None
    for i in range(image_num):
        imageData=cv2.imread(imageNameList[i])
        yield imageData,imageNameList[i].split('/')[-1]
    return imageData,imageNameList[-1].split('/')[-1]

class PerspectiveMatcher:
    def __init__(self,firstImgPath,hessianThreshold=8000,padding=10,stable_num=6):
        self.img1,self.imgName=next(LoadSingleImageFromFolder(firstImgPath))
        self.rows,self.cols,self.chs=self.img1.shape
        self.hessianThreshold=hessianThreshold
        self.padding=padding
        self.stable_num=stable_num
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=self.hessianThreshold)
        gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        mask[self.padding:self.rows - self.padding, self.padding:self.cols - self.padding] = 1
        StarXY1, _ = surf.detectAndCompute(gray, mask)
        self.StarXYPerImageList=[]
        self.StarXYPerImageList.append([[x.pt[0],x.pt[1]] for x in StarXY1[:self.stable_num]])
        self.result=None

    def getAllImage(self,ImgFolder):
        self.Imgs=LoadAllImageFromFolder(ImgFolder)

    def getStableStarXY(self):
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=self.hessianThreshold)
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        mask[self.padding:self.rows - self.padding, self.padding:self.cols - self.padding] = 1
        StarXY=[]
        for im in self.Imgs:
            gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            StarXY, _ = surf.detectAndCompute(gray, mask)
            self.StarXYPerImageList.append([[x.pt[0],x.pt[1]] for x in StarXY[:self.stable_num]])
            
    def PerspectiveCombine(self):
        self.sortAndCut()
        Pt1=np.array(self.StarXYPerImageList[0],dtype=np.float32)
        self.result=np.int16(self.img1)
        for pt in range(1,len(self.StarXYPerImageList)):
            pts=self.StarXYPerImageList[pt]
            PerspectiveMatrix = cv2.getPerspectiveTransform(np.array(pts,dtype=np.float32), Pt1)
            PerspectiveImg = cv2.warpPerspective(self.Imgs[pt-1], PerspectiveMatrix, (self.cols,self.rows))
            self.result=np.int16(np.mean([self.result,PerspectiveImg],axis=0))
        
    def getStarTrackXY(self):
        self.ImageXYPerStar=np.array(self.StarXYPerImageList).transpose([1,0,2])

    
    def finalResult(self):
        self.result=np.where(self.result<0,0,self.result)
        self.result=np.where(self.result>255,255,self.result)
        self.result=np.uint8(self.result)
        return self.result,self.imgName
    
    def sortAndCut(self):
        for i in range(len(self.StarXYPerImageList)):
            self.StarXYPerImageList[i]=sorted(self.StarXYPerImageList[i],key=lambda x:(x[0],x[1]))
            latter=self.StarXYPerImageList[i][0]
            scList=[self.StarXYPerImageList[i][0]]
            for j in self.StarXYPerImageList[i][1:]:
                if np.sum((np.array(latter)-np.array(j))**2)>5:
                    scList.append(j)
                    latter=j
            self.StarXYPerImageList[i]=scList[:4]
            
    def curveAdjust(self,p255=[0,15,30,100,200,255],pc255=[0,10,50,150,210,255],bgrWeight=[1.2,1,1],threshold=10):
        curveFunction=interpolate.interp1d(np.array(p255,dtype=np.int16), np.array(pc255,dtype=np.int16), kind='cubic')
        self.result[:,:,0]=np.int16(curveFunction(self.result[:,:,0])*bgrWeight[0])
        self.result[:,:,1]=np.int16(curveFunction(self.result[:,:,1])*bgrWeight[1])
        self.result[:,:,2]=np.int16(curveFunction(self.result[:,:,2])*bgrWeight[2])
        self.result=np.where(self.result<0,0,self.result)
        self.result=np.where(self.result>255,255,self.result)
        
    def sharpen(self,enhance=0.1,channel='b'):
        self.result=np.array(self.result,dtype=np.int16)
        kernel_5x5 = np.array(
                [[-1, -1, -1, -1, -1],
                 [-1, 1, 2, 1, -1],
                 [-1, 2, 4, 2, -1],
                 [-1, 1, 2, 1, -1],
                 [-1, -1, -1, -1, -1]],dtype=np.int16)
        if channel=='b':
            mask=ndimage.convolve(self.result[:,:,0], kernel_5x5)
        if channel=='g':
            mask=ndimage.convolve(self.result[:,:,1], kernel_5x5)
        if channel=='r':
            mask=ndimage.convolve(self.result[:,:,2], kernel_5x5)
        temp=np.empty((self.rows,self.cols,self.chs),dtype=np.int16)
        temp[:,:,0]=mask
        temp[:,:,1]=mask
        temp[:,:,2]=mask
        self.result+=np.int16(temp*enhance)
        self.result=np.where(self.result<0,0,self.result)
        self.result=np.where(self.result>255,255,self.result)
        
if __name__=='__main__':
    rootPath='/home/ljc/github/StarrySky/data'
    firstPath='/target/HQ_Auriga_Taurus'
    allPath='/major'
    testMatcher=PerspectiveMatcher(rootPath+firstPath)
    testMatcher.getAllImage(rootPath+firstPath+allPath)
    testMatcher.getStableStarXY()
    testMatcher.PerspectiveCombine()
    testMatcher.curveAdjust()
    testMatcher.sharpen()
    img,name=testMatcher.finalResult()
    cv2.imwrite('sr_{}.png'.format(name),img)