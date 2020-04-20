# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:15:57 2020

@author: Praveen
"""

import numpy as np
import cv2
import glob


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit = 2., tileGridSize = (1, 1))
    labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(labImage)
    l = clahe.apply(l)
    labImage = cv2.merge((l, a, b))
    image = cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR)
    return image

def affineLKTracker(image, template, bbox, parameters):
    templateROI = template[bbox[1]:bbox[1] + bbox[3], 
                           bbox[0]:bbox[0] + bbox[2]]
    affineMatrix = np.array([[parameters[0] + 1, parameters[2], parameters[4]], 
                             [parameters[1], parameters[3] + 1, parameters[5]]], 
                             dtype = np.float32)
    iterations = 0
    while iterations <= 30:
        
        warpImage = cv2.warpAffine(image, affineMatrix, (templateROI.shape[1], templateROI.shape[0]))
        cv2.imshow('warp', warpImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        error = templateROI - warpImage
        gradxImage = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3)
        gradyImage = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3)
        warp_gradxImage = cv2.warpAffine(gradxImage, affineMatrix, templateROI.shape)
        warp_gradyImage = cv2.warpAffine(gradyImage, affineMatrix, templateROI.shape)
        x, y = np.meshgrid(np.arange(warp_gradxImage.shape[1]), np.arange(warp_gradxImage.shape[0]))
        jacobian = np.array([[x, np.zeros_like(x), y, np.zeros_like(x), np.ones_like(x), np.zeros_like(x)],
                              [np.zeros_like(x), x, np.zeros_like(x), y, np.zeros_like(x), np.ones_like(x)]], dtype = np.float32)
        warp_gradientImage = np.stack((warp_gradxImage, warp_gradyImage), axis = 0)
        steepestDescent = np.einsum('jhw,jihw->ihw', warp_gradientImage, jacobian)
        # steepestDescent = np.diag(np.dot(warp_gradientImage, jacobian.T))
        steepestDescent = steepestDescent.reshape(6,-1)
        hessian = np.dot(steepestDescent.T, steepestDescent)
        print(hessian)
        deltaParameters = np.dot(np.linalg.inv(hessian), np.dot(steepestDescent.T, error.ravel()))
        deltaParameters = deltaParameters.reshape(parameters.shape)
        parameters = parameters + deltaParameters
        
        if np.linalg.norm(deltaParameters) < 0.05:
            break
        
        iterations = iterations + 1
        
    return parameters

def main():
    imagePath = "Data/Bolt2/img/*.jpg"
    imageName = [image for image in glob.glob(imagePath)]
    imageName.sort()
    rectFile = open("Data\Bolt2\groundtruth_rect.txt", "r")
    groundTruth = rectFile.readlines()
    templateBBOX = groundTruth[0].rstrip().split(',')
    templateBBOX = list(map(int, templateBBOX))
    template = cv2.imread(imageName[0])
    testImage = cv2.imread(imageName[1])
    template = CLAHE(template)
    testImage = CLAHE(testImage)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    parameters = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)
    parameters = affineLKTracker(testImage, template, templateBBOX, parameters)
    print(parameters)
    

if __name__ == '__main__':
    main()
    







































'''
threshold = 0.1

def transform(image, parameters):
    affineMatrix = np.array([[parameters[0] + 1, parameters[2], parameters[4]], 
                             [parameters[1], parameters[3] + 1, parameters[5]]], 
                             dtype = np.float32)
    affineImage = cv2.warpAffine(image, affineMatrix, image.shape[::-1])
    return affineImage

def jacobianMatrix(x,y):
    jacobian = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]], dtype=np.float32)
    return jacobian


def affineLKtracker(image, template, templateBBOX, parameters):
    deltaParameters = np.array([1, 1, 1, 1, 1, 1], dtype = np.float32)
    templateROI = template[templateBBOX[1]:templateBBOX[1] + templateBBOX[3], templateBBOX[0]:templateBBOX[0] + templateBBOX[2]]
    while np.linalg.norm(deltaParameters) > 0.1:
        gradientX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3)
        gradientY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3)
        affineImage = transform(image, parameters)
        affine_gradientX = transform(gradientX, parameters)
        affine_gradientY = transform(gradientY, parameters)
        affineImage_ROI = affineImage[templateBBOX[1]:templateBBOX[1] + templateBBOX[3], 
                                      templateBBOX[0]:templateBBOX[0] + templateBBOX[2]]
        affinegradientX_ROI = affine_gradientX[templateBBOX[1]:templateBBOX[1] + templateBBOX[3], 
                                      templateBBOX[0]:templateBBOX[0] + templateBBOX[2]]
        affinegradientY_ROI = affine_gradientY[templateBBOX[1]:templateBBOX[1] + templateBBOX[3], 
                                      templateBBOX[0]:templateBBOX[0] + templateBBOX[2]]
        error = (templateROI - affineImage_ROI).reshape(-1, 1)
        steepestDescent = [np.zeros(shape=templateROI.shape) for _ in range(6)]
        # steepestDescent = np.zeros((templateROI.shape[0]*templateROI.shape[1], 6))
        for y in range(templateROI.shape[0]):
            for x in range(templateROI.shape[1]):
                jacobian = np.dot(np.array([affinegradientX_ROI[y, x], affinegradientY_ROI[y, x]]), 
                                  jacobianMatrix(x + templateBBOX[0], y + templateBBOX[1]))
                for k in range(6):
                    steepestDescent[k][y, x] = jacobian[k]
                    
        steepestDescent_flatten = np.array([steepestDescent[k].flatten() for k in range(6)])
        hessianMatrix = np.dot(steepestDescent_flatten, steepestDescent_flatten.T)
        steepestDescent_parameters = np.dot(steepestDescent_flatten, error)
        deltaParameters = np.dot(np.linalg.inv(hessianMatrix), steepestDescent_parameters)
        print(deltaParameters.shape)
        print(parameters.shape)
        print(parameters)
        parameters = parameters + deltaParameters
        print(parameters)
    
    return parameters
        

def main():
    imagePath = "Data/Bolt2/img/*.jpg"
    imageName = [image for image in glob.glob(imagePath)]
    imageName.sort()
    rectFile = open("Data\Bolt2\groundtruth_rect.txt", "r")
    groundTruth = rectFile.readlines()
    templateBBOX = groundTruth[0].rstrip().split(',')
    templateBBOX = list(map(int, templateBBOX))
    template = cv2.imread(imageName[0])
    clahe = cv2.createCLAHE(clipLimit = 2., tileGridSize = (1, 1))
    labTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(labTemplate)
    b = clahe.apply(b)
    labTemplate = cv2.merge((l, a, b))
    template = cv2.cvtColor(labTemplate, cv2.COLOR_LAB2BGR)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    parameters = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)
    for name in imageName:
        image = cv2.imread(name)
        clahe = cv2.createCLAHE(clipLimit = 2., tileGridSize = (1, 1))
        labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(labImage)
        b = clahe.apply(b)
        labImage = cv2.merge((l, a, b))
        image = cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = (image*(np.mean(template)/np.mean(image))).astype(float)
        parameters = affineLKtracker(image, template, templateBBOX, parameters)
        transformMatrix = np.array([[parameters[0] + 1, parameters[2], parameters[4]], 
                             [parameters[1], parameters[3] + 1, parameters[5]]], 
                             dtype = np.float32)
        print(parameters.shape)
        print(transformMatrix.shape)
        transformMatrix = np.vstack((np.array([[parameters[0] + 1, parameters[2], parameters[4]], 
                             [parameters[1], parameters[3] + 1, parameters[5]]]), [0, 0, 1])
        bbox_1 = np.dot(transformMatrix, np.array([templateBBOX[0], templateBBOX[1], 1]).T)
        bbox_2 = np.dot(transformMatrix, np.array([templateBBOX[0] + templateBBOX[2], templateBBOX[1] + templateBBOX[3], 1]).T)
        print(bbox_1)
        print(bbox_2)        
    # cv2.imshow('template', templateROI)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()

'''