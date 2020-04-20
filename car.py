# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:23:43 2020

@author: Arun
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
import glob




def template_create(gray_img,points):
    ### crop = img[y1:y2, x1:x2]
    cropped = gray_img[points[0][1]:points[1][1], points[0][0]:points[1][0]]
    return cropped

def warp_affine(frame,M):

    warped = cv2.warpAffine(frame,M,(frame.shape[0],frame.shape[1]))

    return warped,M

def warp_paramters(P,points):

    p1 = P[0]
    p2 = P[1]
    p3 = P[2]
    p4 = P[3]
    p5 = P[4]
    p6 = P[5]

    x_start = points[0][0]
    x_end = points[1][0]
    y_start = points[0][1]
    y_end = points[1][1]

    my_x = np.asarray(list(range(x_start,x_end)))
    my_y = np.asarray(list(range(y_start,y_end)))

    mesh_x,mesh_y = np.meshgrid(my_x,my_y,indexing='ij')

    mesh_x = mesh_x.flatten()   ## x
    mesh_y = mesh_y.flatten()   ## y

    p1_x = np.multiply(1+p1,mesh_y)
    p3_y = np.multiply(p3,mesh_x)
    p5_1 = np.multiply(p5,np.ones(mesh_x.shape[0],))
    p2_x  = np.multiply(p2,mesh_y)
    p4_y  = np.multiply(1+p4,mesh_x)
    p6_1 = np.multiply(p6,np.ones(mesh_y.shape[0],))

    x = p1_x+p3_y+p5_1
    y = p2_x+p4_y+p6_1

    w_new = np.array([x,y]).T

    warp_mat = []

    return w_new

def warped_image(warp_mat,frame,points):

    x_start = points[0][0]
    x_end = points[1][0]
    y_start = points[0][1]
    y_end = points[1][1]

    I_gray = []

    for iter in warp_mat:
        I_gray.append(frame[int(iter[0]),int(iter[1])])

    I_gray_image = np.reshape(np.asarray(I_gray),[y_end-y_start,x_end-x_start], order='F' )

    return I_gray_image

# remove the template from the warped image to obtain the error
    
def error(template,I):
    error = template-I
    return error

def image_gradients(img):

    gradient_sobelx = (cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5))
    gradient_sobely = (cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5))

    return gradient_sobelx,gradient_sobely

# obtian the Jacobian matrix
def jacobian(x,y):
    dw_dp = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
    return dw_dp

def iterative(gray1,points,T,P,count):

    iter = 0

    norm = 10


    steepest_descent = np.zeros([T.shape[0]*T.shape[1],6])

    hessian = np.zeros([6,6])



    while(norm >= 0.05):# and norm <= 0.00):

        warp_params =  warp_paramters(P,points)

        warped_im = warped_image(warp_params,gray1,points)

        # cv2.imshow('warped',warped_im)

        error_image = error(T,warped_im)

        error_image_reshaped = np.reshape(error_image,(error_image.shape[0]*error_image.shape[1],1))

        grad_x , grad_y = image_gradients(gray1)

        ### obtaining the gradients of cropped images in x and y
        warp_grad_x  = warped_image(warp_params,grad_x,points)
        warp_grad_y  = warped_image(warp_params,grad_y,points)


        grad_x_flatten = warp_grad_x.flatten()      ##I_x
        grad_y_flatten = warp_grad_y.flatten()      ##I_y


        my_x = np.asarray(list(range(0,warp_grad_x.shape[1])))
        my_y = np.asarray(list(range(0,warp_grad_x.shape[0])))

        mesh_x,mesh_y = np.meshgrid(my_x,my_y,indexing='ij')

        mesh_x = mesh_x.flatten()   ## x
        mesh_y = mesh_y.flatten()   ## y

        ix_x = np.multiply(grad_x_flatten.T,mesh_x.T)
        iy_x = np.multiply(grad_y_flatten.T,mesh_x.T)
        ix_y = np.multiply(grad_x_flatten.T,mesh_y.T)
        iy_y = np.multiply(grad_y_flatten.T,mesh_y.T)
        ix = grad_x_flatten.T
        iy = grad_y_flatten.T

        steepest_descent = steepest_descent + np.array([ix_x,iy_x,ix_y,iy_y,ix,iy]).T

        hessian =  hessian + np.dot(steepest_descent.T,steepest_descent)

        ### applying the inverse hessian to the "hessian of steepest gradient descent"
        delta_p =  np.matmul(np.linalg.pinv(hessian), np.matmul(steepest_descent.T,error_image_reshaped))

        norm = np.linalg.norm(delta_p)

        #
        # # # ## updating the P
        if iter >3 and count > 3:
            gamma = np.diag([0.5, 0.5, 0.1, 0.1, -0.0005, 0.0005])

        else:
            gamma = np.diag([0.5,0.5,0.1,0.1,-0.001,0.001])

        P = P + np.dot(gamma, delta_p)

        iter+=1
    return P,iter,warped_im

img1 = cv2.imread('car/frame0020.jpg')  ### image at time step 1

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# 290:360, 250:288
points = [[120,90],[340,280]]

P = np.array([[0.001],[0.001],[0.0001],[-0.0001],[1],[1]])
template_image = template_create(gray1,points)

count = 0
all_frames = []
######## obtain the image
filenames = [img for img in glob.glob("car/*.jpg")]
filenames.sort()
for img in filenames:
    n = cv2.imread(img)
    # obtain the gray scale
    count+=1

    gray1 = cv2.cvtColor(n,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('vid',gray1)
    # cv2.waitKey(0)
    print(gray1.shape)

    # print(asdc)
    P,iter, warped_img= iterative(gray1,points,template_image,P,count)

    print(P,count,iter)
    p1,p2,p3,p4,p5,p6 = P[0,0], P[1,0], P[2,0], P[3,0], P[4,0], P[5,0]

    W = np.matrix([[1+p1, p3, p5],[p2, 1+p4, p6]])

    # print(warped_img.shape[0],warped_img.shape[1])

    new_mat_1 = np.array([[points[0][0]],[points[0][1]],[1]])#.astype(int)
    new_mat_2 = np.array([[points[1][0]],[points[1][1]],[1]])#.astype(int)

    wx1 = np.matmul(W,new_mat_1).astype(int)
    wx2 = np.matmul(W,new_mat_2).astype(int)

    cv2.rectangle(n,(wx1[0],wx1[1]),(wx2[0],wx2[1]),color=255)
    all_frames.append(n)

    cv2.imshow('gray1',n)
    cv2.waitKey(0)


print('done video')

source = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_car.avi', source, 5.0, (720, 480))
for iter1 in all_frames:
    out.write(iter1)
    cv2.waitKey(15)
    out.release()