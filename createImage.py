# -*- coding: utf-8 -*-

import numpy as np
import cv2


topic1 = np.array([1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],dtype=np.float32)*255
topic2 = np.array([0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],dtype=np.float32)*255
topic3 = np.array([0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],dtype=np.float32)*255
topic4 = np.array([0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],dtype=np.float32)*255

topic5 = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)*255
topic6 = np.array([0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],dtype=np.float32)*255
topic7 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],dtype=np.float32)*255
topic8 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],dtype=np.float32)*255


for i in range(1000):
    alpha = np.full(8,1,dtype=np.float32)
    theta = np.random.dirichlet(alpha)
    outcome = np.zeros(16,dtype=np.float32)
    outcome += theta[0]*topic1
    outcome += theta[1]*topic2
    outcome += theta[2]*topic3
    outcome += theta[3]*topic4
    outcome += theta[4]*topic5
    outcome += theta[5]*topic6
    outcome += theta[6]*topic7
    outcome += theta[7]*topic8
    cv2.imwrite("image/%d.jpg"%i,outcome.reshape((4,4)).astype(np.uint8))