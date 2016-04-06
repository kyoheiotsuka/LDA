# -*- coding: utf-8 -*-
import numpy as np
import cv2, os, lda


if __name__ == "__main__":

	# set parameter for experiment
	nTopics = 8

	# create folder for saving result
	if not os.path.exists("result"):
		os.mkdir("result")

	# create folder for showing fitting process
	if not os.path.exists("visualization"):
		os.mkdir("visualization")

	# load image files (created by createImage.py)
	data = np.zeros((1000,16),dtype=np.uint8)
	for i in range(1000):
		image =  cv2.resize(cv2.imread("image/%d.jpg"%i,0),(4,4),interpolation=cv2.INTER_NEAREST)
		data[i,:] = image.reshape((16)).astype(np.uint8)
		
	# apply latent dirichlet allocation
	model = lda.LDA()
	model.setData(data)
	model.solve(nTopics=nTopics)

	# show topics obtained
	for i in range(8):
	    topic = model.qPhi[i,:]
	    topic = topic/topic.max()*255
	    topic = topic.reshape((4,4))

	    cv2.imwrite("result/%d.bmp"%i,cv2.resize(topic.astype(np.uint8),(200,200),interpolation=cv2.INTER_NEAREST))

