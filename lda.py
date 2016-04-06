# -*- coding: utf-8 -*-
import numpy as np
import scipy.special
import time, cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class LDA:
    # variational implimentation of smoothed LDA

    def __init__(self):
        # do nothing particularly
        pass
    
    def setData(self,data):
        # data is required to be given in a two dimensional numpy array (nDocuments,nVocabulary)
        # with each element representing the number of times observed
        
        # set parameters
        self.data = data
        self.nDocuments = data.shape[0]
        self.nVocabulary = data.shape[1]
    
    def solve(self,nTopics,epsilon=1e-3,alpha=1.0,beta=0.01):
        
        # set additional parameters
        self.nTopics = nTopics
        self.epsilon = epsilon
        
        # prior distribution for alpha and beta
        self.alpha = np.full(self.nTopics,alpha,dtype=np.float64)
        self.beta = np.full(self.nVocabulary,beta,dtype=np.float64)
        
        # define q(theta)
        self.qTheta = np.empty((self.nDocuments,self.nTopics),dtype=np.float64)
        self.qThetaNew = np.empty((self.nDocuments,self.nTopics),dtype=np.float64)
        
        # define q(phi)
        self.qPhi = np.empty((self.nTopics,self.nVocabulary),dtype=np.float64)
        
        # define and initialize q(z)
        self.qZ = np.random.rand(self.nDocuments,self.nVocabulary,self.nTopics)
        for i in range(self.qZ.shape[0]):
            self.qZ[i] /= self.qZ[i].sum(axis=1).reshape((self.qZ[i].shape[0],1))
        
        # start solving using variational Bayes
        nIteration = 0
        while(1):
            
            deltaMax = 0.0
            tic = time.clock()

            # update qPhi
            qPhi = self.qPhi[:,:]
            qPhi[:] = np.tile(self.beta.reshape((1,self.nVocabulary)),(self.nTopics,1))
            for d in range(self.nDocuments):
                doc = self.data[d,:]
                qZ = self.qZ[d,:,:]
                qPhi += (qZ[:,:] * doc.reshape((doc.shape[0],1))).T
            phiExpLog = scipy.special.psi(self.qPhi[:,:])
            phiExpLog -= np.tile(scipy.special.psi((self.qPhi[:,:]).sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))
            
            # iterate throught all documents
            for d in range(self.nDocuments):
                doc = self.data[d,:]
                qZ = self.qZ[d,:,:]
                qTheta = self.qTheta[d,:]
                qThetaNew = self.qThetaNew[d,:]
                
                # update qTheta
                if nIteration == 0:
                    qTheta[:] = self.alpha
                    qTheta += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta)
                thetaExpLog -= scipy.special.psi((qTheta).sum())

                # update qZ
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1)))
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))
                
                # measure amount of change
                qThetaNew[:] = self.alpha
                qThetaNew += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc.sum()
                deltaMax = max(deltaMax,delta)
            
            # break if converged
            if deltaMax<self.epsilon:
                break

            # display information
            toc = time.clock()
            self.heatmap(nIteration)
            print "nIteration=%d, delta=%f, time=%.5f"%(nIteration,deltaMax,toc-tic)
            nIteration += 1

    def predict(self,dataPredict):
        # dataPredict is required to be given in a two dimensional numpy array (nDocuments,nVocabulary)
        # with each element representing the number of times observed

        # set additional parameters
        nDataPredict = dataPredict.shape[0]

        # utilize topic information with training data
        phiExpLog = scipy.special.psi(self.qPhi[:,:])
        phiExpLog -= np.tile(scipy.special.psi((self.qPhi[:,:]).sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))
        
        # define q(theta) for unseen data
        qThetaPredict = np.empty((nDataPredict,self.nTopics),dtype=np.float32)
        qThetaPredictNew = np.empty((nDataPredict,self.nTopics),dtype=np.float32)

        # define and initialize q(z) for unseen data
        qZPredict = np.random.rand(nDataPredict,self.nVocabulary,self.nTopics)
        for i in range(qZPredict.shape[0]):
            qZPredict[i] /= qZPredict[i].sum(axis=1).reshape((qZPredict[i].shape[0],1))
            
        # start prediction
        nIteration = 0
        while(1):

            deltaMax = 0.0
            tic = time.clock()

            # iterate over all documents
            for d in range(nDataPredict):
                doc = dataPredict[d,:]
                qZ = qZPredict[d,:,:]
                qTheta = qThetaPredict[d,:]
                qThetaNew = qThetaPredictNew[d,:]

                # update qTheta for unseen data
                if nIteration == 0:
                    qTheta[:] = self.alpha
                    qTheta += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta)
                thetaExpLog -= scipy.special.psi((qTheta).sum())

                # update qZ for unseen data
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1)))
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))

                # measure amount of change
                qThetaNew[:] = self.alpha
                qThetaNew += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc.sum()
                deltaMax = max(deltaMax,delta)

            # break if converged
            if deltaMax<self.epsilon:
                break

            # display information
            toc = time.clock()
            print (nIteration,deltaMax,toc-tic)
            nIteration += 1

        return qThetaPredict

    def heatmap(self,nIteration):
        # save heatmap image of topic-word distribution
        topicWordDistribution = self.qPhi/self.qPhi.sum(axis=1).reshape((self.nTopics,1))

        plt.clf()
        fig,ax = plt.subplots()

        # visualize topic-word distribution
        X,Y = np.meshgrid(np.arange(topicWordDistribution.shape[1]+1),np.arange(topicWordDistribution.shape[0]+1))
        image = ax.pcolormesh(X,Y,topicWordDistribution)
        plt.xlim(0,topicWordDistribution.shape[1])
        plt.xlabel("Vocabulary ID")
        plt.ylabel("Topic ID")

        # show colorbar
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="2%",pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(image,cax=ax_cb)
        figure = plt.gcf()
        figure.set_size_inches(16,12)
        plt.tight_layout()

        # save image as a file
        plt.savefig("visualization/nIteration_%d.jpg"%nIteration,dpi=100)
        plt.close()

    def save(self,name):
        # save object as a file
        with open(name,"wb") as output:
            cPickle.dump(self.__dict__,output,protocol=cPickle.HIGHEST_PROTOCOL)

    def load(self,name):
        # load object from a file
        with open(name,"rb") as input:
            self.__dict__.update(cPickle.load(input))
        
