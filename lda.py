# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
import copy, time


class LDA:
    # Variational implimentation of smoothed LDA

    def __init__(self):
        
        # Set values for stopping iteration
        self.epsilon = 1e-4
        self.delta = None
    
    def setData(self,data):
        # Data is required to be given in a three dimensional numpy array, [nDocuments,nVocabulary,nObserved]
        
        # Set additional parameters
        self.nDocuments = data.shape[0]
        self.nVocabulary = data.shape[1]
        self.data = data
    
    def solve(self,nTopics):
        
        # Set additional parameters
        self.nTopics = nTopics
        
        # Prior distribution for alpha and beta
        self.alpha = np.full(self.nTopics,1.0,dtype=np.float32)
        self.beta = np.full(self.nVocabulary,0.01,dtype=np.float32)
        
        # Define q(theta)
        self.qTheta = np.empty((self.nDocuments,self.nTopics),dtype=np.float32)
        self.qThetaNew = np.empty((self.nDocuments,self.nTopics),dtype=np.float32)
        
        # Define q(phi)
        self.qPhi = np.empty((self.nTopics,self.nVocabulary),dtype=np.float32)
        
        # Initialize q(z)
        self.qZ = np.random.rand(self.nDocuments,self.nVocabulary,self.nTopics)
        for i in range(self.qZ.shape[0]):
            self.qZ[i] /= self.qZ[i].sum(axis=1).reshape((self.qZ[i].shape[0],1))
        
        # Start solving using variational Bayes
        nIteration = 0
        while(1):
            
            self.delta = 0.0
            tic = time.clock()
            
            # Update qPhi
            for k in range(self.nTopics):
                qPhi = self.qPhi[k,:]
                qPhi[:] = self.beta
                for d in range(self.nDocuments):
                    doc = self.data[d,:,:]
                    qZ = self.qZ[d,:,:]
                    qPhi += qZ[:,k] * doc[:,1]
            
            # Iterate documents
            for d in range(self.nDocuments):
                
                doc = self.data[d,:,:]
                qZ = self.qZ[d,:,:]
                qTheta = self.qTheta[d,:]
                qThetaNew = self.qThetaNew[d,:]
                
                # Update qTheta
                if nIteration==0:
                    qTheta[:] = self.alpha
                    qTheta += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                
                # Update qZ
                thetaExpLog = scipy.special.psi(qTheta)
                thetaExpLog -= scipy.special.psi(qTheta.sum())
                phiExpLog = scipy.special.psi(self.qPhi[:,:])
                phiExpLog -= np.tile(scipy.special.psi(self.qPhi[:,:].sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1)))
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))
                
                # Measure amount of change
                qThetaNew[:] = self.alpha
                qThetaNew += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc[:,1].sum()
                self.delta = max(self.delta,delta)
            
            toc = time.clock()
            print (nIteration,self.delta,tic-toc)
            nIteration += 1
            
            # Break if converged
            if self.delta<self.epsilon:
                break
            
        return self.qPhi

















