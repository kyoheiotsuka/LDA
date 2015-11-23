# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
import time, cPickle


class LDA:
    # Variational implimentation of smoothed LDA

    def __init__(self):

        # Initialize class 
        pass
    
    def setData(self,data):
        # Data is required to be given in a three dimensional numpy array, [nDocuments,nVocabulary,nObserved]
        
        # Set additional parameters
        self.nDocuments = data.shape[0]
        self.nVocabulary = data.shape[1]
        self.data = data
    
    def solve(self,nTopics,epsilon=1e-4,alpha=1.0,beta=0.01):
        
        # Set additional parameters
        self.nTopics = nTopics
        self.epsilon = epsilon
        
        # Prior distribution for alpha and beta
        self.alpha = np.full(self.nTopics,alpha,dtype=np.float32)
        self.beta = np.full(self.nVocabulary,beta,dtype=np.float32)
        
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
            
            deltaMax = 0.0
            tic = time.clock()

            # Update qPhi
            qPhi = self.qPhi[:,:]
            qPhi[:] = np.tile(self.beta.reshape((1,self.nVocabulary)),(self.nTopics,1))-1.0
            for d in range(self.nDocuments):
                doc = self.data[d,:,:]
                qZ = self.qZ[d,:,:]
                qPhi += (qZ[:,:] * doc[:,1].reshape((doc.shape[0],1))).T
            phiExpLog = scipy.special.psi(self.qPhi[:,:]+1.0)
            phiExpLog -= np.tile(scipy.special.psi(self.qPhi[:,:].sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))
            
            # Iterate documents
            for d in range(self.nDocuments):
                doc = self.data[d,:,:]
                qZ = self.qZ[d,:,:]
                qTheta = self.qTheta[d,:]
                qThetaNew = self.qThetaNew[d,:]
                
                # Update qTheta
                if nIteration == 0:
                    qTheta[:] = self.alpha-1.0
                    qTheta += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta+1.0)
                thetaExpLog -= scipy.special.psi(qTheta.sum())

                # Update qZ
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1)))
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))
                
                # Measure amount of change
                qThetaNew[:] = self.alpha-1.0
                qThetaNew += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc[:,1].sum()
                deltaMax = max(deltaMax,delta)
            
            # Break if converged
            if deltaMax<self.epsilon:
                break

            toc = time.clock()
            print (nIteration,deltaMax,tic-toc)
            nIteration += 1
            
        return

    def save(self,name):
        # Save Object
        with open(name,"wb") as output:
            cPickle.dump(self.__dict__,output,protocol=cPickle.HIGHEST_PROTOCOL)
        return

    def load(self,name):
        # Load Object
        with open(name,"rb") as input:
            self.__dict__.update(cPickle.load(input))
        
    def predict(self,dataPredict):
        # Data to predict is required to be given in a three dimensional numpy array, [nDocuments,nVocabulary,nObserved]

        # Utilize topic information with training data
        phiExpLog = scipy.special.psi(self.qPhi[:,:]+1.0)
        phiExpLog -= np.tile(scipy.special.psi(self.qPhi[:,:].sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))
        
        # Define parameters
        nDataPredict = dataPredict.shape[0]
        qThetaPredict = np.empty((nDataPredict,self.nTopics),dtype=np.float32)
        qThetaPredictNew = np.empty((nDataPredict,self.nTopics),dtype=np.float32)
        qZPredict = np.random.rand(nDataPredict,self.nVocabulary,self.nTopics)
        for i in range(qZPredict.shape[0]):
            qZPredict[i] /= qZPredict[i].sum(axis=1).reshape((qZPredict[i].shape[0],1))
            
        # Start predicting
        nIteration = 0
        while(1):

            deltaMax = 0.0
            tic = time.clock()

            # Iterate documents to predict
            for d in range(nDataPredict):
                doc = dataPredict[d,:,:]
                qZ = qZPredict[d,:,:]
                qTheta = qThetaPredict[d,:]
                qThetaNew = qThetaPredictNew[d,:]

                # Update qTheta
                if nIteration == 0:
                    qTheta[:] = self.alpha-1.0
                    qTheta += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta+1.0)
                thetaExpLog -= scipy.special.psi(qTheta.sum())

                # Update qZ
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1)))
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))

                # Measure amount of change
                qThetaNew[:] = self.alpha-1.0
                qThetaNew += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc[:,1].sum()
                deltaMax = max(deltaMax,delta)

            # Break if converged
            if deltaMax<self.epsilon:
                break

            toc = time.clock()
            print (nIteration,deltaMax,tic-toc)
            nIteration += 1

        return qThetaPredict




