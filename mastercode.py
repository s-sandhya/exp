# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:06:24 2019

@author: ssand
"""

from scipy.spatial.distance import cityblock, euclidean
from scipy import linalg
import numpy as np
import pandas
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
import itertools
import scipy
from sklearn import mixture


def evaluateEER(genuine_scores, intr_scores,auc_arr):
    labels = [0]*len(genuine_scores) + [1]*len(intr_scores)
    fpr, tpr, thresholds = roc_curve(labels, genuine_scores + intr_scores)
    auc_val=auc(fpr,tpr)
    auc_arr.append(auc_val)
    plotgraph(fpr,tpr,auc_val)
    misses = 1 - tpr
    false_ala = fpr

    diffarr = misses - false_ala
    idx1 = np.argmin(diffarr[diffarr >= 0])
    idx2 = np.argmax(diffarr[diffarr < 0])
    
    x = [misses[idx1], false_ala[idx1]]
    y = [misses[idx2], false_ala[idx2]]
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
    eer = x[0] + a * ( y[0] - x[0] )
    return eer


def evaluateEERGMM(genuine_scores, intr_scores,auc_arr):
    thresholds = range(21,50)
    array = np.zeros((len(thresholds),3))
    i = 0
    
    for th in thresholds:
        cnt1 = 0
        cnt2 = 0
        
        for score in genuine_scores:
            if score < th:
                cnt1 = cnt1 + 1  
                
        for score in intr_scores:    
            if score > th:
                cnt2 = cnt2 + 1
            
        FA = float(cnt2) / len(intr_scores) 
        FR = float(cnt1) / len(genuine_scores)
        array[i, 0] = th
        array[i, 1] = FA
        array[i, 2] = FR
        i = i + 1
        
    for j in range(array.shape[0]):
        if array[j,1] < array[j,2]:
            thresh = (array[j,0] + array[j - 1, 0]) / 2
            break
    cnt1 = 0
    cnt2 = 0
    for score in genuine_scores:
        if score < thresh:
            cnt1 = cnt1 + 1
          
    for score in intr_scores:    
        if score > thresh:
            cnt2 = cnt2 + 1
   
    FA = float(cnt2) / len(intr_scores) 
    FR = float(cnt1) / len(genuine_scores)
    labels = [0]*len(genuine_scores) + [1]*len(intr_scores)
    fpr, tpr, thresholds = roc_curve(labels, genuine_scores + intr_scores)
    auc_val=auc(tpr,fpr)
    auc_arr.append(auc_val)
    plotgraph(tpr,fpr,auc_val)
    return (FA + FR) /2

def plotgraph(fpr,tpr,auc_val):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



def manhattan_evaluate(subjects,auc_arr):
    genuine_scores = []
    intr_scores = []
        
    eers = []
 
    for subject in subjects:        
        genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
        intr_data = data.loc[data.subject != subject, :]
        
        train = genuine_data[:200]
        test_genuine = genuine_data[200:]
        test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
 
        mean_vector = train.mean().values         
        
        for i in range(test_genuine.shape[0]):
            cur_score = cityblock(test_genuine.iloc[i].values,mean_vector)
            genuine_scores.append(cur_score)
 
        for i in range(test_imposter.shape[0]):
            cur_score = cityblock(test_imposter.iloc[i].values,mean_vector)
            intr_scores.append(cur_score)

        eers.append(evaluateEER(genuine_scores,intr_scores,auc_arr))
    return np.mean(eers), np.std(eers),np.mean(auc_arr)    
      
       
def filtered_manhattan_evaluate(subjects,auc_arr):
    
    eers = []
    genuine_scores = []
    intr_scores = []
    
 
    for subject in subjects:        
        genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
        intr_data = data.loc[data.subject != subject, :]
        
        train = genuine_data[:200]
        test_genuine = genuine_data[200:]
        test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
 
        mean_vector =train.mean().values
        std_vector =train.std().values
        droppincnt1ndices = []
        for i in range(train.shape[0]):
            cur_score = euclidean(train.iloc[i].values, 
                                   mean_vector)
            if (cur_score > 3*std_vector).all() == True:
                droppincnt1ndices.append(i)
        train = train.drop(train.index[droppincnt1ndices])
        mean_vector =train.mean().values    
        
        for i in range(test_genuine.shape[0]):
            cur_score = cityblock(test_genuine.iloc[i].values,mean_vector)
            genuine_scores.append(cur_score)
 
        for i in range(test_imposter.shape[0]):
            cur_score = cityblock(test_imposter.iloc[i].values,mean_vector)
            intr_scores.append(cur_score)
            
        eers.append(evaluateEER(genuine_scores,intr_scores,auc_arr))
    return np.mean(eers), np.std(eers), np.mean(auc_arr) 

def scaled_manhattan_evaluate(subjects,auc_arr):
    
    eers = []
    genuine_scores = []
    intr_scores = []
    
    for subject in subjects:        
        genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
        intr_data = data.loc[data.subject != subject, :]
        
        train = genuine_data[:200]
        test_genuine = genuine_data[200:]
        test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
 
        mean_vector =train.mean().values
        mad_vector  = train.mad().values
        
        for i in range(test_genuine.shape[0]):
            cur_score = 0
            for j in range(len(mean_vector)):
                cur_score = cur_score +abs(test_genuine.iloc[i].values[j] -mean_vector[j]) /mad_vector[j]
            genuine_scores.append(cur_score)
 
        for i in range(test_imposter.shape[0]):
            cur_score = 0
            for j in range(len(mean_vector)):
                cur_score = cur_score +abs(test_imposter.iloc[i].values[j] -mean_vector[j]) /mad_vector[j]
            intr_scores.append(cur_score)
        eers.append(evaluateEER(genuine_scores,intr_scores,auc_arr))
    return np.mean(eers), np.std(eers), np.mean(auc_arr)      


def eucledian_evaluate(subjects,auc_arr):
    
    eers = []
    genuine_scores = []
    intr_scores = []
    
    for subject in subjects:        
        genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
        intr_data = data.loc[data.subject != subject, :]
        
        train = genuine_data[:200]
        test_genuine = genuine_data[200:]
        test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
 
        mean_vector = train.mean().values         
        
        for i in range(test_genuine.shape[0]):
            cur_score = euclidean(test_genuine.iloc[i].values, 
                                   mean_vector)
            genuine_scores.append(cur_score)
      
        for i in range(test_imposter.shape[0]):
            cur_score = euclidean(test_imposter.iloc[i].values,mean_vector)
            intr_scores.append(cur_score)

        eers.append(evaluateEER(genuine_scores,intr_scores, auc_arr))
    return np.mean(eers), np.std(eers), np.mean(auc_arr)   
        
                  

def normed_eucledian_evaluate(subjects,auc_arr):
    
    eers = []
    genuine_scores = []
    intr_scores = []
    
    for subject in subjects:        
        genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
        intr_data = data.loc[data.subject != subject, :]
        
        train = genuine_data[:200]
        test_genuine = genuine_data[200:]
        test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
 
        mean_vector = train.mean().values         
        
        for i in range(test_genuine.shape[0]):
            cur_score = euclidean(test_genuine.iloc[i].values, 
                                   mean_vector)
            n1=linalg.norm(test_genuine.iloc[i].values)
            n2=linalg.norm(mean_vector)
            genuine_scores.append(cur_score/(n1*n2))
      
        for i in range(test_imposter.shape[0]):
            cur_score = euclidean(test_imposter.iloc[i].values,mean_vector)
            n1=linalg.norm(test_imposter.iloc[i].values)
            n2=linalg.norm(mean_vector)
            intr_scores.append(cur_score/(n1*n2))

        eers.append(evaluateEER(genuine_scores,intr_scores,auc_arr))
    return np.mean(eers), np.std(eers), np.mean(auc_arr)    
        
def svm_evaluate(subjects,auc_arr):
    
    eers = []
    genuine_scores = []
    intr_scores = []
    
    for subject in subjects: 
        genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
        intr_data = data.loc[data.subject != subject, :]
        
        train = genuine_data[:200]
        test_genuine = genuine_data[200:]
        test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]

        clf = OneClassSVM(kernel='rbf',gamma=24)
        clf.fit(train)
        
        genuine_scores = -clf.decision_function(test_genuine)
        intr_scores = -clf.decision_function(test_imposter)
        genuine_scores = list(genuine_scores)
        intr_scores = list(intr_scores)
     
        eers.append(evaluateEER(genuine_scores,intr_scores,auc_arr))
    return np.mean(eers), np.std(eers), np.mean(auc_arr) 



def gmm_evaluate(subjects,auc_arr):
    eers = []
    genuine_scores = []
    intr_scores = []
    
    for subject in subjects: 
        genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
        intr_data = data.loc[data.subject != subject, :]
        
        train = genuine_data[:200]
        test_genuine = genuine_data[200:]
        test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
                             
        gmm = GaussianMixture(n_components = 1, covariance_type = 'spherical', 
                            verbose = False )
        gmm.fit(train)
    
        for i in range(test_genuine.shape[0]):
            j = test_genuine.iloc[i].values
            j=j.reshape(1,-1)
            cur_score = gmm.score(j)
            genuine_scores.append(cur_score)
    
        for i in range(test_imposter.shape[0]):
            j = test_imposter.iloc[i].values
            j=j.reshape(1,-1)
            cur_score = gmm.score(j)
            intr_scores.append(cur_score)        
         
        eers.append(evaluateEERGMM(genuine_scores, intr_scores,auc_arr))
    return np.mean(eers),np.std(eers),np.mean(auc_arr)   

path ="D:\\fdrive\\sem8\\advanced analytics\\ca3\\package\\DSL-StrongPasswordData.csv"
data = pandas.read_csv(path)
subjects = data["subject"].unique()

auc_arr=[]
print("average EER for Eucledian ")
eucledian_evaluate(subjects,auc_arr)

auc_arr=[]
print("average EER for NormedEucledian")
normed_eucledian_evaluate(subjects,auc_arr)

auc_arr=[]
print("average EER for Manhattan")
manhattan_evaluate(subjects,auc_arr)

auc_arr=[]
print("average EER for filtered Manhattan")
filtered_manhattan_evaluate(subjects,auc_arr)

auc_arr=[]
print("average EER for scaled Manhattan")
scaled_manhattan_evaluate(subjects,auc_arr)

auc_arr=[]
print("average EER for One class SVM")
svm_evaluate(subjects,auc_arr)

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']

subject='s042'
genuine_data = data.loc[data.subject == subject,"H.period":"H.Return"]
intr_data = data.loc[data.subject != subject, :]
train = genuine_data[:200]
test_genuine = genuine_data[200:]
test_imposter = intr_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(train)
        bic.append(gmm.bic(train))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
bic=scipy.stats.zscore(bic)
bic=bic+max(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +.2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

auc_arr=[]
print("average EER for GMM")
gmm_evaluate(subjects,auc_arr)
