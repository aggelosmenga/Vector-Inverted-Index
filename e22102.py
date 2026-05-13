import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def retrievefvecs(file):

    data=np.fromfile(file,dtype='float32')
    
    dimension=data.view('int32')[0]
    
    data=data.reshape(-1,dimension + 1)

    return data[:, 1:].copy()

def retriveivecs(file):
    data = np.fromfile(file, dtype='int32')

    dimension=data[0]

    data=data.reshape(-1,dimension+1)

    return data[:, 1:].copy()

def invertedindex(S):

    samplesize=min(10000,S.shape[0])
    trainingsample = S[np.random.permutation(S.shape[0])[:samplesize]]

    centroids,_=kmeans2(trainingsample,80,minit='points')

    invertedindex={i: [] for i in range(80)}

    chunks=100000

    for i in range(0,S.shape[0],chunks):
        
        chunk=S[i:i + chunks]

        distances=cdist(chunk,centroids,metric='euclidean')
        #closest centroids
        nearest=np.argmin(distances,axis=1)
        
        for j, centroid_id in enumerate(nearest):
            invertedindex[centroid_id].append(i + j)

    
    for i in range(80):
        invertedindex[i]=np.array(invertedindex[i],dtype=np.int32)

    return centroids,invertedindex



S=sift_base=retrievefvecs("sift/sift_base.fvecs")
train=sift_learn=retrievefvecs("sift/sift_learn.fvecs")
Q=sift_query=retrievefvecs("sift/sift_query.fvecs")
ground_truth=sift_groundtruth=retriveivecs("sift/sift_groundtruth.ivecs")



centroids,index = invertedindex(S)
print(type(index))