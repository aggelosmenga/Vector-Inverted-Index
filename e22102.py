import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
#retrieve files and add them to a numpy array
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


#inverted index creation
def invertedindex(S,train,nofclusters):

    centroids,_=kmeans2(train,nofclusters,minit='points')

    invertedindex={i: [] for i in range(nofclusters)}

    chunks=train.shape[0]

    #χωρισμος των δεδομενων σε chunks
    for i in range(0,S.shape[0],chunks):
       
        chunk=S[i :i+chunks]

        distances=cdist(chunk,centroids,metric='euclidean')
       
       #closest centroids
        nearest=np.argmin(distances,axis=1)
        
        for j, centroid_id in enumerate(nearest):
            invertedindex[centroid_id].append(i + j)

    
    for i in range(nofclusters):
        invertedindex[i]=np.array(invertedindex[i],dtype=np.int32)

    return centroids,invertedindex

def nearestcentroids(q,centroids :np.ndarray,M:int):
    #returns index of closest centroids
    dist=np.linalg.norm(centroids-q,axis=1)
    ind=np.argmin(dist)[:M] #returns m closest centroids
    print(min(dist))
    return ind

def ApproximateNearestNeighbors(q,index:dict,centroids:np.ndarray,S:np.ndarray,k):
    
    clusters=nearestcentroids(q,centroids,M=4)
    basekeys=index[clusters]
    
    vectorsforcheck=S[basekeys]

    distances=np.linalg.norm(vectorsforcheck-q,axis=1)

    vectorindices=np.argsort(distances)[:k]
    return basekeys[vectorindices]
    
def PreciseNearestNeighbors(q,index:dict, centroids:np.ndarray,S:np.ndarray,k):

    return


dimension=128
clusters = 80 #based on elbow rule

S=sift_base=retrievefvecs("sift/sift_base.fvecs")
train=sift_learn=retrievefvecs("sift/sift_learn.fvecs")
Q=sift_query=retrievefvecs("sift/sift_query.fvecs")
ground_truth=sift_groundtruth=retriveivecs("sift/sift_groundtruth.ivecs")


centroids,index= invertedindex(S,train,clusters)
#inverted index krataei ta keys toy S pinaka !!!!!!
print(nearestcentroids(Q[0],centroids,M=4))
test=ApproximateNearestNeighbors(Q[0],index,centroids,S,k=5)
print(S[test])
approximateresults=S[test]
print(approximateresults.shape)