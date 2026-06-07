import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
import pickle as pkl
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
""""
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

    return centroids,invertedindex"
"""

def ApproximateNearestNeighbors(q:np.ndarray,index:dict,centroids:np.ndarray,S:np.ndarray,k,e=1.5):
    approxresults=[]
    computations=0
    knn = NearestNeighbors(n_neighbors=k,algorithm='brute',n_jobs=-1)
    clusternn= NearestNeighbors(n_neighbors=1000,algorithm='brute',n_jobs=-1)
    for i in range(len(q)):

        clusternn.fit(centroids)
        
        clusterdistances,clusterindices=clusternn.kneighbors(q[i].reshape(1,-1))
        dmin=clusterdistances[0]

        #λογικη πισω απο την αυτοματοποιηση:
            #1+ε Approximate nearest neighbor search για τα κοντινοτερα clusters
            #οριζουμε dmin το κοντινοτερο διανυσμα στο query μας
            #οριζουμε ενα tolerance ratio=1.5,αυτο σημαινει
            #για καθε cluster ελεγχουμε αν ισχυει η ανισοτητα d(q,c(m)) <= e * dmin 
            #αν ισχυει τοτε εισαγουμε το cluster στην αναζητηση, δηλαδη Μ = Μ+1

        clusters=[]
        clusterdistances.flatten()
        print("hello",clusterdistances[0][0])
        for clustid in clusterdistances:
            dist=clusterdistances[clustid]
            if dist > (dmin * e): break
        
            clusters.append(clustid)
        #cluster index
        print(len(clusters))
        basekeys=np.concatenate([index[c] for c in clusters])
        
        computations+=len(basekeys)
        
        vectorsforcheck=S[basekeys]
        knn.fit(vectorsforcheck)
        distances,vectorindices= knn.kneighbors(q[i].reshape(1,-1))
        #distances=np.linalg.norm(vectorsforcheck-q[i],axis=1)
        #argument sort dhladh apothikeyei toys deiktes toy S poy exoyn to k kontinotero dianysma me to q[i] DEN EPISTREFEI TO VECTOR gia logoys mnhmh ypothetw lol 
        
        #vectorindices=np.argsort(distances)[:k]
        approxresults.append(basekeys[vectorindices])
    
    return approxresults,computations
    
def PreciseNearestNeighbors(q,index:dict, centroids:np.ndarray,S:np.ndarray,k):
    computations=len(q) * len(S)
    
    knn = NearestNeighbors(n_neighbors=k,algorithm='brute',n_jobs=-1)
    knn.fit(S)
    distances,indices =knn.kneighbors(q)

    return indices,computations

def evaluations(q,index,centroids,S,groundtruth,k):
    print("Testing phase of algorithms:")
    annstart_time=time.time()
    ann,anncomp=ApproximateNearestNeighbors(q,index,centroids,S,k)
    ann_time=time.time()-annstart_time
    annqps= len(q) / ann_time

    #recall via groundtruth
    matches=0
    for i in range(len(q)):
        trueneighbors=groundtruth[i][:k]
        neighborsann=ann[i]

        common = np.intersect1d(trueneighbors,neighborsann)
        matches+=len(common)

    recall_percentage= (matches / (len(q) * k)) *100

    print("=" * 50)
    print("For approximate nearest neighbors:")
    print(f"Running Time            {ann_time:.3f} Seconds")
    print(f"Speed                   {annqps:.1f} queries/sec")
    print(f"N of calculations       {anncomp:,}")
    print(f"Recall                  {recall_percentage:.2f}%")
    print("="*50)
    
    pnnstart=time.time()
    pnn,pnncomp=PreciseNearestNeighbors(q,index, centroids,S,k)
    pnn_time=time.time()-pnnstart
    pnnqps= len(q) / pnn_time

    pnnmatches=0
    for i in range(len(q)):
        truepnn=groundtruth[i][:k]
        pnnNeighbors=pnn[i]

        commonpnn=np.intersect1d(truepnn,pnnNeighbors)
        pnnmatches+=len(commonpnn)

        pnnrecall=(pnnmatches / (len(q) * k)) *100

    print("=" * 50)
    print("for Precise Nearest Neighbors")
    print(f"Running Time              {pnn_time:.3f} Seconds")
    print(f"Speed                     {pnnqps:.1f} queries/sec")
    print(f"N of calculations         {pnncomp:,}")
    print(f"Recall                    {pnnrecall:.2f}%")
    print("="*50)


dimension=128
clusters=1000

S=sift_base=retrievefvecs("sift/sift_base.fvecs")
train=sift_learn=retrievefvecs("sift/sift_learn.fvecs")
Q=sift_query=retrievefvecs("sift/sift_query.fvecs")
ground_truth=sift_groundtruth=retriveivecs("sift/sift_groundtruth.ivecs")

with open("centroids.pkl", "rb") as f:
    centroids = pkl.load(f)
        
with open("invertedindex.pkl", "rb") as f:
    invertedindex = pkl.load(f)

print(type(invertedindex))
evaluations(Q[:200],invertedindex,centroids,S,ground_truth[:200],k=100)
#print("s",len(S),"Q",len(Q),"learn",len(sift_learn),"groundtruth",len(ground_truth))