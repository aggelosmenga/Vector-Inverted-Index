import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import pickle as pkl

#Script για τη δημιουργια inverted index

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
    #κανουμε train τον kmeans με τη χρηση του training set Που περιεχει το συνολο δεδομενων
    centroids,_=kmeans2(train,nofclusters,minit='points')
    #το inverted index ειναι δομης dictionary οπου για καθε κλειδι υπαρχει μια λιστα που περιεχει ολα τα 
    #κλειδια απο το συνολο S...δηλαδη δεν αποθηκευει τις τιμες αλλα τα Indexes του S
    invertedindex={i: [] for i in range(nofclusters)}

    chunks=train.shape[0]

    #χωρισμος των δεδομενων σε chunks για πιο ευκολη επεξεργασια
    for i in range(0,S.shape[0],chunks):
        #τα πρωτα i μετα τα επομενα i + chunks ...
        chunk=S[i :i+chunks]
        #εχοντας αποθηκευσει τα κεντρα βρισκουμε τα κοντινοτερα κεντρα για καθε chunks δεδομενων 
        distances=cdist(chunk,centroids,metric='euclidean')
       
       #closest centroids
        nearest=np.argmin(distances,axis=1)
        #μετραμε τα κοντινοτερα διανυσματα και τα τοποθετουμε ολα σε ενα cluster για καθε cluster θα εχουμε ~1000 διανυσματα/συσταδα 
        for j, centroid_id in enumerate(nearest):
            invertedindex[centroid_id].append(i + j)

    #μετατρεπουμε καθε λιστα σε Np array 
    for i in range(nofclusters):
        invertedindex[i]=np.array(invertedindex[i],dtype=np.int32)

    return centroids,invertedindex



dimension=128
clusters=1000

S=sift_base=retrievefvecs("sift/sift_base.fvecs")
train=sift_learn=retrievefvecs("sift/sift_learn.fvecs")
Q=sift_query=retrievefvecs("sift/sift_query.fvecs")
ground_truth=sift_groundtruth=retriveivecs("sift/sift_groundtruth.ivecs")

print("Inverted Index Creation and file save")
centroids,index= invertedindex(S,train,clusters)
#χρηση της Pickle για δημιουργια αρχειων Pkl που θα χρησιμοποιηθουν στο προγραμμα     
with open("centroids.pkl", "wb") as f:
    pkl.dump(centroids, f)
        
with open("invertedindex.pkl", "wb") as f:
    pkl.dump(index, f)
