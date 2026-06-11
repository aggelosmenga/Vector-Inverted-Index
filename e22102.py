import numpy as np
import time
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


def ApproximateNearestNeighbors(q:np.ndarray,invindex:dict,centroids:np.ndarray,S:np.ndarray,k,e=1.5):
    approxresults=[]
    computations=0 #μεταβλητες για την εισαγωγη των κ κοντινοτερων για καθε query
    for i in range(len(q)):
        #Για καθε q:
            #κανουμε ευκλειδια αποσταση με καθε συσταδα
            #κανουμε argsort για να παρουμε τα key values του πινακα clusterdistances
            #βρισκουμε τη κοντινοτερη συσταδα
            #υλοποιουμε 1+e Approximate Nearest Neighbors
        clusterdistances=np.linalg.norm(centroids-q[i],axis=1)
        
        sortedclusters=np.argsort(clusterdistances)

        dmin=clusterdistances[sortedclusters[0]]

        #λογικη πισω απο την αυτοματοποιηση:
            #1+ε Approximate nearest neighbor search για τα κοντινοτερα clusters
            #οριζουμε dmin το κοντινοτερο διανυσμα στο query μας
            #οριζουμε ενα tolerance ratio=1.5,αυτο σημαινει
            #για καθε cluster ελεγχουμε αν ισχυει η ανισοτητα d(q,c(m)) <= e * dmin 
            #αν ισχυει τοτε εισαγουμε το cluster στην αναζητηση, δηλαδη Μ = Μ+1

        clusters=[] #ξαναοριζουμε τη λιστα για καθε query εφοσον το καθενα εχει διαφορετικες κοντινοτερες συσταδες
        for clustid in sortedclusters:
            dist=clusterdistances[clustid]
            if dist > (dmin * e): break
        
            clusters.append(clustid)
        
        #Παιρνουμε ολα τα διανυσματα των συσταδων που βγηκαν μετα τον Ελεγχο απο το inverted index
        basekeys=np.concatenate([invindex[c] for c in clusters])
        
        #μεταβλητη για το συνολο υπολογισμων
        computations+=len(basekeys)
        
        #Εχουμε τα κλειδια των διανυσματων που πρεπει να γινει ελεγχος και τα ανακτουμε απο τον πινακα S
        vectorsforcheck=S[basekeys]
        
        #Κ Nearest Neighbors για καθε διανυσμα με το query 
        distances=np.linalg.norm(vectorsforcheck-q[i],axis=1)
        vectorindices=np.argsort(distances)[:k]
        approxresults.append(basekeys[vectorindices]) 
        #απο τον πινακα Basekeys παιρνουμε τα κ μεγαλυτερα και τα βαζουμε στη λιστα αποτελεσματων
    
    return approxresults,computations
    
def PreciseNearestNeighbors(q,invindex:dict, centroids:np.ndarray,S:np.ndarray,k):
    results=[]
    basekeys=np.concatenate(list(invindex.values())) 
    #"Για την υλοποίηση των αλγορίθμων θα χρησιμοποιηθεί μια προσέγγιση που στηρίζεται στη χρήση 
    # αντεστραμμένων ευρετηρίων (inverted indexes)". Συνεπως μπροουμε να παρουμε τα κλειδια με τη δομη που τα εχει το inverted index
    # και να κανουμε Nearest Neighbors με αυτη τη κατανομη

    #Για τον ακριβη αλγοριθμο η διαδικασια ειναι πιο απλη:
        #παιρνουμε καθε q και κανουμε υπολογισμους με ολο το συνολο δεδομενων
        #queries = 10,000 Dataset =1,000,000 συνολικα θα γινουν 10,000,000,000 υπολογισμοι 
    computations=len(q)*len(basekeys)

    for i in range(len(q)):
        distances=np.linalg.norm(S[basekeys]-q[i],axis=1)

        vectorindices=np.argsort(distances)[:k]

        results.append(basekeys[vectorindices]) 
        #απο τον πινακα Basekeys παιρνουμε τα κ μεγαλυτερα και τα βαζουμε στη λιστα αποτελεσματων

    return results,computations


def evaluations(q,index,centroids,S,groundtruth,k):
    print("Testing phase of algorithms:")
    annstart_time=time.time()
    ann,anncomp=ApproximateNearestNeighbors(q,index,centroids,S,k)
    ann_time=time.time()-annstart_time 
    #ο χρονος για τον ANN συμπεριλαμβανει και τον υπολογισμο των κοντινοτερων συσταδων
    annqps= len(q) / ann_time #ποσες πραξεις για queries γινονται το δευτερολεπτο

    #recall με τη χρηση του συνολου groundtruth που μας δινει το dataset
    matches=0
    for i in range(len(q)):
        trueneighbors=groundtruth[i][:k] #για καθε διανυσμα ποιοι ειναι οι αληθινοι Κ κοντινοτεροι γειτονες του
        neighborsann=ann[i]

        common = np.intersect1d(trueneighbors,neighborsann) #κανουμε Intersect τα σωστα αποτελεσματα
        #δηλαδη βαζουμε σε εναν πινακα ολα τα κοινα διανυσματα, αρα ολες τις σωστες απαντησεις
        matches+=len(common)
        #ολα τα σωστα αποτελεσματα / ολα τα πιθανα σωστα αποτελεσματα
        # δηλαδη εχουμε το αθροισμα ολων των σωστων (για το 1ο q αν βρει και τα 100 σωστα τοτε Matches= 0 + 100)
        # συνολικα θα αν τα βρουμε ολα σωστα τοτε το Matches θα εχει τιμη ιση με len(q) * k (100 για ολα τα 10,000 queries)
        # αρα [ (true positive) / (true positive + false negative) ]  
    annrecall= (matches / (len(q) * k)) *100
    
    #print τα αποτελεσματα του ANN
    print("=" * 50)
    print("For approximate nearest neighbors:")
    print(f"Running Time            {ann_time:.3f} Seconds")
    print(f"Speed                   {annqps:.1f} queries/sec")
    print(f"N of calculations       {anncomp:,}")
    print(f"Recall                  {annrecall:.2f}%")
    print("="*50)
    
    pnnstart=time.time()
    pnn,pnncomp=PreciseNearestNeighbors(q,index, centroids,S,k)
    pnn_time=time.time()-pnnstart
    pnnqps= len(q) / pnn_time
    #ιδια διαδικασια για Precise Nearest Neighbors
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
#sqrt(1,000,000) = 1000 clusters... προσπαθησα με elbow rule αλλα δεν δουλευει καλα με μεγαλο συνολο δεδομενων

#παιρνουμε τα δεδομενα 
S=sift_base=retrievefvecs("sift/sift_base.fvecs")
train=sift_learn=retrievefvecs("sift/sift_learn.fvecs")
Q=sift_query=retrievefvecs("sift/sift_query.fvecs")
ground_truth=sift_groundtruth=retriveivecs("sift/sift_groundtruth.ivecs")

#inverted index & cluster centroids from pickle files
with open("centroids.pkl", "rb") as f:
    centroids = pkl.load(f)
        
with open("invertedindex.pkl", "rb") as f:
    invertedindex = pkl.load(f)

#υλοποιηση για τα πρώτα 500 διανυσματα
evaluations(Q[:500],invertedindex,centroids,S,ground_truth[:500],k=100)
