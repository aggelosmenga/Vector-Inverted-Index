import numpy as np
from scipy.cluster.vq import kmeans
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

def elbowrule(base,max=200):
    
    step=20
    samplesize=min(20000,base.shape[0])
    sample=base[np.random.permutation(base.shape[0])[:samplesize]]

    kvals=list(range(20,max+1,step))
    distortions=[]

    for k in kvals:
        centroids,distortion=kmeans(sample,k)

        distortions.append(distortion)
        print(f"  -> Tested k={k} | Distortion: {distortion:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(kvals, distortions, marker='o', linestyle='-', linewidth=2)
    
    # Add labels and formatting
    plt.title('Elbow Method for Optimal Clusters (P)', fontsize=14)
    plt.xlabel('Number of Clusters (P)', fontsize=12)
    plt.ylabel('Distortion (Avg Distance to Centroid)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()



S=sift_base=retrievefvecs("sift/sift_base.fvecs")
train=sift_learn=retrievefvecs("sift/sift_learn.fvecs")
Q=sift_query=retrievefvecs("sift/sift_query.fvecs")
ground_truth=sift_groundtruth=retriveivecs("sift/sift_groundtruth.ivecs")


elbowrule(S)