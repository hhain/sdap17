## new features
## including: dimensionality reduction and new features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def reduce_dim_LDA(data, n_comps=25, label="target"):
    """
    Reduce the Dataframe to n_comps components 
    keeping the label column for LDA
    """
    
    if label is not None:
        y = data[label]
    
    X = data.drop(label, axis=1)
    
    lda = LinearDiscriminantAnalysis(n_components=n_comps)
    
    X_reduced = pd.DataFrame(lda.fit(X, y).transform(X))
    X_reduced.columns = ["lda_"+str(i) for i in range(len(X_reduced.columns))]
    
    if label is not None:
        return  pd.concat( [X_reduced, y], axis=1 )
    
    return X_reduced
   
from sklearn.decomposition import PCA

def reduce_dim_PCA(data, n_comps=10, label=None):
    """
    Reduce the Dataframe to n_comps components 
    leaves label intact
    """
    
    if label is not None:
        y = data[label]
        X = data.drop(label, axis=1)
    else:
        X = data
    
    pca = PCA(n_components=n_comps)
    X_reduced = pd.DataFrame(pca.fit(X).transform(X))
    X_reduced.columns = ["pca_"+str(i) for i in range(len(X_reduced.columns))]
    
    if label is not None:
        return  pd.concat( [X_reduced, y], axis=1 )
    
    return X_reduced

        
    
