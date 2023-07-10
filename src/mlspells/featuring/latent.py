import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlspells.base import StringEnum

from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram

class ConnectivtyStrategy(StringEnum):
    NONE = 'none'
    KNN = 'knn'
    GRID = 'grid'

class AffinityStrategy(StringEnum):
    EUCLIDEAN = 'euclidean'
    L1 = 'l1'
    L2 = 'l2'
    MANHATTAN = 'manhattan'
    COSINE = 'cosine'

class LinkageStrategy(StringEnum):
    WARD = 'ward'
    COMPLETE = 'complete'
    AVERAGE = 'average'
    SINGLE = 'single'
    

def plot_dendrogram(model, file_name:str, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(10,10))
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig(file_name, format='png', bbox_inches='tight')

def feature_agglomeration(data: pd.DataFrame, number_of_features: int, normalize: bool = True,
               connectivity_type: ConnectivtyStrategy = ConnectivtyStrategy.NONE,
               affinity:AffinityStrategy = AffinityStrategy.EUCLIDEAN,
               linkage: LinkageStrategy = LinkageStrategy.WARD, pipeline: Pipeline = None) -> Pipeline:

    tranformations = []
    connectivity = None

    if number_of_features <= 0:
        raise ValueError('The number of components cannot be less or equal to zero.')

    if data.shape[-1] <= number_of_features:
        raise ValueError(f'The number of components ({number_of_features}) should be smaller than the number of features ({data.shape[-1]})')

    if linkage == LinkageStrategy.WARD and affinity != AffinityStrategy.EUCLIDEAN:
        raise ValueError(f"Affinity '{affinity}' cannot be used with linkage '{LinkageStrategy.WARD}'. Only '{AffinityStrategy.EUCLIDEAN} can be used")
 
    if normalize:
        scaler = StandardScaler().fit(data)
        transform_data = scaler.transform(data)
        tranformations.append(('normalize', scaler))
    else:
        transform_data = data.copy()

    if connectivity_type == ConnectivtyStrategy.GRID:
        connectivity = grid_to_graph(data[0].shape[0], 1)
    elif connectivity_type == ConnectivtyStrategy.KNN:
        connectivity = kneighbors_graph(data.T, n_neighbors=math.floor(number_of_features/2), mode='connectivity', metric='minkowski', include_self=False)

    agglo = FeatureAgglomeration(connectivity=connectivity, n_clusters=number_of_features, affinity=str(affinity), linkage=str(linkage), compute_distances=True).fit(transform_data)
    transformed_data = agglo.transform(transform_data)
    tranformations.append(('agglomeration', agglo))

    components_name = [f"col{index}" for index in range(0, number_of_features)]
    df = pd.DataFrame(data = transformed_data, columns = components_name)

    plot_dendrogram(agglo, 'outputs/dendrogram.png')
    
    if pipeline:
        pipeline.steps.append(tranformations)
    else:
        pipeline = Pipeline(tranformations)

    return pipeline
