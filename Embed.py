from Graph import uploadGraph
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

'''Thie file contains different embedding techniques'''


def node2Vec(graph, output_file):  

    ''' generate a node2vec embedding'''

    
    node2vec = Node2Vec(graph, dimensions=128, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format(output_file)
    model.save('modelnode2vec')
    return model


def deepWalk(graph): 

    ''' generate a deep walk embedding ''' 
    
    
    walks=[]
    for node in graph.nodes():
        for _ in range(10):
            walk = [node]
            while len(walk) < 30:
                current = walk[-1]
                neighbors = list(graph.neighbors(current))
                next_node = np.random.choice(neighbors)
                walk.append(next_node)
            walks.append(walk)

    model = Word2Vec(walks, vector_size=128, window=5, min_count=0, sg=1, workers=4)
    model.wv.save_word2vec_format('embeddings')
    model.save('model')

    return model
    




import matplotlib.pyplot as plt



def displayEmbedding(file, labels):
    # Load embeddings from file
    embeddings = {}
    with open(file, 'r') as f:
        next(f)  # skip the header line
        for line in f:
            values = line.split()
            node = int(values[0])
            coords = [float(x) for x in values[1:]]
            embeddings[node] = coords

    # Create a scatter plot
    plt.figure(figsize=(10, 10))
    for node, coords in embeddings.items():
        color = 'red' if labels[node] == 1 else 'green'
        plt.scatter(coords[0], coords[1], marker='o', color=color)  # plot the point
        plt.text(coords[0], coords[1], node, fontsize=9)  # label the point with the node ID

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Node Embeddings')
    plt.show()



def apply_pca(output_file ,input_file, n_components=2):
    # Load data
    df = pd.read_csv(input_file, header=None, sep=" ", skiprows=1)

    # Separate node labels and features
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)

    # Combine labels and PCA results
    result_df = pd.concat([labels, pd.DataFrame(pca_result)], axis=1)

    # Write to output file
    f = open(output_file, "w")
    result_df.to_csv(f, sep=" ", header=False, index=False)


def convert_arrayEmbd(Input_file):
    with open(Input_file, 'r') as f :
        lines = f.readline()
    embed_dict = {}
    for line in lines:
        splitlines = line.strip().split()
        nodeId = splitlines[0]
        embed = [float(val) for val in splitlines[1:]]
        embed_dict[nodeId] = embed
    nodes = list(embed_dict.keys())
    embeddings = np.array([embed_dict[node_id] for node_id in nodes])
    return embeddings
    
def convert_listlbls(dict):
    return list(dict.values())


