from Graph import uploadGraph, visualizeGraph
import networkx as nx


def labelNode(graph):
    graph, count, duplicata= uploadGraph("CL-1000-2d0-trial2/CL-1000-2d0-trial2.edges")
    visualizeGraph(graph)

import numpy as np

def labelNodeByAverage(graph):
    # Get the 'average' attributes of all nodes
    averages = [data['average'] for node, data in graph.nodes(data=True)]
    
    # Compute the 95th percentile
    threshold = np.percentile(averages, 97)

    labels = {}

    # Assign labels
    for node in graph.nodes:
        if graph.nodes[node]['average'] > threshold:
            labels[node] = 1  # critical
        else:
            labels[node] = 0  # not critical

    return labels

def labelNodeByDegree(graph, threshold):
    labels={}

    for node in graph.nodes:
        if graph.degree[node] > threshold:
            labels[node] = 1
        else:
            labels[node]=0
    return labels
import numpy as np

def normalize_features(G):
    # Get the feature names
    features = list(G.nodes(data=True))[0][1].keys()
    
    # Normalizing each feature
    for feature in features:
        values = nx.get_node_attributes(G, feature)
        min_value = min(values.values())
        max_value = max(values.values())
        
        normalized_values = {node: (value-min_value) / (max_value-min_value) 
                             for node, value in values.items()}
        nx.set_node_attributes(G, normalized_values, feature)
    
    # Calculating and adding 'average' attribute
    average_values = {}
    for node in G.nodes:
        node_attributes = list(G.nodes[node].values())
        average_values[node] = np.mean(node_attributes)
    nx.set_node_attributes(G, average_values, 'average')
    
    return G
