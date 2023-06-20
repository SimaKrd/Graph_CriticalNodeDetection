import collections
import networkx as nx
import matplotlib.pyplot as plt

def visualizeGraph(G):
    ''' 
    Display the graph for visualization 
    '''
    plt.figure(figsize=(7,7))
    pos = nx.spring_layout(G, seed=42, k=0.15)
    node_list = list(G.nodes())
    degree_list = [d for n, d in G.degree()]
    color_map = plt.cm.get_cmap('coolwarm', max(degree_list)-min(degree_list)+1)
    nx.draw_networkx_nodes(G, pos, node_color=degree_list, cmap=color_map, node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels={node:node for node in G.nodes()})
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=min(degree_list), vmax=max(degree_list)))
    sm.set_array([])
    plt.colorbar(sm, label='Node Degree')

    plt.title('Graph Visualization with Nodes Colored by Degree')
    plt.show()



def uploadGraph(filename):
    ''' 
    this function Create the nx graph 
    filename specify your edge file path  
    '''
    count = 0
    edges = set()
    duplicata = set()
    with open(filename, 'r') as f:
        for line in f.readlines():
            count+=1
            if not line.strip():
                continue
            u, v = line.strip().split(",")
            edge = frozenset([int(u), int(v)])
            if edge in edges :
                duplicata.add(edge)
            else :
                edges.add((int(u), int(v)))
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def mediumDegree(graph):
    '''
    Return the average degree of a graph ( float )
    '''
    degreeSequence = sorted([d for n, d in graph.degree()], reverse=True)
    degreeCount = collections.Counter(degreeSequence)
    deg, cnt= zip(*degreeCount.items())
    return sum(degreeSequence) / sum(cnt)


def LdmNode(graph, intervalle):
    '''
    Return a list of nodes within the degree range based on the medium degree +/- intervalle/2
    '''
    degreeRange = (mediumDegree(graph) - intervalle/2, mediumDegree(graph) + intervalle/2)
    Ldm = [node for node, degree in graph.degree() if degreeRange[0] <= degree <= degreeRange[1]]
    return Ldm


def isConnectedGraph(graph):
    return nx.is_connected(graph)

   
def removeIsolatedNodes(graph):
    return graph.remove_nodes_from(list(nx.isolates(graph)))


def connectGraph(graph): 

    # Get the connected components as subgraphs
    components = list(nx.connected_components(graph))

    # Connect the components by adding edges
    if len(components) > 1:
        for i in range(len(components) - 1):
            component1 = components[i]
            component2 = components[i + 1]
            graph.add_edge(next(iter(component1)), next(iter(component2)))


    return graph


'''
features : 
'''


def degrePropertie(graph):
    '''
    returns a dictionary where the keys are the nodes of the graph and the values are their degree
    '''
    dic = dict(graph.degree(graph.nodes()))
    nx.set_node_attributes(graph, dic, 'degree')
    return graph

def ClusteringCoefficient(graph):
    '''
    dictionnary where the keys are nodes and the values are their respective clustering coefficient
    '''
    dic = nx.clustering(graph)
    nx.set_node_attributes(graph, dic, 'clustering coefficient')
    return graph

def BetweennessCentrality(graph):
    '''
    dictionnary where the keys are nodes and the values are their respective Betweenness Centrality
    ''' 
    dic = nx.betweenness_centrality(graph)
    nx.set_node_attributes(graph,dic,'betweeness centrality')
    return graph

def ClosenessCentrality(graph):
    '''
    dictionary where the keys are the nodes of the graph
    and the values are their respective closeness centrality.
    '''
    dic = nx.closeness_centrality(graph)
    nx.set_node_attributes(graph, dic, 'closeness centrality')
    return graph

def EigenvectorCentrality(graph):
    '''
    dictionary where the keys are the nodes of the graph
    and the values are their respective eigenvector centrality.
    This measures the influence of a node in a network. 
    It considers the degree of a node and the degree of its neighbours. A node is considered important if it is linked to other important nodes.

    '''
    dic = nx.eigenvector_centrality(graph)
    nx.set_node_attributes(graph,dic,'eigenvector centrality')
    return graph
    


def addAttribute(dictFeatures, graph, FeatureName):
    
    for node, value in dictFeatures.items():
        graph.nodes[node][FeatureName] = value

    return graph

def displayNodeAttributes(graph):
    '''
    this function display all node features
    '''
    for node, attr in graph.nodes(data=True):
        print(f"node : {node} Attribuyes: {attr}")

def normalize_features(graph):
    """
    This function normalizes the node features to a range [0,1] and computes the average.
    """
    # List to store averages
    averages = {}

    # Loop over each node
    for node in graph.nodes:
        # Get features
        features = graph.nodes[node]

        # Normalize the features
        normalized_features = {feature: (value - min(features.values())) / (max(features.values()) - min(features.values())) 
                               for feature, value in features.items()}

        # Compute average of normalized features
        averages[node] = sum(normalized_features.values()) / len(normalized_features)

    # Set the average as an attribute of the node
    nx.set_node_attributes(graph, averages, 'average')

    return graph

def top_nodes(graph, attribute, num_top=10):
    """
    Returns the top 'num_top' nodes based on the specified 'attribute'.
    """
    node_values = nx.get_node_attributes(graph, attribute)
    sorted_node_values = sorted(node_values.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_node_values[:num_top]