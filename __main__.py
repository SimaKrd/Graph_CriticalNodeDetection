from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
import Graph
import BFSgraph
import random
import Labellisation
import Embed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
import networkx as nx
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import ADASYN
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


#upload the graphe ---> create a networkx graphTrain
fileTrain = "/Users/simakrd/Desktop/Graphs/Code/CL-1000-2d0-trial2/CL-1000-2d0-trial2.edges"
fileTest = "/Users/simakrd/Desktop/Graphs/Code/CL-1000-2d1-trial3/CL-1000-2d1-trial3.edges"
graphTrain = Graph.uploadGraph(fileTrain)
graphTest = Graph.uploadGraph(fileTest)


#visualize the graphTrain
'''graphTrain.visualizeGraph(graphTrain)'''


#extracting node features
graphTrain = Graph.EigenvectorCentrality(graphTrain)
graphTrain = Graph.ClosenessCentrality(graphTrain)
graphTrain = Graph.ClusteringCoefficient(graphTrain)
graphTrain = Graph.BetweennessCentrality(graphTrain)
graphTrain = Graph.degrePropertie(graphTrain)

graphTest = Graph.EigenvectorCentrality(graphTest)
graphTest = Graph.ClosenessCentrality(graphTest)
graphTest = Graph.ClusteringCoefficient(graphTest)
graphTest = Graph.BetweennessCentrality(graphTest)
graphTest = Graph.degrePropertie(graphTest)


# create a one average attribute for all the features
graphTrain = Labellisation.normalize_features(graphTrain)
Graph.displayNodeAttributes(graphTrain)

graphTest = Labellisation.normalize_features(graphTest)
Graph.displayNodeAttributes(graphTest)

# display the top nodes based on the feature average
'''print(graphTrain.top_nodes(graphTrain,'average', num_top=10))'''

#label the node based on the feature average
labelsTrain = Labellisation.labelNodeByAverage(graphTrain)
labelsTest = Labellisation.labelNodeByAverage(graphTest)


# print nodes that are critical
'''for node, label in labelsTrain.items():
    if label == 1:
        print(node)
'''

# embedding the graphe
Embed.node2Vec(graphTrain,'embeddingsTrain')
#Embed.apply_pca('ACP_ReductionTrain', 'embeddingsTrain')
#Embed.displayEmbedding('ACP_ReductionTrain', labelsTrain)

Embed.node2Vec(graphTest,'embeddingsTest')
#Embed.apply_pca('ACP_ReductionTest','embeddingsTest')
#Embed.displayEmbedding('ACP_ReductionTest', labelsTest)


#dataTrain = np.loadtxt('/Users/simakrd/Desktop/Graphs/Code/struc2vec/embeddingsTrain', usecols=range(1, 129), skiprows=1)
dataTrain = np.loadtxt('embeddingsTrain', usecols=range(1, 129), skiprows=1)
listTrain = list(labelsTrain.values())
print(dataTrain)
print(listTrain)

#dataTest = np.loadtxt('/Users/simakrd/Desktop/Graphs/Code/struc2vec/embeddingsTest', usecols=range(1, 129), skiprows=1)
dataTest = np.loadtxt('embeddingsTest', usecols=range(1, 129), skiprows=1)
listTest = list(labelsTest.values())
#SMOTE use for unbalanced data
smote = SMOTE(random_state=42)
X_dataTrain, y_listTrain = smote.fit_resample(dataTrain, listTrain)
X_dataTest, y_listTest = smote.fit_resample(dataTest, listTest)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', SVC(random_state=42))
]

clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

#generating the model ML 
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
model.fit(X_dataTrain, y_listTrain)

predictions = model.predict(X_dataTest)
print(classification_report(y_listTest, predictions))


'''precision_curve, recall_curve, _ = precision_recall_curve(y_listTest, predictions)
plt.plot(recall_curve, precision_curve)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()'''

#Embed.displayEmbedding('/Users/simakrd/Desktop/Graphs/Code/struc2vec/emb/trail3embd.emb',labelsTest)


#RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)



































'''
# Getting a list of possible starting nodes and then affect one of them to nodeStart
#extracting subgraphs     TrainSubGraph = 70%    TestSubGraph = 30%
''''''train_size = int(0.6*graphTrain.number_of_nodes())
nodeStart = random.choice(graphTrain.LdmNode(graphTrain,3))
TrainsubGraph = BFSgraph.BFSsubgraph(graphTrain,nodeStart,train_size)
TestsubGraph = nx.complement(TrainsubGraph)

graphTrain.visualizeGraph(TrainsubGraph)

complement_nodes = set(graphTrain.nodes()) - set(TrainsubGraph.nodes())
TestsubGraph = graphTrain.graphTrain(complement_nodes)
graphTrain.visualizeGraph(TestsubGraph)''''''












'''


'''# Getting a list of possible starting nodes and then affect one of them to nodeStart
nodeStart = random.choice(graphTrain.LdmNode(g,3))
print(nodeStart)
print(g.degree(nodeStart))

#extract the graphTrain and display it
graphTrain = BFSgraph.BFSsubgraph(g,nodeStart,100)'''

'''# features of the nodes
graphTrain = graphTrain.EigenvectorCentrality(graphTrain)
graphTrain = graphTrain.ClosenessCentrality(graphTrain)
graphTrain = graphTrain.ClusteringCoefficient(graphTrain)
graphTrain = graphTrain.BetweennessCentrality(graphTrain)
graphTrain = graphTrain.degrePropertie(graphTrain)'''


'''# display the top nodes based on the feature average
print(graphTrain.top_nodes(graphTrain,'average', num_top=10))

#label the node based on the feature average
labels = Labellisation.labelNodeByAverage(graphTrain)'''


'''# print nodes that are critical
for node, label in labels.items():
    if label == 1:
        print(node)'''








