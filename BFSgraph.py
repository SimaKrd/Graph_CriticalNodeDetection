import collections
import networkx as nx

def BFSsubgraph(G, start_node, limit):
    """
    G is the graph you are working on
    start_node is the node where you want to start the search
    limit is the maximum number of nodes you want in the subgraph
    """
    visited = set() 
    queue = collections.deque([start_node])
    while queue and len(visited)<limit:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(set(G.neighbors(node)) - visited)
            # condition to stop the search if the number of visited nodes equals limit
            if len(visited) == limit:
                break
    return G.subgraph(visited)


def DFSsubGraph(G, start_node, limit):
    visited = set() # Set of visited nodes
    stack = [start_node] # Stack for DFS

    while stack and len(visited) < limit:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(n for n in G.neighbors(node) if n not in visited)

    return G.subgraph(visited)


def get_complement_graph(graph, BFS_nodes):
    BFS_graph = graph.subgraph(BFS_nodes)
    complement_graph = nx.complement(BFS_graph)
    return BFS_graph, complement_graph



