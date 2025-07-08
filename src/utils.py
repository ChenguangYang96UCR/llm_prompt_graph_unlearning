import logging
import json
import os
import networkx as nx
import numpy as np
import torch
import re
from sklearn.model_selection import StratifiedKFold

def set_logger(log_file = 'multi_agent.log', log_level=logging.DEBUG):

    """
    Set python logger

    Args:
        log_file (str, optional): logger file path. Defaults to 'multi_agent.log'.
        log_level (_type_, optional): logger level. Defaults to logging.DEBUG.

    Returns:
        class: logger class
    """    

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    # record log in log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # record log in console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

# Global parameter LOGGER
LOGGER = set_logger(log_level = logging.DEBUG)


def mutag_preprocess():

    """
    MUTAG dataset preprocess, write data into json file
    """    

    os.makedirs('output/MUTAG/', exist_ok=True)
    #* Get node number for each graph
    nodes_numbers = []
    with open('dataset/MUTAG/MUTAG/raw/MUTAG_graph_indicator.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        start_indicator = 1
        nodes_number = 0
        for line in lines: 
            if int(line.strip()) == start_indicator:
                nodes_number += 1
            else:
                nodes_numbers.append(nodes_number)
                nodes_number = 1
                start_indicator = int(line.strip())

     #! add last node number into nodes list
    nodes_numbers.append(nodes_number)

    #* Get edge info for each graph
    graphs = []
    with open('dataset/MUTAG/MUTAG/raw/MUTAG_A.txt', 'r', encoding='utf-8') as file:
        with open('dataset/MUTAG/MUTAG/raw/MUTAG_edge_labels.txt', 'r', encoding='utf-8') as edge_file:
            lines = file.readlines()
            edge_lines = edge_file.readlines()
            start_id = 1
            start_index = 1
            graph_index = 0
            graph = []
            for index, line in enumerate(lines):
                node_ids = line.strip().split(',')
                edge_label = edge_lines[index].strip()
                edge = []
                cross_graph = False
                for node_id in node_ids:
                    edge.append(int(node_id.strip()) - start_index)
                    if int(node_id.strip()) >= start_id + nodes_numbers[graph_index]:
                        cross_graph = True
                edge.append(int(edge_label))

                if cross_graph:
                    start_id += nodes_numbers[graph_index]
                    start_index += nodes_numbers[graph_index]

                    # renew the edge information
                    edge.clear()
                    for node_id in node_ids:
                        edge.append(int(node_id.strip()) - start_index)
                    edge.append(int(edge_label))

                    graph_index += 1
                    graphs.append(graph.copy())
                    graph.clear()
                    graph.append(edge)
                else:
                    graph.append(edge)
        
        #! add last graph into graph list
        graphs.append(graph)

    #* Get node label for each node
    graph_node_number = 0
    graph_node_labels = []
    graph_index = 0
    node_id = 1
    start_index = 1
    with open('dataset/MUTAG/MUTAG/raw/MUTAG_node_labels.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        node_labels = {}
        for line in lines:
            node_label = line.strip()
            if graph_node_number >= nodes_numbers[graph_index]:
               start_index += nodes_numbers[graph_index]
               graph_index += 1 
               graph_node_labels.append(node_labels.copy())
               node_labels.clear()
               node_labels[node_id - start_index] = int(node_label)
               graph_node_number = 1
               node_id += 1
            else:
                node_labels[node_id - start_index] = int(node_label)
                graph_node_number += 1
                node_id += 1

        graph_node_labels.append(node_labels)

            
    graph_labels = []
    with open('dataset/MUTAG/MUTAG/raw/MUTAG_graph_labels.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            graph_label = line.strip()
            if int(graph_label) == -1:
                graph_label = '0'
            graph_labels.append(int(graph_label))
            
    assert len(graphs) == len(nodes_numbers) and len(graphs) == len(graph_labels), "Graph length are different!"
    data_list = []
    for index, node_number in enumerate(nodes_numbers):
        data = {"edges": graphs[index], "node_number": node_number, "node_labels": graph_node_labels[index], "graph_label": graph_labels[index]}
        data_list.append(data)
    
    with open('output/MUTAG/output.json', 'w') as output:
        json.dump(data_list, output)


def read_data(data_set: str = 'MUTAG'):

    """
    Read data from dataset

    Args:
        data_set (str, optional): data set name. Defaults to 'MUTAG'.

    Returns:
        list: json data list
    """    

    data_list = ['MUTAG']
    assert data_set in data_list , 'Your data set does not exist!'
    data_set_root = './output/'
    with open(data_set_root + data_set +'/output.json', 'r') as input:
        data = json.load(input)
    return data

def construct_prompt_graph(edge : list, nodes_label : dict, face_list : list, additional_flag = False):

    prompt = f""" edge format is [node_id, node_id, edge_label], and edge list is: {edge}
    node label format is {{ndoe id, node label}} , and node label dict is : {nodes_label} .

Node labels:

  0  C
  1  N
  2  O
  3  F
  4  I
  5  Cl
  6  Br

Edge labels:

  0  aromatic
  1  single
  2  double
  3  triple

    """

    if additional_flag:
        additional_prompt = f""" The graph simplicial complex list is : {face_list} .
        """
        prompt = additional_prompt + prompt

    return prompt

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = {}
        self.node_features = 0
        self.edge_mat = torch.LongTensor(0)
        self.max_neighbor = 0

def load_data(data_set:str, degree_as_tag:bool, latent, erease_num):

    """
    load data to operate GCN 

    Args:
        data_set (str): data set name
        degree_as_tag (bool): determine use degree as tag or not

    Returns:
        list, type number of graph class: [graph_list, type numbers]
    """    

    g_list = []
    label_dict = {}
    feat_dict = {}
    data_list = ['MUTAG']
    assert data_set in data_list , 'Your data set does not exist!'

    # TODO: will change this dictionary as result
    if latent:
        LOGGER.debug(f'GCN Load latent {data_set} data: erase_{erease_num}.json')
        with open(f'store/{data_set}/erase_{erease_num}.json', 'r') as file:
            data = json.load(file)
    else:
        LOGGER.debug(f'GCN Load latest {data_set} data: result.json')
        with open(f'result/{data_set}/result.json', 'r') as file:
            data = json.load(file)

    all_nodes = []
    for graph in data:
        g = nx.Graph()
        node_tags = []
        #* Add nodes into networkx graph
        nodes = graph['node_labels']
        for node_id, label in nodes.items():
            g.add_node(int(node_id))
            node_tags.append(label)
            all_nodes.append(node_id)

        edges = graph['edges']
        for edge in edges:
            g.add_edge(edge[0], edge[1])

        g_list.append(S2VGraph(g, graph['graph_label'], node_tags))
    
    for index, g in enumerate(g_list):
        for i, j in g.g.edges():
            if i not in g.neighbors:
                g.neighbors[i] = [j]
            else:
                g.neighbors[i].append(j)
            if j not in g.neighbors:
                g.neighbors[j] = [i]
            else:
                g.neighbors[j].append(i)
        degree_list = []
        for node_id, neighbors in g.neighbors.items():
            # g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(neighbors))
        if not len(degree_list) == 0:
            g.max_neighbor = max(degree_list)

        # g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        if not len(degree_list) == 0: 
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, 2
        
def separate_data(graph_list, seed, fold_idx):

    """
    separate dataset to training set and test set

    Args:
        graph_list (list): graph data list
        seed (int): random seed
        fold_idx (int): index range limitation

    Returns:
        [list, list]: [training graph list, test graph list]
    """    

    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def extract_delete_node(response : str):

    """
    extract deleted node id from string

    Args:
        response (str): llm response

    Returns:
        list: list of node which need to be deleted
    """    

    answer_line = response.split('\n')[-1]
    numbers = re.findall(r'\d+\.?\d*', answer_line)
    int_numbers = [int(float(num)) for num in numbers]
    return int_numbers


def delete_and_reorder_with_mapping(node_dict, to_delete):

    """
    delete node which need to be deleted and reorder remain nodes

    Args:
        node_dict (dict): node dictionary ({node_id : node label})
        to_delete (list): nodes list which need to be deleted 

    Returns:
        [list, list]: [remain node list, new_id to old_id map]
    """    

    to_delete_set = set(to_delete)

    # Step 1: Filter remaining items
    remaining_items = [
        (int(k), v) for k, v in node_dict.items() if int(k) not in to_delete_set
    ]

    # Step 2: Sort by original key
    remaining_items.sort(key=lambda x: x[0])

    # Step 3: Create new dict and index mapping
    new_dict = {}
    index_map = {}

    for new_idx, (old_idx, val) in enumerate(remaining_items):
        new_dict[str(new_idx)] = val
        index_map[old_idx] = new_idx

    return new_dict, index_map


def delete_node(graph, deleted_node:list):

    """
    delete nodes which need to be deleted 

    Args:
        graph (dict): graph dictionary
        deleted_node (list): nodes which need to be deleted 

    Returns:
        dict: new remain graph dictionary
    """    
    
    new_graph = {}
    #! Delete graph first
    edges = graph['edges']
    temp_edges = [edge for edge in edges if edge[0] not in deleted_node and edge[1] not in deleted_node]

    #! Delete node
    node_dict = graph['node_labels']
    new_dict, index_map = delete_and_reorder_with_mapping(node_dict, deleted_node)

    #! Rewrite graph edge
    new_edges = []
    for edge in temp_edges:
        new_edges.append([index_map[edge[0]], index_map[edge[1]], edge[2]])
    
    new_graph['edges'] = new_edges
    new_graph['node_number'] = len(new_dict) 
    new_graph['node_labels'] = new_dict
    new_graph['graph_label'] = graph['graph_label']

    return new_graph

def get_faces(graph):

    """
    Returns a list of the faces in an undirected graph
    """

    #! Construct networkx graph
    G = nx.Graph()
    node_tags = []
    #* Add nodes into networkx graph
    nodes = graph['node_labels']
    for node_id, label in nodes.items():
        G.add_node(int(node_id))
        node_tags.append(label)

    edges = graph['edges']
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    #! Extract the list of the faces in the graph
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            if e1[0] == e2[0]:
                shared = e1[0]
                e3 = (e1[1], e2[1])
            elif e1[1] == e2[0]:
                shared = e1[1]
                e3 = (e1[0], e2[1])
            elif e1[0] == e2[1]:
                shared = e1[0]
                e3 = (e1[1], e2[0])
            elif e1[1] == e2[1]:
                shared = e1[1]
                e3 = (e1[0], e2[0])
            else: # edges don't connect
                continue

            if e3[0] in G[e3[1]]: # if 3rd edge is in graph
                faces.append(tuple(sorted((shared, *e3))))

    return list(sorted(set(faces)))



    






        


