import logging
import json
import os
import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

def set_logger(log_file = 'multi_agent.log', log_level=logging.DEBUG):
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

LOGGER = set_logger(log_level = logging.DEBUG)


def mutag_preprocess():
    os.makedirs('./output/MUTAG/', exist_ok=True)
    #* Get node number for each graph
    nodes_numbers = []
    with open('./dataset/MUTAG/MUTAG/raw/MUTAG_graph_indicator.txt', 'r', encoding='utf-8') as file:
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
    with open('./dataset/MUTAG/MUTAG/raw/MUTAG_A.txt', 'r', encoding='utf-8') as file:
        with open('./dataset/MUTAG/MUTAG/raw/MUTAG_edge_labels.txt', 'r', encoding='utf-8') as edge_file:
            lines = file.readlines()
            edge_lines = edge_file.readlines()
            start_id = 1
            graph_index = 0
            graph = []
            for index, line in enumerate(lines):
                node_ids = line.strip().split(',')
                edge_label = edge_lines[index].strip()
                edge = []
                cross_graph = False
                for node_id in node_ids:
                    edge.append(int(node_id.strip()))
                    if int(node_id.strip()) >= start_id + nodes_numbers[graph_index]:
                        cross_graph = True
                edge.append(int(edge_label))
                if cross_graph:
                    start_id += nodes_numbers[graph_index]
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
    with open('./dataset/MUTAG/MUTAG/raw/MUTAG_node_labels.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        node_labels = {}
        for line in lines:
            node_label = line.strip()
            if graph_node_number >= nodes_numbers[graph_index]:
               graph_index += 1 
               graph_node_labels.append(node_labels.copy())
               node_labels.clear()
               node_labels[node_id] = int(node_label)
               graph_node_number = 1
               node_id += 1
            else:
                node_labels[node_id] = int(node_label)
                graph_node_number += 1
                node_id += 1

        graph_node_labels.append(node_labels)

            
    graph_labels = []
    with open('./dataset/MUTAG/MUTAG/raw/MUTAG_graph_labels.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            graph_label = line.strip()
            graph_labels.append(int(graph_label))
            
    assert len(graphs) == len(nodes_numbers) and len(graphs) == len(graph_labels), "Graph length are different!"
    data_list = []
    for index, node_number in enumerate(nodes_numbers):
        data = {"edges": graphs[index], "node_number": node_number, "node_labels": graph_node_labels[index], "graph_label": graph_labels[index]}
        data_list.append(data)
    
    with open('./output/MUTAG/output.json', 'w') as output:
        json.dump(data_list, output)


def read_data(data_set: str = 'MUTAG'):
    data_list = ['MUTAG']
    assert data_set in data_list , 'Your data set does not exist!'
    data_set_root = './output/'
    with open(data_set_root + data_set +'/output.json', 'r') as input:
        data = json.load(input)
    return data

def construct_prompt_graph(edge : list, nodes_label : dict):
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
        self.edge_mat = 0

        self.max_neighbor = 0

def load_data(data_set:str, degree_as_tag:bool):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    data_list = ['MUTAG']
    assert data_set in data_list , 'Your data set does not exist!'
    # TODO: will change this dictionary as result
    with open('./output/' + data_set + '/output.json', 'r') as file:
        data = json.load(file)
    
    for graph in data:
        g = nx.Graph()
        node_tags = []
        #* Add nodes into networkx graph
        nodes = graph['node_labels']
        for node_id, label in nodes.items():
            g.add_node(node_id)
            node_tags.append(label)

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
        g.max_neighbor = max(degree_list)

        # g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
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

    return g_list, len(g_list)
        
def separate_data(graph_list, seed, fold_idx):
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


        


