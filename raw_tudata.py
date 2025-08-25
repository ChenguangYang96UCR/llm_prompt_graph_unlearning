import argparse
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
from tqdm import tqdm
from src.agent import NormAgent
from src.utils import separate_data, LOGGER
from src.models.graphcnn import GraphCNN
from scipy import stats
import random


criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        print(f"ouput size: {output.shape}, labels size: {labels.shape}")
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    LOGGER.debug("loss training: %f" % (average_loss))    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    LOGGER.debug("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

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


def load_data(data_set:str, degree_as_tag:bool):

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
    data_list = ['MUTAG', 'PROTEINS', 'BZR', 'COX2', 'ENZYMES']
    assert data_set in data_list , 'Your data set does not exist!'

    LOGGER.debug(f'GCN Load latest {data_set} data: output.json')
    with open(f'output/{data_set}/output.json', 'r') as file:
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
        edge_to_delete = random.choice(edges)
        remaining_edges = [e for e in edges if e != edge_to_delete]

        for edge in remaining_edges:
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

    if data_set == "MUTAG" or data_set == "PROTEINS" or data_set == "BZR" or data_set == "COX2":
        graph_type = 2

    if data_set == "ENZYMES":
        graph_type = 6

    print('# classes: %d' % graph_type)
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, graph_type

def main(args):
    #! execute the GCN for erased graph
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    #set up seeds and gpu device

    max_acc_list = []
    for key in range(10):
        max_acc = 0.0
        LOGGER.debug(f"******************************************************** {args.dataset} Random Seed {key}********************************************************")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)   
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
        train_graphs, test_graphs = separate_data(graphs, args.seed, key)

        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in range(1, args.epochs + 1):
            scheduler.step()

            avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
            acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

            if not args.filename == "":
                with open(args.filename, 'w') as f:
                    f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                    f.write("\n")

            LOGGER.debug("")
            LOGGER.debug(model.eps)
            max_acc = max(max_acc, acc_test)

        LOGGER.debug(f"Erase {args.erase_num} Fold Index {key} max test accuracy is {max_acc * 100}")
        max_acc_list.append(max_acc * 100)
    acc = ""
    for max_accs in max_acc_list:
        acc += f"& {max_accs} "

    mean = np.mean(max_acc_list)
    se = stats.sem(max_acc_list)
    acc += f"& {mean}$\pm${se}"
    LOGGER.debug(f"Erase {args.erase_num} Fold Index {key} final result: {acc}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG), supported dataset: MUTAG, PROTEINS, COX2, ENZYMES, BZR')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--erase_num', type = int, default = 1,
                                        help='erease number of node or edge')
    parser.add_argument('--latent', action="store_true",
    					help='use latent graph')
    parser.add_argument('--additional_flag', action="store_true",
    					help='use additional prompt information')
    parser.add_argument('--erase_type', type=int, default=0,
                        help='erase type in graph, 0: nodes, 1: edges, 2:feature (default: 0)')
    parser.add_argument('--addition_type', type=str, default="sc",
                        help='name of addition type (default: sc), supported addition type: sc, tda, combine')
    args = parser.parse_args()
    main(args)