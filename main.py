import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
from tqdm import tqdm
from src.agent import NormAgent
from src.utils import read_data, construct_mutag_prompt_graph, load_data, separate_data, delete_node, extract_delete_node, get_faces, extract_delete_edge, LOGGER, delete_edge, construct_protein_prompt_graph, construct_bzr_prompt_graph, construct_cox2_prompt_graph, construct_enzymes_prompt_graph, get_tda, power_tda
from src.models.graphcnn import GraphCNN
from scipy import stats


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

def main(args):
    if not args.latent:
        latent_string = "False"
        #! 1. Node Erase Type
        if args.erase_type == 0:
            if args.additional_flag:
                additional_string = "True " + args.addition_type
            else:
                additional_string = "False"
            LOGGER.debug(f"******************************************************** {args.dataset} Erase Node {args.erase_num}, Latent:{latent_string}, Additional: {additional_string} ********************************************************")

            #! Load orginal data and initialize agent
            data = read_data(args.dataset)
            llm_agnet =  NormAgent(1, f'NodeEraser_{args.erase_num}', 'Openai')
            os.makedirs(f'result/{args.dataset}/', exist_ok=True)
            if args.additional_flag:
                os.makedirs(f'store/node_erase/{args.dataset}/{args.addition_type}/', exist_ok=True)
            else:
                os.makedirs(f'store/node_erase/{args.dataset}/', exist_ok=True)
            new_data = []

            #! Ask LLM to delete nodes and store new graph into json file
            for index,  graph in enumerate(data):
                #* Construct graph prompt
                edge = graph['edges']
                node_lable = graph['node_labels']
                face_list = get_faces(graph)
                diff_list = power_tda(graph)

                if args.dataset == "MUTAG":
                    graph_prompt = construct_mutag_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "PROTEINS":
                    graph_prompt = construct_protein_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "BZR":
                    graph_prompt = construct_bzr_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "COX2":
                    graph_prompt = construct_cox2_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "ENZYMES":
                    graph_prompt = construct_enzymes_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)

                LOGGER.debug(f'Graph {index} LLM Prompt: {graph_prompt}')

                response =  llm_agnet.get_response(query = graph_prompt)
                LOGGER.debug(f'Graph {index} LLM Response: {response}')

                node_list = extract_delete_node(response)
                LOGGER.debug(f'Nodes need to be deleted: {node_list}')

                new_graph = delete_node(graph, node_list)
                new_data.append(new_graph)

            with open(f'result/{args.dataset}/result.json', 'w') as output:
                    json.dump(new_data, output)
            
            if args.additional_flag:
                with open(f'store/node_erase/{args.dataset}/{args.addition_type}/erase_{args.erase_num}.json', 'w') as output:
                    json.dump(new_data, output)
            else:
                with open(f'store/node_erase/{args.dataset}/erase_{args.erase_num}.json', 'w') as output:
                    json.dump(new_data, output)
        
        #! 2. Edge Erase Type
        if args.erase_type == 1:
            if args.additional_flag:
                additional_string = "True " + args.addition_type
            else:
                additional_string = "False"
            LOGGER.debug(f"******************************************************** {args.dataset} Erase Edge {args.erase_num}, Latent:{latent_string}, Additional: {additional_string} ********************************************************")

            #! Load orginal data and initialize agent
            data = read_data(args.dataset)
            llm_agnet =  NormAgent(1, f'EdgeEraser_{args.erase_num}', 'Openai')
            os.makedirs(f'result/{args.dataset}/', exist_ok=True)
            if args.additional_flag:
                os.makedirs(f'store/edge_erase/{args.dataset}/{args.addition_type}/', exist_ok=True)
            else:
                os.makedirs(f'store/edge_erase/{args.dataset}/', exist_ok=True)
            new_data = []

            #! Ask LLM to delete nodes and store new graph into json file
            for index,  graph in enumerate(data):
                #* Construct graph prompt
                edge = graph['edges']
                node_lable = graph['node_labels']
                face_list = get_faces(graph)
                diff_list = power_tda(graph)

                if args.dataset == "MUTAG":
                    graph_prompt = construct_mutag_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "PROTEINS":
                    graph_prompt = construct_protein_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "BZR":
                    graph_prompt = construct_bzr_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "COX2":
                    graph_prompt = construct_cox2_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                if args.dataset == "ENZYMES":
                    graph_prompt = construct_enzymes_prompt_graph(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type)
                LOGGER.debug(f'Graph {index} LLM Prompt: {graph_prompt}')

                response =  llm_agnet.get_response(query = graph_prompt)
                LOGGER.debug(f'Graph {index} LLM Response: {response}')

                edge_list = extract_delete_edge(response)
                LOGGER.debug(f'Edges need to be deleted: {edge_list}')

                new_graph = delete_edge(graph, edge_list)
                new_data.append(new_graph)

            with open(f'result/{args.dataset}/result.json', 'w') as output:
                    json.dump(new_data, output)

            if args.additional_flag:
                with open(f'store/edge_erase/{args.dataset}/{args.addition_type}/erase_{args.erase_num}.json', 'w') as output:
                    json.dump(new_data, output)
            else:
                with open(f'store/edge_erase/{args.dataset}/erase_{args.erase_num}.json', 'w') as output:
                    json.dump(new_data, output)

    #! execute the GCN for erased graph
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag, args.latent, args.erase_num, args.erase_type, args.additional_flag, args.addition_type)
    #set up seeds and gpu device

    if args.erase_type == 0: 
        type_tring = " Node "
    if args.erase_type == 1:
        type_tring = " Edge " 

    if args.additional_flag:
        additional_string = "True " + args.addition_type
    else:
        additional_string = "False"
    max_acc_list = []
    for key in range(10):
        max_acc = 0.0
        LOGGER.debug(f"******************************************************** {args.dataset} Erase {type_tring} {args.erase_num} Random Seed {key}, Additional: {additional_string}, lr: {args.lr}, epoch: {args.epochs}, batch: {args.batch_size} ********************************************************")
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
    LOGGER.debug(f"Erase {args.erase_num} final result: {acc}")

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