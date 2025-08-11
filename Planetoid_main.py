import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import random
import os
from tqdm import tqdm
from src.agent import PlanetoidAgent
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from src.utils import read_data, delete_node, extract_delete_node, get_faces, extract_delete_edge, LOGGER, delete_edge, get_tda, power_tda, load_planetoid_dataset
from src.data_utils import class_rand_splits, load_fixed_splits, eval_rocauc, eval_acc
from src.models.graphcnn import GraphCNN
from scipy import stats
from src.models.mpnns import MPNNs
from src.eval import *
from src.logger import Logger


criterion = nn.CrossEntropyLoss()

def parse_method(args, n, c, d, device):
    print(f"mpnns parameter : {d}, {args.hidden_channels}, {c}, {args.local_layers}, {args.dropout}")
    model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout, 
    heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear, res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn = args.gnn).to(device)
    
    return model
        

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
    assert args.dataset == 'Cora' or args.dataset == 'CiteSeer', "Only accept planetoid dataset for this python file!"

    if not args.latent:
        latent_string = "False"
        #! 1. Edge Erase Type
        if args.erase_type == 1:
            if args.additional_flag:
                additional_string = "True " + args.addition_type
            else:
                additional_string = "False"
            LOGGER.debug(f"******************************************************** {args.dataset} Erase Edge {args.erase_num}, Latent:{latent_string}, Additional: {additional_string}, Cot: {args.cot} ********************************************************")

            #! Load orginal data and initialize agent
            data = read_data(args.dataset)
            llm_agnet =  PlanetoidAgent(1, f'EdgeEraser_{args.erase_num}', 'Openai')
            os.makedirs(f'result/{args.dataset}/', exist_ok=True)
            if args.additional_flag:
                os.makedirs(f'store/edge_erase/{args.dataset}/{args.addition_type}/', exist_ok=True)
            else:
                os.makedirs(f'store/edge_erase/{args.dataset}/', exist_ok=True)
            new_data = []

            #! Ask LLM to delete nodes and store new graph into json file
            #* Construct graph prompt
            edge = data['edges']
            node_lable = data['node_labels']
            face_list = []
            diff_list = []
            if args.additional_flag:
                face_list = get_faces(data)
                diff_list = get_tda(data)

            edge_list = llm_agnet.get_response(edge, node_lable, face_list, diff_list, args.additional_flag, args.addition_type, args.cot)
            LOGGER.debug(f'Edges need to be deleted: {edge_list}')

            new_graph = delete_edge(data, edge_list)
            new_data.append(new_graph)

            with open(f'result/{args.dataset}/result.json', 'w') as output:
                    json.dump(new_data, output)

            if args.additional_flag:
                with open(f'store/edge_erase/{args.dataset}/{args.addition_type}/erase_{args.erase_num}.json', 'w') as output:
                    json.dump(new_data, output)
            else:
                with open(f'store/edge_erase/{args.dataset}/erase_{args.erase_num}.json', 'w') as output:
                    json.dump(new_data, output)

    #! execute the mpnns for erased graph
    def fix_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ### Parse args ###
    parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
    seed_best_test = []
    for seed in range(42, 46):
        fix_seed(seed)

        if args.cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

        dataset = load_planetoid_dataset(args.dataset, args.latent, args.erase_num, args.erase_type, args.additional_flag, args.addition_type)
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)

        if args.rand_split:
            split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                            for _ in range(args.runs)]
        elif args.rand_split_class:
            split_idx_lst = [class_rand_splits(
                dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]


        dataset.label = dataset.label.to(device)

        ### Basic information of datasets ###
        n = dataset.graph['num_nodes']
        e = dataset.graph['edge_index'].shape[1]
        c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
        d = dataset.graph['node_feat'].shape[1]

        LOGGER.debug(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

        dataset.graph['edge_index'], dataset.graph['node_feat'] = \
            dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

        ### Load method ###
        model = parse_method(args, n, c, d, device)

        ### Loss function (Single-class, Multi-class) ###
        if args.dataset in ('questions'):
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.NLLLoss()

        ### Performance metric (Acc, AUC) ###
        if args.metric == 'rocauc':
            eval_func = eval_rocauc
        else:
            eval_func = eval_acc

        args.method = args.gnn
        logger = Logger(args.runs, LOGGER, args)

        model.train()
        # LOGGER.debug('MODEL:', model)

        ### Training loop ###
        if args.erase_type == 0: 
            type_tring = " Node "
        if args.erase_type == 1:
            type_tring = " Edge " 

        if args.additional_flag:
            additional_string = "True " + args.addition_type
        else:
            additional_string = "False"
        
        best_test_run = []
        LOGGER.debug(f"******************************************************** {args.dataset} Erase {type_tring} {args.erase_num}, seed {seed}, Additional: {additional_string}, lr: {args.lr}, epoch: {args.epochs}, hidden channels: {args.hidden_channels}, dropout: {args.dropout}, cot: {args.cot} ********************************************************")
        for run in range(args.runs):
            if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
                split_idx = split_idx_lst[0]
            else:
                split_idx = split_idx_lst[run]
            train_idx = split_idx['train'].to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
            best_val = float('-inf')
            best_test = float('-inf')

            for epoch in range(args.epochs):
                model.train()
                optimizer.zero_grad()

                out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                if args.dataset in ('questions'):
                    if dataset.label.shape[1] == 1:
                        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                    else:
                        true_label = dataset.label
                    loss = criterion(out[train_idx], true_label.squeeze(1)[
                        train_idx].to(torch.float))
                else:
                    out = F.log_softmax(out, dim=1)
                    loss = criterion(
                        out[train_idx], dataset.label.squeeze(1)[train_idx])
                loss.backward()
                optimizer.step()

                result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

                logger.add_result(run, result[:-1])

                if result[1] > best_val:
                    best_val = result[1]
                if result[2] > best_test:
                    best_test = result[2]

                if epoch % args.display_step == 0:
                    LOGGER.debug(f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * result[0]:.2f}%, '
                        f'Valid: {100 * result[1]:.2f}%, '
                        f'Test: {100 * result[2]:.2f}%, '
                        f'Best Valid: {100 * best_val:.2f}%, '
                        f'Best Test: {100 * best_test:.2f}%')
            best_test_run.append(best_test)
            logger.print_statistics(run)

        seed_best_test.append(100 * max(best_test_run))
        results = logger.print_statistics()
    
    mean = np.mean(seed_best_test)
    se = stats.sem(seed_best_test)
    acc = ""
    for max_accs in seed_best_test:
        acc += f"& {max_accs} "
    acc += f"& {mean}$\pm${se}"
    LOGGER.debug(f"Erase {args.erase_num} final result: {acc}")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='name of dataset (default: MUTAG), supported dataset: Cora, CiteSeer')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.6,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--erase_num', type = int, default = 1,
                                        help='erease number of node or edge')
    parser.add_argument('--latent', action="store_true",
    					help='use latent graph')
    parser.add_argument('--additional_flag', action="store_true",
    					help='use additional prompt information')
    parser.add_argument('--erase_type', type=int, default=1,
                        help='erase type in graph, 0: nodes, 1: edges, 2:feature (default: 0)')
    parser.add_argument('--addition_type', type=str, default="sc",
                        help='name of addition type (default: sc), supported addition type: sc, tda, combine')
    parser.add_argument('--cot', action="store_true",
    					help='add cot information in prompt flag')
    args = parser.parse_args()
    main(args)