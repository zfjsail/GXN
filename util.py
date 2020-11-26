from __future__ import print_function
import random
import os
from os.path import join
from tqdm import tqdm
import numpy as np
import networkx as nx
import argparse
import torch
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-data_root', default='GraphClassificationData/', help='The root dir of dataset')
cmd_opt.add_argument('-data', default="twitter", help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=42, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-num_class', type=int, default=2, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='32-32-32-1', help='dimension(s) of latent layers')
cmd_opt.add_argument('-k1', type=float, default=0.9, help='The scale proportion of scale 1')
cmd_opt.add_argument('-k2', type=float, default=0.7, help='The scale proportion of scale 2')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of regression')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')
cmd_opt.add_argument('-deg-as-tag', type=bool, default=True, help='whether to extract final graph features')
cmd_opt.add_argument('-cross_weight', type=float, default=1.0, help='weights for hidden layer cross')
cmd_opt.add_argument('-fuse_weight', type=float, default=1.0, help='weights for final fuse')
cmd_opt.add_argument('-Rhop', type=int, default=1, help='neighborhood hop')
cmd_opt.add_argument('-weight', type=str, default=None, help='saved model parameters')


cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.g = g
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree()).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])


def load_data(root_dir, degree_as_tag):

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    data_file = os.path.join(root_dir, '%s/%s.txt' % (cmd_args.data, cmd_args.data))

    with open(data_file, 'r') as f:
        n_g = int(f.readline().strip())
        row_list = []
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            row_list.append(int(n))
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])

                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
                g.add_edge(j, j)

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_tags, node_features))

        print('max node num: ', np.max(row_list), 'min node num: ', np.min(row_list), 'mean node num: ', np.mean(row_list))


    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags_ = list(dict(g.g.degree).values())

        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags_))
        tagset = list(tagset)
        tag2index = {tagset[i]:i for i in range(len(tagset))}

        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags_), len(tagset))
            g.node_features[range(len(g.node_tags_)), [tag2index[tag] for tag in g.node_tags_]] = 1
        node_feature_flag = True

    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)  # maximum node label (tag)
    if node_feature_flag == True:
        cmd_args.attr_dim = len(tagset)  # dim of node features (attributes)
    else:
        cmd_args.attr_dim = 0

    print('# classes: %d' % cmd_args.num_class)
    print("# data: %d" % len(g_list))

    g_list = sklearn.utils.shuffle(g_list, random_state=cmd_args.seed)

    return g_list


def gen_graph(adj, inf_features, label, cur_node_features):
    # g = nx.Graph()
    # g.add_nodes_from(list(range(len(cur_vids))))
    g = nx.from_numpy_array(adj)
    node_features = np.concatenate((cur_node_features, inf_features), axis=1)  #todo
    g.label = label
    g.remove_nodes_from(list(nx.isolates(g)))
    g.node_features = torch.FloatTensor(node_features)
    return g


def load_self_data(cmd_args):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    file_dir = join(settings.DATA_DIR, cmd_args.data)
    print('loading data ...')

    graphs = np.load(join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

    # wheather a user has been influenced
    # wheather he/she is the ego user
    influence_features = np.load(
        join(file_dir, "influence_feature.npy")).astype(np.float32)
    logger.info("influence features loaded!")

    labels = np.load(join(file_dir, "label.npy"))
    logger.info("labels loaded!")

    vertices = np.load(join(file_dir, "vertex_id.npy"))
    logger.info("vertex ids loaded!")

    vertex_features = np.load(join(file_dir, "vertex_feature.npy"))
    vertex_features = preprocessing.scale(vertex_features)
    # vertex_features = torch.FloatTensor(vertex_features)
    logger.info("global vertex features loaded!")

    n_g = len(graphs)

    for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
        cur_vids = vertices[i]
        cur_node_features = vertex_features[cur_vids]
        g = gen_graph(graphs[i], influence_features[i], labels[i], cur_node_features)
        node_tags = list(range(len(influence_features[0])))
        g_list.append(S2VGraph(g, g.label, node_tags, g.node_features))

        if i > settings.TEST_SIZE:
            break

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        # g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if cmd_args.deg_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

        # tagset = set([])
        # for g in g_list:
        #     tagset = tagset.union(set(g.node_tags_))
        # tagset = list(tagset)
        # tag2index = {tagset[i]:i for i in range(len(tagset))}

        # for g in g_list:
        #     g.node_features = torch.zeros(len(g.node_tags_), len(tagset))
        #     g.node_features[range(len(g.node_tags_)), [tag2index[tag] for tag in g.node_tags_]] = 1
        # node_feature_flag = True

    n_g = len(g_list)
    cmd_args.feat_dim = len(influence_features[0])

    cmd_args.attr_dim = g_list[0].node_features.shape[1]  # dim of node features (attributes)
    print("attr dim", cmd_args.attr_dim)

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)

    return g_list


def sep_data(root_dir, graph_list, fold_idx, seed=0):
    train_idx = np.loadtxt(os.path.join(root_dir, '%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, fold_idx)), dtype=np.int32).tolist()
    test_idx = np.loadtxt(os.path.join(root_dir, '%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, fold_idx)), dtype=np.int32).tolist()
    return [graph_list[i] for i in train_idx], [graph_list[i] for i in test_idx]