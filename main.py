import sys
import os
import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import math
import shutil
from network import GXN
from mlp_dropout import MLPClassifier
from tensorboard_logger import tensorboard_logger
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from util import cmd_args, load_data, sep_data, load_self_data

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

tensorboard_log_dir = 'tensorboard/%s_%s' % ("gxn", cmd_args.data)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
tensorboard_logger.configure(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        model = GXN

        print("latent dim is ", cmd_args.latent_dim)

        self.s2v = model(latent_dim=cmd_args.latent_dim,
                         output_dim=cmd_args.out_dim,
                         num_node_feats=cmd_args.feat_dim + cmd_args.attr_dim,
                         num_edge_feats=0,
                         k=cmd_args.sortpooling_k,
                         ks=[cmd_args.k1, cmd_args.k2],
                         cross_weight=cmd_args.cross_weight,
                         fuse_weight=cmd_args.fuse_weight,
                         R=cmd_args.Rhop)

        print("num_node_feats: ", cmd_args.feat_dim + cmd_args.attr_dim)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = self.s2v.dense_dim

        self.mlp = MLPClassifier(input_size=out_dim,
                                 hidden_size=cmd_args.hidden,
                                 num_class=cmd_args.num_class,
                                 with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag:
                tmp = batch_graph[i].node_features.type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag is False and node_tag_flag:
            node_feat = node_tag
        elif node_feat_flag and node_tag_flag is False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)

        node_feat = node_feat.to(device)
        labels = labels.to(device)

        return node_feat, labels

    def forward(self, batch_graph, device=torch.device('cpu')):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)  # node_feat的尺寸是 [N, D] (DD: n*82)
        N, D = node_feat.shape

        labels = labels.to(device)
        embed, ret_s1, ret_s2 = self.s2v(batch_graph, node_feat, None)
        lbl_t_s1 = torch.ones(N)
        lbl_f_s1 = torch.zeros(N)

        lbl_t_s2 = torch.ones(ret_s2.shape[0] // 2)
        lbl_f_s2 = torch.zeros(ret_s2.shape[0] // 2)

        milbl_s1 = torch.cat((lbl_t_s1, lbl_f_s1), 0).to(device)
        milbl_s2 = torch.cat((lbl_t_s2, lbl_f_s2), 0).to(device)
        logits, cls_loss, acc = self.mlp(embed, labels)
        return logits, cls_loss, acc, ret_s1, milbl_s1, ret_s2, milbl_s2


def loop_dataset(g_list, classifier, mi_loss, sample_idxes, epoch, optimizer=None,
                 bsize=cmd_args.batch_size, device=torch.device('cpu')):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, cls_loss, acc, ret_s1, milbl_s1, ret_s2, milbl_s2 = classifier(batch_graph, device)
        all_scores.append(logits[:, 1].detach())  # for binary classification

        miloss_s1 = mi_loss[0](ret_s1, milbl_s1)
        miloss_s2 = mi_loss[1](ret_s2, milbl_s2)
        miloss = (miloss_s1 + miloss_s2) / 2
        loss = cls_loss + miloss * (2 - epoch / cmd_args.num_epochs)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cls_loss = cls_loss.data.cpu().numpy()
        miloss = miloss.data.cpu().numpy()
        loss = loss.data.cpu().numpy()
        pbar.set_description('cls_loss: %0.5f miloss: %0.5f loss: %0.5f acc: %0.5f' % (cls_loss, miloss, loss, acc))
        total_loss.append(np.array([cls_loss, miloss, loss, acc]) * len(selected_idx))
        n_samples += len(selected_idx)

        # ------------------------------------------------------------------------------------------------------------------
        if optimizer is None:
            print(acc)
        # ------------------------------------------------------------------------------------------------------------------

    if optimizer is None:
        assert n_samples == len(sample_idxes)

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    tensorboard_logger.log_value('train_loss', avg_loss[-2], epoch + 1)

    avg_loss = np.concatenate((avg_loss, [auc]))

    return avg_loss


def evaluate(g_list, classifier, mi_loss, sample_idxes, epoch, optimizer=None,
             bsize=cmd_args.batch_size, device=torch.device('cpu'), return_best_thr=False, thr=None):
    # total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []
    losses = []

    # n_samples = 0
    y_true, y_pred, y_score = [], [], []
    total = 0

    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        out, cls_loss, acc, ret_s1, milbl_s1, ret_s2, milbl_s2 = classifier(batch_graph, device)
        all_scores.append(out[:, 1].detach())  # for binary classification

        miloss_s1 = mi_loss[0](ret_s1, milbl_s1)
        miloss_s2 = mi_loss[1](ret_s2, milbl_s2)
        miloss = (miloss_s1 + miloss_s2) / 2
        loss = cls_loss + miloss * (2 - epoch / cmd_args.num_epochs)

        cls_loss = cls_loss.data.cpu().numpy()
        miloss = miloss.data.cpu().numpy()
        loss = loss.data.cpu().numpy()
        pbar.set_description('cls_loss: %0.5f miloss: %0.5f loss: %0.5f acc: %0.5f' % (cls_loss, miloss, loss, acc))
        # losses.append(np.array([cls_loss, miloss, loss, acc]) * len(selected_idx))
        total += len(selected_idx)
        losses.append(loss)

        y_true += targets
        y_pred += out.max(1)[1].data.tolist()
        y_score += out[:, 1].data.tolist()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                sum(losses) / total, auc, prec, rec, f1)
    loss_ret = sum(losses) / total

    if return_best_thr:
        log_desc = "valid_"
    else:
        log_desc = "test_"
    tensorboard_logger.log_value(log_desc + 'loss', loss_ret, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'auc', auc, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'f1', f1, epoch + 1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return loss_ret, [prec, rec, f1, auc], best_thr
    else:
        return loss_ret, [prec, rec, f1, auc], None


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_run(cmd_args, g_list, device, foldidx, first_timstr):
    train_graphs, test_graphs = sep_data(cmd_args.data_root, g_list, foldidx)

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier().to(device)
    print("Number of Model Parameters: ", count_parameters(classifier))

    optimizer = optim.Adam(classifier.parameters(),
                           lr=cmd_args.learning_rate,
                           amsgrad=True,
                           weight_decay=0.001)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    max_acc = 0.0
    mi_loss = [nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()]

    timstr = datetime.datetime.now().strftime("%m%d-%H%M%S")
    logfile = './log_%s/log_%s/testlog_%s_%s.txt' % (cmd_args.data, first_timstr, cmd_args.data, timstr)

    if not os.path.exists('./log_%s/log_%s' % (cmd_args.data, first_timstr)):
        os.makedirs('./log_%s/log_%s' % (cmd_args.data, first_timstr))
    if not os.path.exists('./result_%s/result_%s' % (cmd_args.data, first_timstr)):
        os.makedirs('./result_%s/result_%s' % (cmd_args.data, first_timstr))
        with open('./result_%s/result_%s/acc_result_%s_%s.txt' % (
        cmd_args.data, first_timstr, cmd_args.data, first_timstr), 'a+') as f:
            f.write(str(cmd_args) + '\n')

    if not os.path.exists('./checkpoint_%s/time_%s/FOLD%s' % (cmd_args.data, first_timstr, foldidx)):
        os.makedirs('./checkpoint_%s/time_%s/FOLD%s' % (cmd_args.data, first_timstr, foldidx))

    if cmd_args.weight is not None:
        classifier.load_state_dict(torch.load(cmd_args.weight))
        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, mi_loss, list(range(len(test_graphs))), epoch=0,
                                 device=device)
        with open(logfile, 'a+') as log:
            log.write('clsloss: %.5f miloss: %.5f loss %.5f acc %.5f auc %.5f'
                      % (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]) + '\n')
        print('Best Acc:', test_loss[3])
        raise ValueError('Stop Testing')

    with open(logfile, 'a+') as log:
        log.write(str(cmd_args) + '\n')
        log.write('Fold index: ' + str(foldidx) + '\n')

    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, mi_loss, train_idxes, epoch, optimizer=optimizer,
                                device=device)
        avg_loss[4] = 0.0
        print('\033[92maverage training of epoch %d: clsloss: %.5f miloss: %.5f loss %.5f acc %.5f auc %.5f\033[0m'
              % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4]))  # noqa

        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, mi_loss, list(range(len(test_graphs))), epoch, device=device)
        test_loss[4] = 0.0
        print('\033[93maverage test of epoch %d: clsloss: %.5f miloss: %.5f loss %.5f acc %.5f auc %.5f\033[0m'
              % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]))  # noqa

        with open(logfile, 'a+') as log:
            log.write('test of epoch %d: clsloss: %.5f miloss: %.5f loss %.5f acc %.5f auc %.5f'
                      % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]) + '\n')

        if test_loss[3] > max_acc:
            max_acc = test_loss[3]
            fname = './checkpoint_%s/time_%s/FOLD%s/model_epoch%s.pt' % (
            cmd_args.data, first_timstr, foldidx, str(epoch))
            torch.save(classifier.state_dict(), fname)

    with open('./result_%s/result_%s/acc_result_%s_%s.txt' % (cmd_args.data, first_timstr, cmd_args.data, first_timstr),
              'a+') as f:
        f.write('\n')
        f.write('Fold index: ' + str(foldidx) + '\t')
        f.write(str(max_acc) + '\n')

    if cmd_args.extract_features:
        features, labels = classifier.output_features(train_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_train.txt',
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
        features, labels = classifier.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_test.txt',
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

    return max_acc


def model_run_new(cmd_args, g_list, device, first_timstr):
    n_g = len(g_list)
    train_ratio = 0.5
    valid_ratio = 0.25
    n_train = int(n_g * train_ratio)
    n_valid = int((train_ratio + valid_ratio) * n_g) - n_train
    train_graphs = [g_list[i] for i in range(0, n_train)]
    valid_graphs = [g_list[i] for i in range(n_train, n_train + n_valid)]
    test_graphs = [g_list[i] for i in range(n_train + n_valid, n_g)]

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier().to(device)
    print("Number of Model Parameters: ", count_parameters(classifier))

    optimizer = optim.Adam(classifier.parameters(),
                           lr=cmd_args.learning_rate,
                           amsgrad=True,
                           weight_decay=0.001)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    max_acc = 0.0
    mi_loss = [nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()]

    timstr = datetime.datetime.now().strftime("%m%d-%H%M%S")
    logfile = './log_%s/log_%s/testlog_%s_%s.txt' % (cmd_args.data, first_timstr, cmd_args.data, timstr)

    if not os.path.exists('./log_%s/log_%s' % (cmd_args.data, first_timstr)):
        os.makedirs('./log_%s/log_%s' % (cmd_args.data, first_timstr))
    if not os.path.exists('./result_%s/result_%s' % (cmd_args.data, first_timstr)):
        os.makedirs('./result_%s/result_%s' % (cmd_args.data, first_timstr))
        with open('./result_%s/result_%s/acc_result_%s_%s.txt' % (
        cmd_args.data, first_timstr, cmd_args.data, first_timstr), 'a+') as f:
            f.write(str(cmd_args) + '\n')

    if not os.path.exists('./checkpoint_%s/time_%s/FOLD%s' % (cmd_args.data, first_timstr, None)):
        os.makedirs('./checkpoint_%s/time_%s/FOLD%s' % (cmd_args.data, first_timstr, None))

    with open(logfile, 'a+') as log:
        log.write(str(cmd_args) + '\n')

    best_model = None

    classifier.eval()

    val_loss, val_metrics, thr = evaluate(valid_graphs, classifier, mi_loss, list(range(len(valid_graphs))), -1,
                                          device=device, return_best_thr=True)
    print("\nvalidation loss:", val_loss, "metrics", val_metrics, "thr:", thr)

    test_loss, test_metrics, _ = evaluate(test_graphs, classifier, mi_loss, list(range(len(test_graphs))), -1,
                                          device=device, thr=thr)
    print("\ntest loss:", test_loss, "metrics", test_metrics)

    best_thr = None
    best_valid_metrics = None
    best_test_metrics = None
    best_epoch = 0

    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, mi_loss, train_idxes, epoch, optimizer=optimizer,
                                device=device)
        avg_loss[4] = 0.0
        print('\033[92m\naverage training of epoch %d: clsloss: %.5f miloss: %.5f loss %.5f acc %.5f auc %.5f\033[0m'
              % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4]))  # noqa

        classifier.eval()

        val_loss, val_metrics, thr = evaluate(valid_graphs, classifier, mi_loss, list(range(len(valid_graphs))), epoch, device=device, return_best_thr=True)
        print("\nvalidation loss:", val_loss, "metrics", val_metrics, "thr:", thr)

        test_loss, test_metrics, _ = evaluate(test_graphs, classifier, mi_loss, list(range(len(test_graphs))), epoch, device=device, thr=thr)
        print("\ntest loss:", test_loss, "metrics", test_metrics)

        # if not cmd_args.printAUC:
        #     test_loss[2] = 0.0
        # print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2])) # noqa
        # max_acc = max(max_acc, test_loss[1])
        if val_metrics[-1] > max_acc:
            max_acc = val_metrics[-1]
            best_thr = thr
            best_valid_metrics = val_metrics
            best_test_metrics = test_metrics
            logger.info("******************BEST UNTIL NOW***********************")

    print("Finally, best thr:", best_thr, "best epoch", best_epoch)
    print("best valid metrics", best_valid_metrics)
    print("best test metrics", best_test_metrics)


if __name__ == '__main__':
    set_randomseed(cmd_args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_timstr = datetime.datetime.now().strftime("%m%d-%H%M%S")

    if cmd_args.data in ['DD', 'PROTEINS']:
        g_list = load_data(cmd_args.data_root, degree_as_tag=False)
    elif cmd_args.data in ['COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'ENZYMES']:
        g_list = load_data(cmd_args.data_root, degree_as_tag=True)
    else:
        g_list = load_self_data(cmd_args)

    # print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))
    print('# num of classes: ', cmd_args.num_class)

    print('Lets start a single-fold validation')
    print('start training ------> fold', cmd_args.fold)
    # model_run(cmd_args, g_list, device, cmd_args.fold, first_timstr)
    model_run_new(cmd_args, g_list, device, first_timstr)
