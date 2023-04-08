import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import random
import numpy as np

from opt import * 
from EV_GCN import EV_GCN
#from utils.metrics import accuracy, auc, prf
from utils.metrics import accuracy, auc_2, prf_2 #修改metrics.py之后
from dataloader import dataloader, dataloader_adni, dataloader_odir


if __name__ == '__main__':
    set_seed(123)
    opt = OptInit().initialize()

    print('  Loading dataset ...')
    if opt.dataset == 'ABIDE':
        dl = dataloader()
    if opt.dataset == 'ADNI':
        dl = dataloader_adni()
    if opt.dataset == 'ODIR':
        dl = dataloader_odir()

    raw_features, y, nonimg = dl.load_data()
    # np.savetxt('ODIR1.csv', y, delimiter=",")
    # np.savetxt('ODIR2.csv', nonimg, delimiter=",")

    # np.savetxt('sample.csv', y, delimiter=",")
    n_folds = 5
    cv_splits = dl.data_split(n_folds)


    corrects = np.zeros(n_folds, dtype=np.int32) 
    accs = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold)) 
        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 

        print('  Constructing graph data...')
        # extract node features  
        node_ftr = dl.get_node_features(train_ind)
        # np.savetxt('ODIR3.csv', node_ftr, delimiter=",")

        # get PAE inputs
        edge_index, edgenet_input, pd_ftr_dim, nonimg, n = dl.get_PAE_inputs(nonimg, opt) #1对边（节点对）进行删选：返回值有3个加上, edge_index_copy
        edge_index1, edgenet_input1 = dl.edge_index1(edge_index, pd_ftr_dim, nonimg, n, opt)
        edge_index2, edgenet_input2 = dl.edge_index2(edge_index, pd_ftr_dim, nonimg, n, opt)

        # normalization for PAE
        edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        edgenet_input1 = (edgenet_input1 - edgenet_input1.mean(axis=0)) / edgenet_input1.std(axis=0)
        edgenet_input2 = (edgenet_input2 - edgenet_input2.mean(axis=0)) / edgenet_input2.std(axis=0)

        # new 对节点特征进行操作
        # node_ftr, node_ftr_ = dl.get_node_features1(edge_index, opt) #节点特征添加噪声  #2对边（节点对）进行删选：edge_index-->edge_index_copy,
        
        # build network architecture  
        model = EV_GCN(node_ftr.shape[1], opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg, edgenet_input_dim=2*nonimg.shape[1]).to(opt.device)
        model = model.to(opt.device)

        # build loss, optimizer, metric 
        loss_fn =torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        features_cuda1 = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        # features_cuda2 = torch.tensor(node_ftr_, dtype=torch.float32).to(opt.device) #节点特征添加噪声

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)

        edge_index1 = torch.tensor(edge_index1, dtype=torch.long).to(opt.device)
        edgenet_input1 = torch.tensor(edgenet_input1, dtype=torch.float32).to(opt.device)

        edge_index2 = torch.tensor(edge_index2, dtype=torch.long).to(opt.device)
        edgenet_input2 = torch.tensor(edgenet_input2, dtype=torch.float32).to(opt.device)

        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        def train(): 
            print("  Number of training samples %d" % len(train_ind))
            print("  Start training...\r\n")
            acc = 0
            for epoch in range(opt.num_iter):

                model.train()  
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    node_logits = model(features_cuda1, edge_index, edgenet_input)
                    node_logits1 = model(features_cuda1, edge_index1, edgenet_input1)
                    node_logits2 = model(features_cuda1, edge_index2, edgenet_input2) #节点特征添加噪声：features_cuda2
                    #（1）分类loss
                    loss1 = loss_fn(node_logits1[train_ind], labels[train_ind]) + loss_fn(node_logits2[train_ind], labels[train_ind])
                    #（2）特征loss
                    loss2 = model.loss(node_logits1, node_logits2)
                    loss = loss1 + loss2

                    loss.backward()
                    optimizer.step()
                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                model.eval()

                with torch.set_grad_enabled(False):
                    node_logits = model(features_cuda1, edge_index, edgenet_input)
                    # node_logits1 = model(features_cuda1, edge_index1, edgenet_input1)
                    # node_logits2 = model(features_cuda1, edge_index2, edgenet_input2) #节点特征添加噪声：features_cuda2
                # logits_test1 = node_logits1[test_ind].detach().cpu().numpy()
                # logits_test2 = node_logits2[test_ind].detach().cpu().numpy()
                # logits_test = (logits_test1 + logits_test2) / 2
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                # auc_test = auc(logits_test,y[test_ind])
                # prf_test = prf(logits_test,y[test_ind])
                auc_test = auc_2(logits_test, y[test_ind], opt.num_classes)
                prf_test = prf_2(logits_test, y[test_ind], opt.num_classes)


                print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f}".format(epoch, loss.item(), acc_train.item()))
                if acc_test > acc and epoch >9:
                    acc = acc_test
                    correct = correct_test 
                    aucs[fold] = auc_test
                    prfs[fold] = prf_test

                    if opt.ckpt_path !='':
                        if not os.path.exists(opt.ckpt_path): 
                            #print("Checkpoint Directory does not exist! Making directory {}".format(opt.ckpt_path))
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs[fold] = acc 
            corrects[fold] = correct
            print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))
#<--------------------------------------------------------------------------------------------------------------------->
        # def evaluate():
        #     print("  Number of testing samples %d" % len(test_ind))
        #     print('  Start testing...')
        #     model.load_state_dict(torch.load(fold_model_path))
        #     model.eval()
        #     node_logits, _ = model(features_cuda1, edge_index, edgenet_input)
        #
        #     logits_test = node_logits[test_ind].detach().cpu().numpy()
        #     corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
        #     #auc_test = auc(logits_test,y[test_ind])
        #     #prf_test = prf(logits_test,y[test_ind])
        #     auc_test = auc_2(logits_test,y[test_ind],opt.num_classes)
        #     prf_test = prf_2(logits_test,y[test_ind],opt.num_classes)
        #
        #     print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))
#<--------------------------------------------------------------------------------------------------------------------->

        if opt.train==1: #√
            train()
        # elif opt.train==0: #×
        #     evaluate()

    print("\r\n========================== Finish ==========================") 
    n_samples = raw_features.shape[0]
    acc_nfold = np.sum(corrects)/n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    print("=> Average test sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(se, sp, f1))
    print("代码结果验证 ODIR --> Overall")

    f = open("result_acc_auc_sen_spe_f1.txt", "a+")
    f.write("%.5f  %.5f  %.5f  %.5f  %.5f" % (acc_nfold, np.mean(aucs), se, sp, f1))
    f.write("\r\n\n")
    f.close()

