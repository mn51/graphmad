# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
GraphMAD.
'''
import argparse
import logging

import numpy as np
import random
from matplotlib import pyplot as plt
import scipy as sp

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import degree,to_dense_adj,dense_to_sparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import os
import csv

from model import *
from utils import *
from gmixup import *
from graphon import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')

# INPUT:
# row: Vector of data to write to row of csv
# file_name: Name of file to be written to
# path: Path of file
def save_results(row,file_name=None,path=None):
    if path is None:
        path = os.getcwd()
    # Create file if not in folder path
    if file_name is not None:
        file_path = path
        # If folder does not exist, make it
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Write given row to file
        with open(file_path+'/'+file_name,'a') as f:
            writer = csv.writer(f,delimiter=',')
            writer.writerow(row)
    else:
        # Default name: 'resultsXX.txt' with lowest unused number in XX
        file_count = 0
        file_name = 'results'+f'{file_count:03d}'+'.txt'
        file_path = path
        # If folder does not exist, make it
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Find lowest number that has not been used yet
        while os.path.exists(file_path+'/'+file_name):
            file_count += 1
            file_name = 'results'+f'{file_count:03d}'+'.txt'
        # Write given row to file
        with open(file_path+'/'+file_name,'w') as f:
            writer = csv.writer(f,delimiter=',')
            writer.writerow(row)
    return

# ~INPUT~:
# message: String to be output
# status_file_name: File to write string to, if given
# logger: Logging object to write string to, if given (if neither logger nor status_file_name, print to output)
# ~INPUT~:
# message
def update_status(message,status_file_name=None,logger=None):
    if status_file_name is not None:
        exp_status_file = open(status_file_name,'a')
        exp_status_file.write(message + '\n')
        exp_status_file.close()
    if logger is not None:
        logger.info(message)
    if (logger is None) and (status_file_name is None):
        print(message)
    return message

# ~INPUT~:
# model: PyTorch model
# loader: PyTorch data loader
# ~OUTPUT~:
# model: Updated PyTorch model
# train_acc: Accuracy of training data
# train_loss: Training loss
def train(model,loader):
    model.train()
    loss_all = 0
    graph_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x,data.edge_index,data.batch)
        y = data.y.view(-1,num_classes)
        loss = mixup_cross_entropy_loss(output,y)
        loss.backward()
        loss_all += loss.item()*data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    train_loss = loss_all/graph_all
    train_acc = 0
    return model,train_acc,train_loss

# ~INPUT~:
# model: PyTorch model
# loader: PyTorch data loader
# ~OUTPUT~:
# test_acc: Accuracy of testing data
# test_loss: Testing loss
def test(model,loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x,data.edge_index,data.batch)
        pred = output.max(dim=1)[1]
        y = data.y.view(-1,num_classes)
        loss += mixup_cross_entropy_loss(output,y).item()*data.num_graphs
        y = y.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    test_acc = correct/total
    test_loss = loss/total
    return test_acc,test_loss

# ~INPUT~:
# dataset_name: String. If 'Synthetic', download synthetic dataset from datapath. If not, look for or load to PyG dataset in data_path.
# ~OUTPUT~:
# dataset with one-hot vector labels
def prepare_dataset(dataset_name,data_path):
    if dataset_name=='Synthetic':
        dataset = torch.load(data_path)
    else:
        dataset = TUDataset(data_path+dataset_name, name=dataset_name)
    dataset = list(dataset)

    # One-hot labels
    y_set = set()
    for graph in dataset:
        graph.y = graph.y.view(-1)
        y_set.add(int(graph.y))
    num_classes = len(y_set)
    for graph in dataset:
        graph.y = F.one_hot(graph.y, num_classes=num_classes).to(torch.float)[0]

    return dataset

# ~INPUT~:
# lmbda_bounds: Smallest and largest values for mixup parameter.
# aug_num: Number of new samples
# aug_ratio: Ratio of new samples to original number of training samples
# ~OUTPUT~:
# lmbda_list: List of mixup parameter values
def select_mixup_parameter(lmbda_bounds,aug_num,aug_ratio):
    lmbda_list = list(np.linspace(lmbda_bounds[0]*.5,lmbda_bounds[1]*.5,int(aug_num/2)+2)[1:-1])+list(1-np.flip(np.linspace(lmbda_bounds[0]*.5,lmbda_bounds[1]*.5,aug_num-int(aug_num/2)+2)[1:-1]))
    return lmbda_list

# ~INPUT~:
# dataset: PyTorch dataset
# num_classes: Number of classes in dataset. Will find if not given.
# mixup_batch_size: Number of graphs to be converted to graphons
# nKnots: Knots in B-spline approximation of graphon
# sorted: If sorting graphs before estimating graphons
# sas_only: Estimate graphons via SAS
# z_init: Value of graphon latent sample points (B-spline estimation only)
# ~OUTPUT~:
# Ws_subset: List of graphons (Graphon objects)
# labels_subset: Labels of graphs from which graphons estimated
def graphs_to_graphons(dataset,num_classes=None,mixup_batch_size=20,nKnots=10,sorted=False,sas_only=True,z_init=None):
    # Get labels from training dataset
    if num_classes is None:
        num_classes = len(dataset[0].y)
    labels = np.array([int(np.where(dataset[i].y==1)[0]) for i in range(len(dataset[:train_nums]))])

    # Get all adjacency matrices from dataset
    As = [np.array(to_dense_adj(dataset[i].edge_index)[0]) for i in range(len(dataset))]
    # If sampling subsets of each class
    if mixup_batch_size<len(dataset):
        class_idx = [[] for c_i in range(num_classes)]
        for c_i in range(num_classes):
            class_loc = np.array([int(dataset[i].y[c_i]==1) for i in range(len(dataset))])
            class_idx[c_i] = np.where(class_loc==1)[0]
            class_idx[c_i] = class_idx[c_i][np.random.permutation(len(class_idx[c_i]))[:mixup_batch_size]]
        As = [As[class_idx[c_i][i]] for c_i in range(num_classes) for i in range(mixup_batch_size)]
        labels_subset = [labels[class_idx[c_i][i]] for c_i in range(num_classes) for i in range(mixup_batch_size)]

    # Estimate graphon per graph
    Ws_subset = []
    for A in As:
        (N_curr,N_curr) = A.shape

        # If sorting adjacency matrices before graphon estimation
        if sorted:
            deg = np.sum(A,axis=0)
            ord_deg = np.argsort(deg)
            A = A[ord_deg,:][:,ord_deg]
            z = (deg[ord_deg]/max(deg+1)) if (z_init is None) else np.sort(z_init)
        else:
            z = np.linspace(0,1,N_curr) if (z_init is None) else z_init

        # If SAS only or using B-spline estimation
        if sas_only or N_curr>200:
            if N_curr<4:
                num_bins = int(N_curr)
            elif N_curr<10:
                num_bins=int(N_curr/4)
            else:
                num_bins=int(N_curr/10)
            W_sample = graphon_from_sas(A,z=z,num_bins=num_bins,nKnots=nKnots)
        else:
            W_sample = graphon_from_graph_bspline(A,z=z,nKnots=nKnots)
        Ws_subset.append(W_sample)
    return Ws_subset,labels_subset

# ~INPUT~:
# dataset: PyTorch dataset
# train_nums: Number of training samples
# mixup_func: Mixup function for labels
# lmbda_list: List of mixup parameter values
# num_sample: Number of graphs to sample per mixup parameter
# nKnots: Knots in B-spline approximation of graphon
# y_list: Labels of graphons in Theta
# Theta_clusterpath: CARP clusterpath of Theta, subset of graphons
# V_clusterpath: CARP cluster assignment matrices of Theta, subset of graphons
# gamma_clusterpath: CARP values of mixup parameters
# aligned: Align adjacency matrices before estimating class graphons
# sas_only: Estimate graphons via SAS
# sorted: If new graphs sampled from graphons are sorted by latent points
# z_init: Graphon latent sample points (B-spline estimation only)
# z_mixup: Graphon latent sample points to sample from in graphons
# ~OUTPUT~:
# new_graph: New dataset of G-Mixup graphs
# W_gmixup: List of class graphons
def mixup_gmixup(dataset,train_nums,mixup_func,lmbda_list,
                 num_sample=50,nKnots=10,
                 y_list=None,Theta_clusterpath=None,V_clusterpath=None,gamma_clusterpath=None,
                 aligned=False,sas_only=True,sorted=False,
                 z_init=None,z_mixup=None
                ):
    num_all_nodes = [dataset[i].num_nodes for i in range(len(dataset))]
    median_num_nodes = int(np.median(num_all_nodes))

    # Estimate graphons per class
    class_graphs = split_class_graphs(dataset[:train_nums])
    K = len(class_graphs)
    graphons = []
    for label, graphs in class_graphs:
        if aligned:
            # Align graphs and pad if necessary
            align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
                graphs, padding=True, N=int(median_num_nodes))
            graphs = [align_graphs_list[i].copy() for i in range(len(align_graphs_list))]
            deg = np.sum(np.sum(graphs,axis=0)/len(graphs),axis=0)
            ord_deg = np.argsort(deg)
            z = (deg[ord_deg]/max(deg+1)) if (z_init is None) else np.sort(z_init)
        else:
            # Pad if necessary
            padded_graphs = []
            max_nodes = np.max(num_all_nodes)
            for i in range(len(graphs)):
                padded_graph = np.zeros((max_nodes,max_nodes))
                N_curr = graphs[i].shape[0]
                padded_graph[:N_curr,:N_curr] = graphs[i].copy()
                padded_graph = padded_graph[:int(median_num_nodes),:int(median_num_nodes)]
                padded_graphs.append(padded_graph)
            graphs = [padded_graphs[i].copy() for i in range(len(padded_graphs))]
            z = np.linspace(0,1,int(median_num_nodes)) if (z_init is None) else z_init

        # Estimate graphon for each class
        if median_num_nodes>200 or sas_only:
            # SAS if graphs are too large
            graphon = graphon_from_sas(np.sum(graphs,axis=0)/len(graphs),z=z,num_bins=int(median_num_nodes/10),nKnots=nKnots)
        else:
            graphon = graphon_from_graph_bspline(np.sum(graphs,axis=0)/len(graphs),z=z,nKnots=nKnots)
        graphons.append((label, graphon))
    # Graphon per class
    W_gmixup = [graphons[i][1] for i in range(len(graphons))]

    # Number of nodes for new samples
    if z_mixup is not None:
        N_mixup = len(z_mixup)
    else:
        N_mixup = int(median_num_nodes)

    # Sample graphs per fusion parameter
    new_graph = []
    for lmbda in lmbda_list:
        graphon_sel = np.random.permutation(K)[:2]
        
        # When lmbda=0, choose graphon_sel[1] and vice versa
        new_theta = lmbda * W_gmixup[graphon_sel[0]].theta + (1 - lmbda) * W_gmixup[graphon_sel[1]].theta
        W_newgmixup = Graphon(theta=new_theta,size=int(median_num_nodes),nKnots=nKnots)

        # Mixup labels
        if mixup_func=='linear':
            y_gmixup = lmbda*graphons[graphon_sel[0]][0] + (1-lmbda)*graphons[graphon_sel[1]][0]
        elif mixup_func=='clusterpath':
            # If clusterpath not provided, use linear interpolation
            if (Theta_clusterpath is None) or (V_clusterpath is None) or (gamma_clusterpath is None):
                y_gmixup = lmbda*graphons[graphon_sel[0]][0] + (1-lmbda)*graphons[graphon_sel[1]][0]
            else:
                # Interpolation paths
                Theta_interp = sp.interpolate.interp1d(gamma_clusterpath,np.array(Theta_clusterpath),axis=0)
                V_interp = np.array([[shrink_dist(V_clusterpath[i][j,:],0) for i in range(len(Theta_clusterpath))] for j in range(V_clusterpath[0].shape[0])])
                V_interp = np.array([lowtri2mat(V_interp[:,i]) for i in range(len(gamma_clusterpath))])
                V_interp = sp.interpolate.interp1d(gamma_clusterpath,V_interp,axis=0)
                
                # Find critical fusion points
                lmbda_nontriv,K_nontriv = find_inf_lmbda(Theta_clusterpath,V_clusterpath,gamma_clusterpath,K=1,fid_dist=fid_dist,shrink_dist=shrink_dist,lmbda_hi=1,lmbda_low=0.,eps_thresh=eps_thresh)
                lmbda_max,K_max = find_inf_lmbda(Theta_clusterpath,V_clusterpath,gamma_clusterpath,K=K-1,fid_dist=fid_dist,shrink_dist=shrink_dist,lmbda_hi=lmbda_nontriv,lmbda_low=0.,eps_thresh=eps_thresh)
                lmbda_min,K_min = find_inf_lmbda(Theta_clusterpath,V_clusterpath,gamma_clusterpath,K=K,fid_dist=fid_dist,shrink_dist=shrink_dist,lmbda_hi=lmbda_max,lmbda_low=0.,eps_thresh=eps_thresh)
                if K_max<2 or K==2:
                    lmbda_nontriv = lmbda_max
                    K_nontriv = K_max
                lmbda_crit = .5*(lmbda_max+lmbda_min)
                theta_fus = Theta_interp(lmbda_crit)
                clust_fus,K_fus = get_clustassignment(diff_theta=V_interp(lmbda_crit),shrink_dist=shrink_dist,eps_thresh=eps_thresh)
                if K_fus<2:
                    for pow_iter in np.linspace(0,lmbda_min,10):
                        lmbda_crit = lmbda_min-pow_iter
                        theta_fus = Theta_interp(lmbda_crit)
                        clust_fus,K_fus = get_clustassignment(diff_theta=V_interp(lmbda_crit),shrink_dist=shrink_dist,eps_thresh=eps_thresh)
                        if K_fus>1:
                            break
                
                # Get clusterpath per main branch
                lmbda_trav = [np.linspace(0,1,len(gamma_clusterpath)) for k in range(K_fus)]
                theta_cc = list(Theta_interp(lmbda_trav[0]))
                theta_clusterpath = [[[] for i in range(len(gamma_clusterpath))] for k in range(K_fus)]
                for k in range(K_fus):
                    for lmbda_idx in range(len(gamma_clusterpath)):
                        theta_clusterpath[k][lmbda_idx] = np.sum([theta_cc[lmbda_idx][i,:] for i in clust_fus[k]],axis=0)/len(clust_fus[k])
                lmbda_clusterpath = [[] for k in range(K_fus)]
                for k in range(K_fus):
                    lmbda_diff = [0]+[fid_dist(theta_clusterpath[k][lmbda_idx],theta_clusterpath[k][lmbda_idx+1]) for lmbda_idx in np.arange(len(gamma_clusterpath)-1)]
                    lmbda_diff = np.cumsum(lmbda_diff)
                    lmbda_clusterpath[k] = lmbda_diff/lmbda_diff[-1]
                
                # Select branches for fusing
                # Number of branches may not match true classes
                if K_fus==1:
                    # If one branch, can only choose one
                    cp_sel = np.array([0,0])
                elif K_fus>1 and K_fus<K:
                    # If too many branches, find best fitting branches for each class (if given labels for list)
                    if y_list is not None:
                        y_inds = [np.where(np.array(y_list)==c_i)[0] for c_i in range(num_classes)]
                        y_hat = [np.zeros(num_classes) for i in range(K_fus)]
                        clust_fus = [np.array(clust_fus[k]) for k in range(K_fus)]
                        for k in range(K_fus):
                            for c_i in range(num_classes):
                                y_hat[k][c_i] = np.sum(clust_fus[k].reshape(-1,1)==y_inds[c_i].reshape(1,-1))/len(clust_fus[k])
                        cp_sel = np.array([0,0])
                        cp_sel[0] = np.argmax(np.array(y_hat)[:,graphon_sel[0]])
                        cp_sel[1] = np.argmax(np.array(y_hat)[:,graphon_sel[1]])
                    else:
                        cp_sel = np.random.randint(0,K_fus,K).astype(int)
                elif K_fus>K:
                    # If too many branches, find best fitting branches for each class (if given labels for list)
                    if y_list is not None:
                        y_inds = [np.where(np.array(y_list)==c_i)[0] for c_i in range(num_classes)]
                        y_hat = [np.zeros(num_classes) for i in range(K_fus)]
                        clust_fus = [np.array(clust_fus[k]) for k in range(K_fus)]
                        for k in range(K_fus):
                            for c_i in range(num_classes):
                                y_hat[k][c_i] = np.sum(clust_fus[k].reshape(-1,1)==y_inds[c_i].reshape(1,-1))/len(clust_fus[k])
                        cp_sel = np.array([0,0])
                        cp_sel[0] = np.argmax(np.array(y_hat)[:,graphon_sel[0]])
                        cp_sel[1] = np.argmax(np.array(y_hat)[:,graphon_sel[1]])
                    else:
                        cp_sel = np.random.permutation(K_fus)[:K]
                else:
                    # If number of branches matches classes, assume that they match classes
                    cp_sel = graphon_sel.copy()
                # Create clusterpath between the two branches selected for classes
                label_clusterpair = np.concatenate((.5*lmbda_clusterpath[cp_sel[0]][:-1],1-.5*np.flip(lmbda_clusterpath[cp_sel[1]])))
                # Create sp interpolate object between 0 and 1, where lmbda controls path along pairwise clusterpath
                clusterpath_interp = sp.interpolate.interp1d(np.linspace(0,1,len(label_clusterpair)),label_clusterpair)
                lmbda_cp = float(clusterpath_interp(lmbda))
                y_gmixup = lmbda_cp*graphons[graphon_sel[0]][0] + (1-lmbda_cp)*graphons[graphon_sel[1]][0]
        elif mixup_func=='sigmoid':
            lmbda_sigmoid = 1/(1+np.exp(-2*(lmbda*10-5)))
            y_gmixup = lmbda_sigmoid*graphons[graphon_sel[0]][0] + (1-lmbda_sigmoid)*graphons[graphon_sel[1]][0]
        elif mixup_func=='logit':
            lmbda_scaled = lmbda*99/101+1/101
            lmbda_logit = np.log(lmbda_scaled/(1-lmbda_scaled))/(2*2*5)+.5
            y_gmixup = lmbda_logit*graphons[graphon_sel[0]][0] + (1-lmbda_logit)*graphons[graphon_sel[1]][0]

        sample_graphs = []
        for i in range(num_sample):
            # Sample new graph from current graphon
            sample_graph,_ = W_newgmixup.sample_graph(N=N_mixup,z=z_mixup,sorted=sorted)
            # If no nodes, skip adding new graph
            if np.sum(sample_graph)==0:
                continue
            # Remove isolated nodes
            sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
            sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]
            
            A = torch.from_numpy(sample_graph)
            edge_index, _ = dense_to_sparse(A)
            num_nodes = sample_graph.shape[0]

            pyg_graph = Data()
            pyg_graph.y = torch.Tensor(y_gmixup)
            pyg_graph.edge_index = edge_index
            pyg_graph.num_nodes = num_nodes
            sample_graphs.append(pyg_graph)
        new_graph += sample_graphs
    return new_graph,W_gmixup

# ~INPUT~:
# y_list: Labels of graphons in Theta
# lmbda_list: List of mixup parameter values
# mixup_func: Mixup function for labels
# Theta_clusterpath: CARP clusterpath of Theta, subset of graphons
# V_clusterpath: CARP cluster assignment matrices of Theta, subset of graphons
# gamma_clusterpath: CARP values of mixup parameters
# fid_dist: Fidelity distance for convex clustering
# shrink_dist: Shrinkage distance for convex clustering
# num_sample: Number of graphs to sample per mixup parameter
# eps_thresh: Threshold for clustering samples in CARP
# sorted: If new graphs sampled from graphons are sorted by latent points
# z_mixup: Graphon latent sample points to sample from in graphons
# ks_mixup: Clusterpath branches to select
# ~OUTPUT~:
# new_graph: New dataset of G-Mixup graphs
# Ws_graphmad: List of graphon centroids for each mixup parameter
def mixup_graphmad(lmbda_list,y_list,mixup_func,
                   Theta_clusterpath,V_clusterpath,gamma_clusterpath,
                   fid_dist=fid_dist,shrink_dist=shrink_dist,
                   num_sample=50,eps_thresh=0.,
                   sorted=False,
                   z_mixup=None,ks_mixup=None
                  ):
    # Interpolation paths
    Theta_interp = sp.interpolate.interp1d(gamma_clusterpath,np.array(Theta_clusterpath),axis=0)
    V_interp = np.array([[shrink_dist(V_clusterpath[i][j,:],0) for i in range(len(Theta_clusterpath))] for j in range(V_clusterpath[0].shape[0])])
    V_interp = np.array([lowtri2mat(V_interp[:,i]) for i in range(len(gamma_clusterpath))])
    V_interp = sp.interpolate.interp1d(gamma_clusterpath,V_interp,axis=0)

    # Find critical fusion points
    lmbda_nontriv,K_nontriv = find_inf_lmbda(Theta_clusterpath,V_clusterpath,gamma_clusterpath,K=1,fid_dist=fid_dist,shrink_dist=shrink_dist,lmbda_hi=1,lmbda_low=0.,eps_thresh=eps_thresh)
    lmbda_max,K_max = find_inf_lmbda(Theta_clusterpath,V_clusterpath,gamma_clusterpath,K=K-1,fid_dist=fid_dist,shrink_dist=shrink_dist,lmbda_hi=lmbda_nontriv,lmbda_low=0.,eps_thresh=eps_thresh)
    lmbda_min,K_min = find_inf_lmbda(Theta_clusterpath,V_clusterpath,gamma_clusterpath,K=K,fid_dist=fid_dist,shrink_dist=shrink_dist,lmbda_hi=lmbda_max,lmbda_low=0.,eps_thresh=eps_thresh)
    if K_max<2 or K==2:
        lmbda_nontriv = lmbda_max
        K_nontriv = K_max
    lmbda_crit = .5*(lmbda_max+lmbda_min)
    theta_fus = Theta_interp(lmbda_crit)
    clust_fus,K_fus = get_clustassignment(diff_theta=V_interp(lmbda_crit),shrink_dist=shrink_dist,eps_thresh=eps_thresh)
    if K_fus<2:
        for pow_iter in np.linspace(0,lmbda_min,10):
            lmbda_crit = lmbda_min-pow_iter
            theta_fus = Theta_interp(lmbda_crit)
            clust_fus,K_fus = get_clustassignment(diff_theta=V_interp(lmbda_crit),shrink_dist=shrink_dist,eps_thresh=eps_thresh)
            if K_fus>1:
                break

    # From CARP gamma path and CARP theta path, get interpolation function
    new_graph = []
    Ws_graphmad = []
    for i_lmbda,lmbda in enumerate(lmbda_list):
        # Get theta, cluster assignments, and number of clusters at lambda (scaled to [0,lmbda_nontriv])
        theta_mixup = Theta_interp(lmbda)
        clust_mixup,K_mixup = get_clustassignment(diff_theta=V_interp(lmbda),shrink_dist=shrink_dist,eps_thresh=eps_thresh)
        if K_mixup>=K_fus:
            # Extended clusterpath at number of classes
            K_mixup = K_fus
            theta_centroids_mixup = [np.sum([theta_mixup[i,:] for i in clust_fus[k]],axis=0)/len(clust_fus[k]) for k in range(K_fus)]

            # Class ratio per branch
            y_inds = [np.where(np.array(y_list)==c_i)[0] for c_i in range(num_classes)]
            y_hat = [np.zeros(num_classes) for i in range(K_fus)]
            clust_fus = [np.array(clust_fus[k]) for k in range(K_fus)]
            for k in range(K_fus):
                for c_i in range(num_classes):
                    y_hat[k][c_i] = np.sum(clust_fus[k].reshape(-1,1)==y_inds[c_i].reshape(1,-1))/len(clust_fus[k])
        else:
            # Fewer clusters than classes
            theta_centroids_mixup = [np.sum([theta_mixup[i,:] for i in clust_mixup[k]],axis=0)/len(clust_mixup[k]) for k in range(K_mixup)]
            
            # Class ratio per branch
            y_inds = [np.where(np.array(y_list)==c_i)[0] for c_i in range(num_classes)]
            y_hat = [np.zeros(num_classes) for i in range(K_mixup)]
            clust_mixup = [np.array(clust_mixup[k]) for k in range(K_mixup)]
            for k in range(K_mixup):
                for c_i in range(num_classes):
                    y_hat[k][c_i] = np.sum(clust_mixup[k].reshape(-1,1)==y_inds[c_i].reshape(1,-1))/len(clust_mixup[k])
        K_mixup = np.minimum(K_mixup,K_fus)

        # Choose which branch to select
        if ks_mixup is None:
            k_mixup = np.random.randint(K_mixup)
        elif np.isscalar(ks_mixup):
            k_mixup = np.maximum(np.minimum(ks_mixup,K_mixup,1))
        else:
            assert len(ks_mixup)==len(lmbda_list)
            k_mixup = ks_mixup[i_lmbda]

        # Number of nodes
        if z_mixup is not None:
            N_mixup = len(z_mixup)
        else:
            N_mixup = int(median_num_nodes)

        # Get mixup theta at selected branch
        theta_graphmad = theta_centroids_mixup[k_mixup]

        # Get mixup label given mixup function
        if mixup_func=='linear':
            y_graphmad = y_hat[k_mixup]*(1-lmbda) + (1/num_classes)*np.ones(num_classes)*lmbda
        elif mixup_func=='clusterpath':
            # Get clusterpath per main branch
            lmbda_trav = [np.linspace(0,1,len(gamma_clusterpath)) for k in range(K_fus)]
            theta_cc = list(Theta_interp(lmbda_trav[0]))
            theta_clusterpath = [[[] for i in range(len(gamma_clusterpath))] for k in range(K_fus)]
            for k in range(K_fus):
                for lmbda_idx in range(len(gamma_clusterpath)):
                    theta_clusterpath[k][lmbda_idx] = np.sum([theta_cc[lmbda_idx][i,:] for i in clust_fus[k]],axis=0)/len(clust_fus[k])
            lmbda_clusterpath = [[] for k in range(K_fus)]
            for k in range(K_fus):
                lmbda_diff = [0]+[fid_dist(theta_clusterpath[k][lmbda_idx],theta_clusterpath[k][lmbda_idx+1]) for lmbda_idx in np.arange(len(gamma_clusterpath)-1)]
                lmbda_diff = np.cumsum(lmbda_diff)
                lmbda_clusterpath[k] = lmbda_diff/lmbda_diff[-1]
            y_clusterpath = [[] for k in range(K_mixup)]
            for k in range(K_mixup):
                y_clusterpath[k] = y_hat[k].reshape(-1,1)*(1-lmbda_clusterpath[k].reshape(1,-1)) + (1/num_classes)*np.ones(num_classes).reshape(-1,1)*(lmbda_clusterpath[k].reshape(1,-1))
            y_graphmad = np.array([float(sp.interpolate.interp1d(lmbda_clusterpath[k_mixup],y_clusterpath[k_mixup][l])(lmbda)) for l in range(K)])
        elif mixup_func=='sigmoid':
            lmbda_sigmoid = 1/(1+np.exp(-2*(lmbda*10-5)))
            y_graphmad = y_hat[k_mixup]*(1-lmbda_sigmoid) + (1/num_classes)*np.ones(num_classes)*lmbda_sigmoid
        elif mixup_func=='logit':
            lmbda_scaled = lmbda*99/101+1/101
            lmbda_logit = np.log(lmbda_scaled/(1-lmbda_scaled))/(2*2*5)+.5
            y_graphmad = y_hat[k_mixup]*(1-lmbda_logit) + (1/num_classes)*np.ones(num_classes)*lmbda_logit

        W_graphmad = Graphon(theta=theta_graphmad)
        Ws_graphmad.append([Graphon(theta=theta_centroids_mixup[k]) for k in range(len(theta_centroids_mixup))])
        
        sample_graphs = []
        for i in range(num_sample):
            A_sampled,_ = W_graphmad.sample_graph(N=N_mixup,z=z_mixup,sorted=sorted)
            if np.sum(A_sampled)==0:
                continue
            A_sampled = A_sampled[A_sampled.sum(axis=1) != 0]
            A_sampled = A_sampled[:, A_sampled.sum(axis=0) != 0]
            
            A_new = torch.from_numpy(A_sampled)
            edge_index,_ = dense_to_sparse(A_new)

            num_nodes = int(torch.max(edge_index)) + 1

            pyg_graph = Data()
            pyg_graph.y = torch.Tensor(y_graphmad)
            pyg_graph.edge_index = edge_index
            pyg_graph.num_nodes = num_nodes
            sample_graphs.append(pyg_graph)

        new_graph += sample_graphs
    return new_graph,Ws_graphmad

# ~INPUT~:
# Choose between diff_theta or theta_list (default to theta_list)
#   diff_theta: Different matrix of samples to assign to clusters
#   theta_list: List of samples to assign to clusters
# shrink_dist: Shrinkage distance for convex clustering
# eps_thresh: Threshold for clustering samples in CARP
# ~OUTPUT~:
# clust: List of cluster assignments
# K_est: Number of clusters
def get_clustassignment(diff_theta=None,theta_list=None,shrink_dist=shrink_dist,eps_thresh=1e-8):
    if diff_theta is None and theta_list is None:
        print('Must provide distance matrix or list of samples.')
        return
    if theta_list is not None:
        T = len(theta_list)
        len_theta = len(theta_list[0])
        diff_theta = lowtri2mat(np.array([shrink_dist(theta_list[i],theta_list[j]) for i in range(T) for j in range(i+1,T)]))
    else:
        (T,T) = diff_theta.shape

    clust = [[i] for i in range(T)]
    orig_clust = clust.copy()
    clustered = []
    num_clust = 0
    for i in range(T):
        if (i==np.array(clustered)).any():
            continue
        clust = [clust[ii] for ii in range(num_clust+1)]
        for j in range(i+1,T):
            if (j==np.array(clustered)).any():
                continue
            if diff_theta[i,j]<=eps_thresh:
                clust[num_clust] += orig_clust[j]
            else:
                clust.append(orig_clust[j])
        clustered+=clust[num_clust]
        num_clust+=1

    # Number of clusters
    if np.max(np.abs(diff_theta)) <= eps_thresh:
        K_est = 1
    else:
        K_est = len(clust)
    return clust,K_est

# ~INPUT~:
# Theta_clusterpath: CARP clusterpath of Theta, subset of graphons
# V_clusterpath: CARP cluster assignment matrices of Theta, subset of graphons
# gamma_clusterpath: CARP values of mixup parameters
# K: Minimum number of clusters to consider
# fid_dist: Fidelity distance for convex clustering
# shrink_dist: Shrinkage distance for convex clustering
# lmbda_hi: Upper bound for lmbda
# lmbda_low: Lower bound for lmbda
# num_iter: Number of binary search iterations
# eps_thresh: Threshold for clustering samples in CARP
# ~OUTPUT~:
# lmbda: Smallest lambda where number of clusters is greater than or equal to K
# K_est: Number of clusters at lmbda
def find_inf_lmbda(Theta_clusterpath,V_clusterpath,gamma_clusterpath,K=1,fid_dist=fid_dist,shrink_dist=shrink_dist,lmbda_hi=1.,lmbda_low=0.,num_iter=8,eps_thresh=1e-8):
    T = len(Theta_clusterpath[0])
    len_theta = len(Theta_clusterpath[0][0])

    Theta_interp = sp.interpolate.interp1d(gamma_clusterpath,np.array(Theta_clusterpath),axis=0)
    V_interp = np.array([[shrink_dist(V_clusterpath[i][j,:],0) for i in range(len(Theta_clusterpath))] for j in range(V_clusterpath[0].shape[0])])
    V_interp = np.array([lowtri2mat(V_interp[:,i]) for i in range(len(gamma_clusterpath))])
    V_interp = sp.interpolate.interp1d(gamma_clusterpath,V_interp,axis=0)
    
    lmbda = .5*(lmbda_hi+lmbda_low)
    bin_num = [0]
    for bin_iter in range(num_iter):
        if lmbda>gamma_clusterpath[-1]:
            theta_interp = np.array([np.mean(Theta_clusterpath,axis=0) for i in range(len(Theta_clusterpath))])
        else:
            theta_interp = Theta_interp(lmbda)
        diff_theta = V_interp(lmbda)
        clust,K_inf = get_clustassignment(diff_theta=diff_theta,eps_thresh=eps_thresh,shrink_dist=shrink_dist)

        bin_num.append(-2*int(K_inf<=K)+1)
        lmbda = lmbda_low + (lmbda_hi-lmbda_low) * (.5 + np.sum([bin_num[i]*2**(-i-1) for i in range(len(bin_num))]))

    if lmbda>gamma_clusterpath[-1]:
        theta_interp = [np.mean(Theta_clusterpath,axis=0) for i in range(len(Theta_clusterpath))]
    else:
        theta_interp = Theta_interp(lmbda)
    diff_theta = V_interp(lmbda)
    clust,K_inf = get_clustassignment(diff_theta=diff_theta,eps_thresh=eps_thresh,shrink_dist=shrink_dist)
    return lmbda,K_inf

# TO DO: Add gamma_max, default as None, and break loop if above gamma_max
# TO DO: Change U and V updates based on fid_dist. Right now, default is squared L2-norm
# ~INPUT~:
# theta_list: List of graphon vector approximations
# fid_dist: Fidelity distance for convex clustering
# shrink_dist: Shrinkage distance for convex clustering
# weights: Convex clustering weights
# eps_thresh: Threshold for clustering samples in CARP
# max_iter: Maximum number of CARP iterations
# burn_in: Iteration to start saving iterations
# keep: Skip this many between saving iterations
# center: Whether to center data to 0
# scale: Whether to scale data to 1
# ~OUTPUT~:
# U_fus: List of cluster centroids along clusterpath, length is number of mixup parameters
# V_fus: List of assignment matrices along clusterpath, length is number of mixup parameters
# clus_fus: List of cluster assignments along clusterpath, length is number of mixup parameters
# K_fus: List of number of clusters along clusterpath, length is number of mixup parameters
# gamma_fus: List of mixup parameter along clusterpath
def clusterpath_carp(theta_list,fid_dist=fid_dist,shrink_dist=shrink_dist,weights=None,eps_thresh=1e-8,
                     epsilon = 0.000001,t=1.05,gamma_max=1,
                     max_iter = 100000,burn_in = 50,keep = 10,
                     center=False,scale=False
                    ):
    T = len(theta_list)
    len_theta = len(theta_list[0])
    Theta = np.array(theta_list)

    if weights is None:
        weights = np.ones(int(T*(T-1)/2))
    
    # Precompute D
    D = np.zeros((int(T*(T-1)/2),T))
    upp_tri_ind = np.where(np.triu(np.ones((T,T)))-np.eye(T))
    for l in range(D.shape[0]):
        D[l, upp_tri_ind[0][l]] = 1
        D[l, upp_tri_ind[1][l]] = -1

    # Precompute L
    L = sp.linalg.cholesky( np.eye(T)+D.T@D, lower=True )
    LL = np.linalg.inv(L.T)@np.linalg.inv(L)

    # Center and scale if needed
    Theta = (Theta-np.mean(Theta)*int(center))
    if scale and np.std(Theta):
        Theta /= np.std(Theta)
    
    # Initialize CARP variables
    iter = 0
    U = Theta.copy()
    V = D@Theta
    V_last = V.copy()
    Z = D@Theta
    clust = [[i] for i in range(T)]
    gamma = epsilon
    K_est = T

    # Save first iteration of CARP variables
    U_fus = [U]
    V_fus = [V]
    Z_fus = [Z]
    clust_fus = [clust]
    gamma_fus = [gamma]
    K_fus = [K_est]

    # Compute number of fusions
    v_norm = np.array([shrink_dist(V[i,:],0) for i in range(V.shape[0])])
    v_zeros = v_norm.copy()
    v_zeros[v_norm >  eps_thresh] = 1
    v_zeros[v_norm <= eps_thresh] = 0

    while iter<max_iter and np.sum(v_zeros)>0:
        # Save last fusions
        v_last = v_zeros.copy()
        
        # ADMM step
        # U-update:
        U = LL@(Theta + D.T@(V-Z))
        DU = D@U

        # V-update:
        V = V_last.copy()
        DUZ = DU + Z
        for i_Vrow in range(V.shape[0]):
            V[i_Vrow,:] = soft_thresh(DUZ[i_Vrow,:],gamma*weights[i_Vrow])
        V_last = V.copy()

        # Z-update:
        Z += DU - V

        # Compute number of fusions
        v_norm = np.array([shrink_dist(V[i,:],0) for i in range(V.shape[0])])
        v_zeros = v_norm.copy()
        v_zeros[v_norm >  eps_thresh] = 1
        v_zeros[v_norm <= eps_thresh] = 0

        # If fusion occurred or iterations are past burn-in, save current values
        if (np.abs(np.sum(v_zeros)-np.sum(v_last))>0) or (iter%keep==0 and iter>burn_in):
            U_fus.append(U)
            V_fus.append(V)
            Z_fus.append(Z)
            gamma_fus.append(gamma)

            diff_theta = lowtri2mat(v_norm)
            clust,K_est = get_clustassignment(diff_theta=diff_theta,eps_thresh=eps_thresh,shrink_dist=shrink_dist)
            
            clust_fus.append(clust)
            K_fus.append(K_est)
            
        iter += 1

        # Increase fusion parameter
        if (iter>=burn_in):
            gamma *= t

    gamma_fus = np.array(gamma_fus)
    gamma_fus[0] = 0
    gamma_fus = list((gamma_fus)/np.max(gamma_fus))

    return U_fus,V_fus,clust_fus,K_fus,gamma_fus

def robustclusterpath_carp(theta_list,fid_dist=fid_dist,shrink_dist=shrink_dist,weights=None,eps_thresh=1e-8,
                           epsilon = 0.000001,t=1.05,gamma_max=1,beta=.1,
                           max_iter = 100000,burn_in = 50,keep = 10,
                           center=False,scale=False
                          ):
    T = len(theta_list)
    len_theta = len(theta_list[0])
    Theta = np.array(theta_list)

    if weights is None:
        weights = np.ones(int(T*(T-1)/2))
    
    # Precompute D
    D = np.zeros((int(T*(T-1)/2),T))
    upp_tri_ind = np.where(np.triu(np.ones((T,T)))-np.eye(T))
    for l in range(D.shape[0]):
        D[l, upp_tri_ind[0][l]] = 1
        D[l, upp_tri_ind[1][l]] = -1

    # Precompute L
    L = sp.linalg.cholesky( np.eye(T)+D.T@D, lower=True )
    LL = np.linalg.inv(L.T)@np.linalg.inv(L)

    # Center and scale if needed
    Theta = (Theta-np.mean(Theta)*int(center))
    if scale and np.std(Theta):
        Theta /= np.std(Theta)
    
    # Initialize CARP variables
    iter = 0
    P = Theta.copy()
    Q = np.zeros_like(P)
    V = D@Theta
    V_last = V.copy()
    Z = D@Theta
    clust = [[i] for i in range(T)]
    gamma = epsilon
    K_est = T

    # Save first iteration of CARP variables
    P_fus = [P]
    Q_fus = [Q]
    V_fus = [V]
    Z_fus = [Z]
    clust_fus = [clust]
    gamma_fus = [gamma]
    K_fus = [K_est]

    # Compute number of fusions
    v_norm = np.array([shrink_dist(V[i,:],0) for i in range(V.shape[0])])
    # v_norm = np.array([shrink_dist(U[i,:],U[j,:]) for i in range(U.shape[0]) for j in range(i+1,U.shape[0])])
    v_zeros = v_norm.copy()
    v_zeros[v_norm >  eps_thresh] = 1
    v_zeros[v_norm <= eps_thresh] = 0

    while iter<max_iter and np.sum(v_zeros)>0:
        # Save last fusions
        v_last = v_zeros.copy()
        
        # ADMM step
        # P-update:
        P = LL@(Theta - Q + D.T@(V-Z))
        DP = D@P

        # Q-update:
        Q = np.maximum(0,1-beta/np.sum(Theta-P,axis=1).reshape(-1,1))*(Theta-P)

        # V-update:
        V = V_last.copy()
        DPZ = DP + Z
        for i_Vrow in range(V.shape[0]):
            V[i_Vrow,:] = soft_thresh(DPZ[i_Vrow,:],gamma*weights[i_Vrow])
        V_last = V.copy()

        # Z-update:
        Z += DP - V
        
        # Compute number of fusions
        v_norm = np.array([shrink_dist(V[i,:],0) for i in range(V.shape[0])])
        # v_norm = np.array([shrink_dist(U[i,:],U[j,:]) for i in range(U.shape[0]) for j in range(i+1,U.shape[0])])
        v_zeros = v_norm.copy()
        v_zeros[v_norm >  eps_thresh] = 1
        v_zeros[v_norm <= eps_thresh] = 0

        # If fusion occurred or iterations are past burn-in, save current values
        if (np.abs(np.sum(v_zeros)-np.sum(v_last))>0) or (iter%keep==0 and iter>burn_in):
            P_fus.append(P)
            Q_fus.append(Q)
            V_fus.append(V)
            Z_fus.append(Z)
            gamma_fus.append(gamma)

            diff_theta = lowtri2mat(v_norm)
            clust,K_est = get_clustassignment(diff_theta=diff_theta,eps_thresh=eps_thresh,shrink_dist=shrink_dist)
            
            clust_fus.append(clust)
            K_fus.append(K_est)
            
        iter += 1

        # Increase fusion parameter
        if (iter>=burn_in):
            gamma *= t

    gamma_fus = np.array(gamma_fus)
    gamma_fus[0] = 0
    gamma_fus = list((gamma_fus)/np.max(gamma_fus))

    return P_fus,Q_fus,V_fus,clust_fus,K_fus,gamma_fus

# ~INPUT~:
# graph_desciptors: List of graph descriptors
# phi: Gaussian kernel parameter
# fid_dist: Fidelity distance for convex clustering
# ~OUTPUT~:
# weights1: Uniform weights
# weights2: Gaussian kernel weights
# weights3: Sparse Gaussian kernel weights
def compute_weights(graph_descriptors,phi=None,fid_dist=fid_dist):
    T = len(graph_descriptors)

    weights1 = np.ones(int(T*(T-1)/2))

    if phi is not None:
        weights2 = np.array([np.exp(-phi*( fid_dist(graph_descriptors[i],graph_descriptors[j]) )) for i in range(T) for j in range(i+1,T)])
    else:
        phi_range = np.linspace(0,10,10)
        max_var = 0
        idx_phi = 0
        for i_phi in range(len(phi_range)):
            weights2_all = np.array([np.exp(-phi_range[i_phi]*( fid_dist(graph_descriptors[i],graph_descriptors[j]) )) for i in range(T) for j in range(i+1,T)])
            curr_var = np.var(weights2_all)
            if curr_var>max_var:
                max_var = curr_var
                idx_phi = i_phi
        weights2 = np.array([np.exp(-phi_range[idx_phi]*( fid_dist(graph_descriptors[i],graph_descriptors[j]) )) for i in range(T) for j in range(i+1,T)])

    weights3 = weights2.copy()
    Weights = lowtri2mat(weights3)
    Lapl_Weights = np.diag(np.sum(Weights,axis=0)) - Weights
    eval_Weights,evev_Weights = np.linalg.eigh(Lapl_Weights)
    Weights_curr = Weights.copy()
    Weights_last = Weights.copy()
    while (np.sum(np.abs(eval_Weights)<1e-8)<=1):
        Weights_last = Weights_curr.copy()
        Weights_curr[Weights_curr==np.min(Weights_curr[Weights_curr>0])]=0
        Lapl_Weights = np.diag(np.sum(Weights_curr,axis=0)) - Weights_curr
        eval_Weights,evev_Weights = np.linalg.eigh(Lapl_Weights)
    Weights = Weights_last.copy()
    weights3 = mat2lowtri(Weights)

    return weights1,weights2,weights3

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--log_screen', type=str, default='False')
    parser.add_argument('--debug', type=str, default='True')

    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--num_trials', type=int, default=8)

    parser.add_argument('--data_path', type=str, default='/tmp/')
    parser.add_argument('--dataset', type=str, default='Synthetic')

    parser.add_argument('--model', type=str, default='GIN')
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--num_hidden', type=int, default=32)

    parser.add_argument('--nomixup', type=str, default='True')
    parser.add_argument('--gmixup', type=str, default='False')
    parser.add_argument('--graphmad', type=str, default='False')
    parser.add_argument('--mixup_func', type=str, default='linear')
    parser.add_argument('--aug_ratio', type=int, default=.15)
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--lmbda_bounds', type=str, default='[0.,1.]')
    parser.add_argument('--nKnots', type=int, default=10)

    parser.add_argument('--mixup_batch_size', type=int, default=20)
    parser.add_argument('--weight_type', type=str, default='2')
    parser.add_argument('--epsilon', type=float, default=.000001)
    parser.add_argument('--max_carp_iter', type=int, default=10000)
    parser.add_argument('--burn_in', type=int, default=50)
    parser.add_argument('--keep', type=int, default=10)
    parser.add_argument('--t', type=float, default=1.02)
    parser.add_argument('--eps_thresh', type=float, default=0.0)

    parser.add_argument('--sorted', type=str, default='True')
    parser.add_argument('--aligned', type=str, default='True')
    parser.add_argument('--sas_only', type=str, default='True')
    # parser.add_argument('--', type=, default=)

    args = parser.parse_args()
    experiment_name = args.experiment_name
    log_screen = eval(args.log_screen)
    DEBUG = eval(args.debug)

    seed = args.seed
    num_trials = args.num_trials

    data_path = args.data_path
    dataset_name = args.dataset

    model_name = args.model
    num_epochs = args.epoch
    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr

    # Flags for all mixup types to evaluate
    nomixup = eval(args.nomixup)
    gmixup = eval(args.gmixup)
    graphmad_mixup = eval(args.graphmad)
    mixup_func = args.mixup_func
    aug_ratio = args.aug_ratio
    aug_num = args.aug_num
    lmbda_bounds = eval(args.lmbda_bounds)
    nKnots = args.nKnots

    mixup_batch_size = args.mixup_batch_size
    weight_type = args.weight_type
    epsilon = args.epsilon
    max_carp_iter = args.max_carp_iter
    burn_in = args.burn_in
    keep = args.keep
    t = args.t
    eps_thresh = args.eps_thresh
    if weight_type=='3':
        t=1.05

    sorted = eval(args.sorted)
    aligned = eval(args.aligned)
    sas_only = eval(args.sas_only)

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.info('Parser.prog: {}'.format(parser.prog))
    logger.info("Args: {}".format(args))

    path = os.getcwd()
    folder_path = path+'/'+experiment_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DEBUG:
        status_file_name = path+'/'+experiment_name + '/status.txt'
        exp_status_file = open(status_file_name,'w')
        exp_status_file.writelines('STATUS FILE'+'\n')
        exp_status_file.close()
    else:
        status_file_name = None

    update_status(f'Dataset: {dataset_name}',status_file_name,logger)
    update_status(f'Model: {model_name}',status_file_name,logger)
    update_status(f'Epochs: {num_epochs}, no. hidden: {num_hidden}, batch size: {batch_size}, learning rate: {learning_rate}',status_file_name,logger)
    if gmixup:
        update_status(f'Mixup: G-Mixup',status_file_name,logger)
    if graphmad_mixup:
        update_status(f'Mixup: GraphMAD',status_file_name,logger)
    if gmixup or graphmad_mixup:
        update_status(f'Mixup function: {mixup_func}',status_file_name,logger)
        update_status(f'Aug. ratio: {aug_ratio}, aug. number: {aug_num}',status_file_name,logger)
        update_status(f'Lambda bounds: [{lmbda_bounds[0]}, {lmbda_bounds[1]}]',status_file_name,logger)
    if graphmad_mixup or (mixup_func=='clusterpath'):
        update_status(f'Mixup batch size: {mixup_batch_size}',status_file_name,logger)
    update_status(f'Running device: {device}',status_file_name,logger)

    if nomixup:
        test_acc_nomixup_total = np.zeros((num_trials,num_epochs-1))
        val_acc_nomixup_total = np.zeros((num_trials,num_epochs-1))
        test_loss_nomixup_total = np.zeros((num_trials,num_epochs-1))
        val_loss_nomixup_total = np.zeros((num_trials,num_epochs-1))
        train_loss_nomixup_total = np.zeros((num_trials,num_epochs-1))
    if gmixup:
        test_acc_gmixup_total = np.zeros((num_trials,num_epochs-1))
        val_acc_gmixup_total = np.zeros((num_trials,num_epochs-1))
        test_loss_gmixup_total = np.zeros((num_trials,num_epochs-1))
        val_loss_gmixup_total = np.zeros((num_trials,num_epochs-1))
        train_loss_gmixup_total = np.zeros((num_trials,num_epochs-1))
    if graphmad_mixup:
        test_acc_graphmad_total = np.zeros((num_trials,num_epochs-1))
        val_acc_graphmad_total = np.zeros((num_trials,num_epochs-1))
        test_loss_graphmad_total = np.zeros((num_trials,num_epochs-1))
        val_loss_graphmad_total = np.zeros((num_trials,num_epochs-1))
        train_loss_graphmad_total = np.zeros((num_trials,num_epochs-1))

    torch.manual_seed(seed)
    for trial_iter in range(num_trials):
        update_status(f'Trial {trial_iter+1}',status_file_name,logger)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Set up dataset
        dataset = prepare_dataset(dataset_name,data_path)
        if 'edge_attr' in dataset[0]:
            [dataset[i].pop('edge_attr') for i in range(len(dataset))]
        num_classes = len(dataset[0].y)
        K = num_classes
        random.shuffle( dataset )
        train_nums = int(len(dataset) * 0.7)
        train_val_nums = int(len(dataset) * 0.8)
        num_all_nodes = [dataset[i].num_nodes for i in range(len(dataset))]
        median_num_nodes = int(np.median(num_all_nodes))
        num_sample = int( train_nums * aug_ratio / aug_num )
        if num_classes>5:
            mixup_batch_size = 10
        elif num_classes>10:
            mixup_batch_size = 8

        # Mixup parameters
        if gmixup or graphmad_mixup:
            lmbda_list = select_mixup_parameter(lmbda_bounds,aug_num,aug_ratio)
            update_status(f'Mixup parameters: {np.round(lmbda_list,2)}',status_file_name,logger)

        # Clusterpath (need the clusterpath for labels or GraphMAD mixup)
        lmbda_clusterpath=None
        Theta_carp=None
        V_carp=None
        gamma_carp=None
        y_list=None
        if ((mixup_func=='clusterpath') or graphmad_mixup):
            update_status('Estimate graph descriptors...',status_file_name,logger)

            # Estimate graph descriptors for clustering
            Ws_estimate,y_list = graphs_to_graphons(dataset[:train_nums],num_classes=num_classes,mixup_batch_size=mixup_batch_size,nKnots=nKnots,sorted=sorted,sas_only=sas_only)
            theta_list = [np.maximum(Ws_estimate[i].theta,0) for i in range(len(Ws_estimate))]

            # Compute weights for clustering
            if weight_type!='1' and weight_type!='2' and weight_type!='3':
                weightmat = (1/5/num_classes)*np.ones((len(y_list),len(y_list)))
                for k in range(num_classes):
                    weightmat[np.where(np.array(y_list)==k)[0].reshape(-1,1),np.where(np.array(y_list)==k)[0].reshape(1,-1)]=1
                weights = mat2lowtri(weightmat)
            else:
                weights1,weights2,weights3 = compute_weights(theta_list,fid_dist=fid_dist)
                if weight_type=='1':
                    weights=weights1.copy()
                elif weight_type=='2':
                    weights=weights2.copy()
                elif weight_type=='3':
                    weights=weights3.copy()

            class_thetas = [[] for class_idx in range(num_classes)]
            for class_idx in range(num_classes):
                class_thetas[class_idx] = [theta_list[i_class] for i_class in range(class_idx*mixup_batch_size,(class_idx+1)*mixup_batch_size)]
            Westimate_diff_mean = np.mean(np.linalg.norm([theta_list[i]-theta_list[j] for i in range(len(theta_list)) for j in range(i+1,len(theta_list))],2,axis=1))
            Westimate_diff_std  = np.std(np.linalg.norm([theta_list[i]-theta_list[j] for i in range(len(theta_list)) for j in range(i+1,len(theta_list))],2,axis=1))
            Westimate_class_diff_mean = []
            Westimate_class_diff_std = []
            for class_idx in range(num_classes):
                Westimate_class_diff_mean.append(np.mean(np.linalg.norm([class_thetas[class_idx][i]-class_thetas[class_idx][j] for i in range(mixup_batch_size) for j in range(i+1,mixup_batch_size)],2,axis=1)))
                Westimate_class_diff_std.append(np.std(np.linalg.norm([class_thetas[class_idx][i]-class_thetas[class_idx][j] for i in range(mixup_batch_size) for j in range(i+1,mixup_batch_size)],2,axis=1)))
            update_status(f'Avg diff. between all class graphons of {int(len(theta_list)*(len(theta_list)-1)/2)} pairs: {Westimate_diff_mean:.2f} (std. {Westimate_diff_std:.2f})',status_file_name,logger)
            for class_idx in range(num_classes):
                update_status(f'Avg diff. between class {class_idx+1} graphons of {int(mixup_batch_size*(mixup_batch_size-1)/2)} pairs: {Westimate_class_diff_mean[class_idx]:.2f} (std. {Westimate_class_diff_std[class_idx]:.2f})',status_file_name,logger)

            Theta_carp,V_carp,clust_carp,K_carp,gamma_carp = clusterpath_carp(theta_list,fid_dist=fid_dist,shrink_dist=shrink_dist,weights=weights,
                                                                        epsilon=epsilon,max_iter=max_carp_iter,burn_in = burn_in,keep=keep,
                                                                        t=t,eps_thresh=eps_thresh
                                                                        )

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        # G-Mixup
        if gmixup:
            update_status(f'G-mixup...',status_file_name,logger)
            
            # Later: selected aligned, sorted, and z_mixup automatically if synthetic or real dataset
            newgraph_gmixup,W_gmixup = mixup_gmixup(dataset=dataset,train_nums=train_nums,mixup_func=mixup_func,lmbda_list=lmbda_list,
                                                    num_sample=num_sample,nKnots=nKnots,
                                                    y_list=y_list,Theta_clusterpath=Theta_carp,V_clusterpath=V_carp,gamma_clusterpath=gamma_carp,
                                                    aligned=aligned,sas_only=sas_only,sorted=sorted,
                                                    z_init=None,z_mixup=None)

            Wgmixup_diff_mean = np.mean(np.linalg.norm([W_gmixup[i].theta-W_gmixup[j].theta for i in range(len(W_gmixup)) for j in range(i+1,len(W_gmixup))],2,axis=1))
            Wgmixup_diff_std  = np.std(np.linalg.norm([W_gmixup[i].theta-W_gmixup[j].theta for i in range(len(W_gmixup)) for j in range(i+1,len(W_gmixup))],2,axis=1))
            update_status(f'Avg diff. between all class graphons of {int(num_classes*(num_classes-1)/2)} pairs: {Wgmixup_diff_mean:.2f} (std. {Wgmixup_diff_std:.2f})',status_file_name,logger)

            update_status(f'{len(newgraph_gmixup)} = {num_sample}*{aug_num} new samples',status_file_name,logger)
            num_all_nodes = [dataset[i].num_nodes for i in range(len(dataset))]
            median_num_nodes = int(np.median(num_all_nodes))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # GraphMAD mixup
        if graphmad_mixup:
            update_status('GraphMAD...',status_file_name,logger)
            update_status('Sample graphs...',status_file_name,logger)

            # Later: selected sorted, z_mixup, ks_mixup automatically if synthetic or real dataset
            newgraph_graphmad,Ws_graphmad = mixup_graphmad(
                                                            lmbda_list=lmbda_list,y_list=y_list,mixup_func=mixup_func,
                                                            Theta_clusterpath=Theta_carp,V_clusterpath=V_carp,gamma_clusterpath=gamma_carp,
                                                            fid_dist=fid_dist,shrink_dist=shrink_dist,
                                                            num_sample=num_sample,eps_thresh=eps_thresh,sorted=sorted,z_mixup=None
                                                           )

            update_status(f'{len(newgraph_graphmad)} = {num_sample}*{aug_num} new samples',status_file_name,logger)
            num_all_nodes = [dataset[i].num_nodes for i in range(len(dataset))]
            median_num_nodes = int(np.median(num_all_nodes))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # If performing G-Mixup, train model
        if gmixup:
            dataset_gmixup = newgraph_gmixup + dataset
            train_nums = train_nums + len( newgraph_gmixup )
            train_val_nums = train_val_nums + len( newgraph_gmixup )
        
            max_degree = 0
            degs = []
            for graph in dataset_gmixup:
                degs += [degree(graph.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())
                graph.num_nodes = int(torch.max(graph.edge_index)) + 1
            if max_degree < 2000:
                for graph in dataset_gmixup:
                    degs = degree(graph.edge_index[0], dtype=torch.long)
                    graph.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
            else:
                deg = torch.cat(degs,dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                for graph in dataset_gmixup:
                    degs = degree(graph.edge_index[0], dtype=torch.long)
                    graph.x = ((degs-mean)/std).view(-1,1)

            num_features = dataset_gmixup[0].x.shape[1]
            num_classes = dataset_gmixup[0].y.shape[0]

            train_dataset = dataset_gmixup[:train_nums]
            random.shuffle(train_dataset)
            val_dataset = dataset_gmixup[train_nums:train_val_nums]
            test_dataset = dataset_gmixup[train_val_nums:]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Model initialized as GIN
            if model_name == "GIN":
                model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)
            else:
                # print(f"No model.")
                logging.info(f"No model.")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
            scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

            # Get train, validation, and test accuracies
            update_status(f'Training model...',status_file_name,logger)
            for epoch in range(1, num_epochs):
                model,train_acc,train_loss = train(model,train_loader)
                val_acc,val_loss   = test(model,val_loader)
                test_acc,test_loss = test(model,test_loader)
                scheduler.step()
                update_status(f'Epoch: {epoch:03d}, '+
                            f'Train Loss: {train_loss:.6f}, '+
                            f'Val Loss: {val_loss:.6f}, '+
                            f'Test Loss: {test_loss:.6f}, '+
                            f'Val Acc: {val_acc: .6f}, '+
                            f'Test Acc: {test_acc: .6f}',
                            status_file_name,logger)
                
                test_acc_gmixup_total[trial_iter,epoch-1] = test_acc
                val_acc_gmixup_total[trial_iter,epoch-1] = val_acc
                test_loss_gmixup_total[trial_iter,epoch-1] = test_loss
                val_loss_gmixup_total[trial_iter,epoch-1] = val_loss
                train_loss_gmixup_total[trial_iter,epoch-1] = train_loss
            
            save_results(row=test_acc_gmixup_total[trial_iter,:],  file_name='test_acc_gmixup.txt',  path=path+'/'+experiment_name)
            save_results(row=val_acc_gmixup_total[trial_iter,:],   file_name='val_acc_gmixup.txt',   path=path+'/'+experiment_name)
            save_results(row=train_loss_gmixup_total[trial_iter,:],file_name='train_loss_gmixup.txt',path=path+'/'+experiment_name)
            save_results(row=test_loss_gmixup_total[trial_iter,:], file_name='test_loss_gmixup.txt', path=path+'/'+experiment_name)
            save_results(row=val_loss_gmixup_total[trial_iter,:],  file_name='val_loss_gmixup.txt',  path=path+'/'+experiment_name)

        if graphmad_mixup:
            dataset_graphmad = newgraph_graphmad + dataset
            train_nums = train_nums + len( newgraph_graphmad )
            train_val_nums = train_val_nums + len( newgraph_graphmad )
        
            max_degree = 0
            degs = []
            for graph in dataset_graphmad:
                degs += [degree(graph.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())
                graph.num_nodes = int(torch.max(graph.edge_index)) + 1
            if max_degree < 2000:
                for graph in dataset_graphmad:
                    degs = degree(graph.edge_index[0], dtype=torch.long)
                    graph.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
            else:
                deg = torch.cat(degs,dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                for graph in dataset_graphmad:
                    degs = degree(graph.edge_index[0], dtype=torch.long)
                    graph.x = ((degs-mean)/std).view(-1,1)

            num_features = dataset_graphmad[0].x.shape[1]
            num_classes = dataset_graphmad[0].y.shape[0]

            train_dataset = dataset_graphmad[:train_nums]
            random.shuffle(train_dataset)
            val_dataset = dataset_graphmad[train_nums:train_val_nums]
            test_dataset = dataset_graphmad[train_val_nums:]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            if model_name == "GIN":
                model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)
            else:
                # print(f"No model.")
                logging.info(f"No model.")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
            scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

            update_status(f'Training model...',status_file_name,logger)
            for epoch in range(1, num_epochs):
                model,train_acc,train_loss = train(model,train_loader)
                val_acc,val_loss   = test(model,val_loader)
                test_acc,test_loss = test(model,test_loader)
                scheduler.step()
                update_status(f'Epoch: {epoch:03d}, '+
                            f'Train Loss: {train_loss:.6f}, '+
                            f'Val Loss: {val_loss:.6f}, '+
                            f'Test Loss: {test_loss:.6f}, '+
                            f'Val Acc: {val_acc: .6f}, '+
                            f'Test Acc: {test_acc: .6f}',
                            status_file_name,logger)
                
                test_acc_graphmad_total[trial_iter,epoch-1] = test_acc
                val_acc_graphmad_total[trial_iter,epoch-1] = val_acc
                test_loss_graphmad_total[trial_iter,epoch-1] = test_loss
                val_loss_graphmad_total[trial_iter,epoch-1] = val_loss
                train_loss_graphmad_total[trial_iter,epoch-1] = train_loss
            
            save_results(row=test_acc_graphmad_total[trial_iter,:],  file_name='test_acc_graphmad.txt',  path=path+'/'+experiment_name)
            save_results(row=val_acc_graphmad_total[trial_iter,:],   file_name='val_acc_graphmad.txt',   path=path+'/'+experiment_name)
            save_results(row=train_loss_graphmad_total[trial_iter,:],file_name='train_loss_graphmad.txt',path=path+'/'+experiment_name)
            save_results(row=test_loss_graphmad_total[trial_iter,:], file_name='test_loss_graphmad.txt', path=path+'/'+experiment_name)
            save_results(row=val_loss_graphmad_total[trial_iter,:],  file_name='val_loss_graphmad.txt',  path=path+'/'+experiment_name)

        if nomixup:
            train_nums = int(len(dataset) * 0.7)
            train_val_nums = int(len(dataset) * 0.8)
        
            max_degree = 0
            degs = []
            for graph in dataset:
                degs += [degree(graph.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())
                graph.num_nodes = int(torch.max(graph.edge_index)) + 1
            if max_degree < 2000:
                for graph in dataset:
                    degs = degree(graph.edge_index[0], dtype=torch.long)
                    graph.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
            else:
                deg = torch.cat(degs,dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                for graph in dataset:
                    degs = degree(graph.edge_index[0], dtype=torch.long)
                    graph.x = ((degs-mean)/std).view(-1,1)

            num_features = dataset[0].x.shape[1]
            num_classes = dataset[0].y.shape[0]

            train_dataset = dataset[:train_nums]
            random.shuffle(train_dataset)
            val_dataset = dataset[train_nums:train_val_nums]
            test_dataset = dataset[train_val_nums:]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            if model_name == "GIN":
                model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)
            else:
                # print(f"No model.")
                logging.info(f"No model.")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
            scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

            update_status(f'Training model...',status_file_name,logger)
            for epoch in range(1, num_epochs):
                model,train_acc,train_loss = train(model,train_loader)
                val_acc,val_loss   = test(model,val_loader)
                test_acc,test_loss = test(model,test_loader)
                scheduler.step()
                update_status(f'Epoch: {epoch:03d}, '+
                            f'Train Loss: {train_loss:.6f}, '+
                            f'Val Loss: {val_loss:.6f}, '+
                            f'Test Loss: {test_loss:.6f}, '+
                            f'Val Acc: {val_acc: .6f}, '+
                            f'Test Acc: {test_acc: .6f}',
                            status_file_name,logger)
                
                test_acc_nomixup_total[trial_iter,epoch-1] = test_acc
                val_acc_nomixup_total[trial_iter,epoch-1] = val_acc
                test_loss_nomixup_total[trial_iter,epoch-1] = test_loss
                val_loss_nomixup_total[trial_iter,epoch-1] = val_loss
                train_loss_nomixup_total[trial_iter,epoch-1] = train_loss
            
            save_results(row=test_acc_nomixup_total[trial_iter,:],  file_name='test_acc_nomixup.txt',  path=path+'/'+experiment_name)
            save_results(row=val_acc_nomixup_total[trial_iter,:],   file_name='val_acc_nomixup.txt',   path=path+'/'+experiment_name)
            save_results(row=train_loss_nomixup_total[trial_iter,:],file_name='train_loss_nomixup.txt',path=path+'/'+experiment_name)
            save_results(row=test_loss_nomixup_total[trial_iter,:], file_name='test_loss_nomixup.txt', path=path+'/'+experiment_name)
            save_results(row=val_loss_nomixup_total[trial_iter,:],  file_name='val_loss_nomixup.txt',  path=path+'/'+experiment_name)
