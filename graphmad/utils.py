# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Functions for CARP clusterpath.
'''
import numpy as np
import scipy.interpolate as interpolate
import scipy as sp
import cvxopt
import scipy.optimize as optimize
from copy import copy
import math
from typing import List, Tuple
import cvxpy as cp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,Dataset
from torch_geometric.utils import degree,to_dense_adj,dense_to_sparse

from graphon import *

#---------------------------------------------------------------
class SyntheticGraphDataset(Dataset):
    def __init__(self,graphs,labels,feats=None):
        self.y = torch.Tensor(labels).to(torch.int64)
        graph_tensor = [torch.from_numpy(graphs[i]).float() for i in range(len(graphs))]
        edge_indices = [dense_to_sparse(graph_tensor[i])[0] for i in range(len(graphs))]
        # graph_tensor = torch.from_numpy(np.array(graphs)).float()
        # edge_indices = [dense_to_sparse(graph_tensor[i,:,:])[0] for i in range(len(graphs))]
        num_nodes = [int(torch.max(edge_indices[0]))+1 for i in range(len(graphs))]
        self.edge_index = edge_indices
        self.num_nodes  = num_nodes

        max_degree = 0
        degs = []
        for edge_ind in self.edge_index:
            degs += [degree(edge_ind[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        degs = []
        feats = []
        if max_degree < 2000:
            for edge_ind in self.edge_index:
                degs = degree(edge_ind[0], dtype=torch.long)
                feats.append(F.one_hot(degs, num_classes=max_degree+1).to(torch.float))
        else:
            deg = torch.cat(degs,dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for edge_ind in self.edge_index:
                degs = degree(edge_ind[0], dtype=torch.long)
                feats.append(((degs-mean)/std).view(-1,1))
        self.x = feats
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # sample = {'edge_index':self.edge_index[i], 'y':self.y[i], 'x':self.x[i]}
        sample = Data()
        sample.edge_index = self.edge_index[idx]
        sample.y = self.y[idx]
        sample.x = self.x[idx]
        return sample

def generate_synthetic_dataset(len_dataset=2000,num_classes=2,N=100):
    if num_classes!=2:
        print('Only implemented for binary classes.')
    W1 = graphon_from_example(0)
    W2 = graphon_from_example(3)

    num_per_class = int(len_dataset/num_classes)

    A1 = [W1.sample_graph(N,sorted=True)[0] for i in range(num_per_class)]
    A2 = [W2.sample_graph(N,sorted=True)[0] for i in range(num_per_class)]

    A = A1+A2
    y = [0 for i in range(num_per_class)]+[1 for i in range(num_per_class)]

    dataset = SyntheticGraphDataset(A,y)
    return dataset

#---------------------------------------------------------------
# Given a list of data points, two functions for fidelity and difference terms, compute clusterpath
def fid_dist(x,y,cvxpy=False):
    if cvxpy:
        return cp.norm(x-y,2)**2
    else:
        return np.linalg.norm(x-y,2)**2
def shrink_dist(x,y,cvxpy=False):
    if cvxpy:
        return cp.norm(x-y,1)
    else:
        return np.linalg.norm(x-y,1)

def extended_clusterpath(theta_cc,K_cc,clust_cc,lmbda_range,K):
    if np.sum(K_cc==K)>0:
        lmbda_at_K = lmbda_range[K_cc==K][0]
        lmbda_at_K_idx = np.where(K_cc==K)[0][0]
        K_est = K
    else:
        lmbda_at_K = lmbda_range[K_cc>=K][-1]
        lmbda_at_K_idx = np.where(K_cc>=K)[0][-1]
        K_est = K_cc[lmbda_at_K_idx]

    num_lmbda = len(lmbda_range)
    len_theta = len(theta_cc[0][0])

    if K_est is None:
        K_est = K

    theta_cc_centroids = []
    lmbda_traverse = []
    for lmbda_idx in range(num_lmbda):
        if K_cc[lmbda_idx]<K:
            break
        theta_cc_centroids.append([])
        for k in range(K_est):
            theta_cc_centroids[lmbda_idx].append(np.zeros(len_theta))
            for j in clust_cc[lmbda_at_K_idx][k]:
                theta_cc_centroids[lmbda_idx][k] += theta_cc[lmbda_idx][j]
            theta_cc_centroids[lmbda_idx][k] /= len(clust_cc[lmbda_at_K_idx][k])
        lmbda_traverse.append(lmbda_range[lmbda_idx])
    theta_cc_traverse = []
    for lmbda_idx in range(len(theta_cc_centroids)):
        theta_cc_traverse.append([])
        theta_cc_traverse[-1] = theta_cc_centroids[lmbda_idx][0]
    for lmbda_idx in range(len(theta_cc_centroids)):
        theta_cc_traverse.append([])
        theta_cc_traverse[-1] = theta_cc_centroids[-lmbda_idx-1][1]
    lmbda_traverse = np.array(lmbda_traverse)
    lmbda_traverse = np.concatenate(( -np.flip(lmbda_traverse),lmbda_traverse ))
    return theta_cc_traverse,lmbda_traverse

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

'''
Reference:
https://github.com/DataSlingers/clustRviz
Weylandt, Michael, Nagorski, John, and Allen, Genevera I.
"Dynamic Visualization and Fast Computation for Convex Clustering via Algorithmic Regularization."
Journal of Computational and Graphical Statistics 29.1 (2020): 87-96.
'''
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
    # v_norm = np.array([shrink_dist(U[i,:],U[j,:]) for i in range(U.shape[0]) for j in range(i+1,U.shape[0])])
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
        # v_norm = np.array([shrink_dist(U[i,:],U[j,:]) for i in range(U.shape[0]) for j in range(i+1,U.shape[0])])
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

def cvxclust_cvxpy(theta_list,fid_dist,shrink_dist,lmbda=0,weights=None,eps_thresh=1e-4):
    T = len(theta_list)
    len_theta = len(theta_list[0])

    if weights is None:
        weights = np.ones(int(T*(T-1)/2))

    D = np.zeros((int(T*(T-1)/2),T))
    upp_tri_ind = np.where(np.triu(np.ones((T,T)))-np.eye(T))
    for l in range(D.shape[0]):
        D[l, upp_tri_ind[0][l]] = 1
        D[l, upp_tri_ind[1][l]] = -1
    D = weights.reshape(-1,1)*D
    W = np.kron(D,np.eye(len_theta))

    # Concatenate theta vectors
    theta_vec = np.zeros(len_theta*T)
    for i in range(T):
        theta_vec[len_theta*i:len_theta*(i+1)] = theta_list[i]

    # Optimization problem
    u_cc = cp.Variable(len_theta*T)
    cost = 0
    cost += (1-lmbda)*fid_dist(u_cc,theta_vec,cvxpy=True)
    cost += lmbda*shrink_dist(W@u_cc,0,cvxpy=True)
    prob = cp.Problem(cp.Minimize(cost))
    try:
        result = prob.solve()
    except cp.SolverError:
        result = prob.solve(solver=cp.SCS)

    theta_current = []
    for i in range(T):
        theta_current.append(u_cc.value[len_theta*i:len_theta*(i+1)])
    theta_result = theta_current.copy()

    # Difference of theta
    diff_theta = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            diff_theta[i,j] = fid_dist(theta_result[i],theta_result[j])
    diff_theta+=diff_theta.T

    # Form clustered
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

    return theta_result,clust,K_est

#---------------------------------------------------------------
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

#---------------------------------------------------------------
# Vectorize square matrix
# @njit
def vec(X):
    (p,p) = X.shape
    x = np.empty(p**2)
    for col_idx in range(p):
        x[col_idx*p:(col_idx+1)*p] = X[:,col_idx]
    return x

#---------------------------------------------------------------
# Soft thresholding for l1-norm proximal operator
# @njit
def soft_thresh(z,lmbda):
    return np.sign(z)*np.maximum(np.abs(z)-lmbda,0)

#---------------------------------------------------------------
# Symmetric matrix to lower triangle vector
def mat2lowtri(A):
    (N,N) = A.shape
    L = int(N*(N-1)/2)

    low_tri_indices = np.where(np.triu(np.ones((N,N)))-np.eye(N))
    a = A[low_tri_indices[1],low_tri_indices[0]]
    return a

#---------------------------------------------------------------
# Lower triangle vector to symmetric matrix
def lowtri2mat(a):
    L = len(a)
    N = int(.5 + np.sqrt(2*L + .25))

    A = np.full((N,N),0,dtype=type(a[0]))
    low_tri_indices = np.where(np.triu(np.ones((N,N)))-np.eye(N))
    A[low_tri_indices[1],low_tri_indices[0]] = a
    A += A.T
    return A
