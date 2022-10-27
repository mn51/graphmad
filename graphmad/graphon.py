# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Graphon functions.
'''

import numpy as np
import scipy.interpolate as interpolate
import cvxopt
import scipy.optimize as optimize
from copy import copy
import math
from typing import List, Tuple
import cvxpy as cp

from utils import *

#---------------------------------------------------------------
'''
Reference:
https://github.com/BenjaminSischka/GraphonPy
Sischka, Benjamin and Kauermann, Goran.
"EM-based smooth graphon estimation using MCMC and spline-based approaches."
Social Networks 68. (2022): 279-295.
'''
# Step function approximation from function
def sbm_from_func(func,size):
    if np.isscalar(size):
        Ui = Uj = np.linspace(0,1,size)
        size0 = size1 = size
    else:
        Ui = np.linspace(0,1,size[0])
        Uj = np.linspace(0,1,size[1])
        (size0,size1) = size
    try:
        if func(np.array([0.3,0.7]),np.array([0.3,0.7])).ndim == 1:
            if len(Ui)<len(Uj):
                sbm = np.array([func(Ui[i],Uj) for i in range(size0)])
            else:
                sbm = np.array([func(Ui,Uj[j]) for j in range(size1)])
        else:
            sbm = func(Ui,Uj)
    except ValueError:
        print('Not appropriate graphon definition, slow from function to matrix derivation.')
        sbm = np.zeros((size0,size1))
        for i in range(size0):
            for j in range(size1):
                sbm[i,j] = func(Ui[i],Uj[j])
    return sbm

# Function approximation from B-Spline coefficients
def func_from_theta(theta,tau,order=1,nKnots=None):
    if nKnots is None:
        nKnots = int(np.sqrt(len(theta)))

    if order==0:
        prob_mat = theta.reshape((nKnots,nKnots))
        def _grad_func(x_eval,y_eval):
            vec_x = np.maximum(np.searchsorted(tau, np.array(x_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            vec_y = np.maximum(np.searchsorted(tau, np.array(y_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            return prob_mat[vec_x][:,vec_y]
    else:
        def _grad_func(x_eval,y_eval):
            x_eval_order = np.argsort(x_eval)
            y_eval_order = np.argsort(y_eval)
            func_eval_order=interpolate.bisplev(x= np.array(x_eval, ndmin=1, copy=False)[x_eval_order], y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order], tck=(tau, tau, theta, order, order), dx=0, dy=0)
            return eval('func_eval_order' + (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if len(x_eval_order) > 1 else ('[' if len(y_eval_order) > 1 else '')) + ('np.argsort(y_eval_order)]' if len(y_eval_order) > 1 else ''))
    return _grad_func

# Function approximation from step function
def func_from_sbm(sbm):
    def _step_func_aux(u,v):
        if np.isscalar(u):
            return(sbm[np.minimum(np.floor(u*sbm.shape[0]).astype(int), sbm.shape[0]-1)][np.minimum(np.floor(v*sbm.shape[1]).astype(int), sbm.shape[1]-1)])
        else:
            return(sbm[np.minimum(np.floor(u*sbm.shape[0]).astype(int), sbm.shape[0]-1)][:, np.minimum(np.floor(v*sbm.shape[1]).astype(int), sbm.shape[1]-1)])
    return _step_func_aux

# B-Spline coefficients approximation from function
# More accurate: if you create system of equations and solve for theta given nKnots**2 equations
def theta_from_func(func,nKnots=10):
    U = np.linspace(0,1,nKnots)
    try:
        if func(np.array([0.3,0.7]),np.array([0.3,0.7])).ndim == 1:
            prob_mat = np.array([func(U,U[j]) for j in range(nKnots)])
        else:
            prob_mat = func(U,U)
    except ValueError:
        print('Not appropriate graphon definition, slow from function to matrix derivation.')
        prob_mat = np.zeros((nKnots,nKnots))
        for i in range(nKnots):
            for j in range(nKnots):
                prob_mat[i,j] = func(U[i],U[j])
    theta = prob_mat.reshape(nKnots**2)
    return theta

# Ways to represent graphon: b-spline theta, function, or matrix (needs size)
class Graphon:
    def __init__(self, func=None, sbm=None, theta=None, nKnots=10, order=1, size=501):
        if func is None and sbm is None and theta is None:
            print('Error. No information about graphon.')

        if theta is not None:
            nKnots = int(np.sqrt(len(theta)))
        self.nKnots = nKnots
        self.tau = np.concatenate(([0],np.linspace(0,1,self.nKnots),[1]))
        self.order = order

        if func is not None:
            self.func = func
            self.sbm = sbm_from_func(func,size=size)
            self.theta = theta_from_func(func,nKnots=nKnots)
        elif sbm is not None:
            self.sbm = sbm
            self.func = func_from_sbm(sbm)
            self.theta = theta_from_func(self.func,nKnots=nKnots)
        else:
            self.theta = theta
            self.func = func_from_theta(theta,tau=self.tau,order=order,nKnots=nKnots)
            self.sbm = sbm_from_func(self.func,size)
    def sample_graph(self,N,z=None,sorted=False):
        if z is None:
            z = np.random.random(N)
        elif len(z)!=N:
            N = len(z)
        if sorted:
            z = np.sort(z)
        try:
            if self.func(np.array([0.3,0.7]),np.array([0.3,0.7])).ndim == 1:
                A = np.array([np.random.binomial(1,self.func(z,z[i])) for i in range(N)])
            else:
                A = np.random.binomial(1,self.func(z,z))
        except ValueError:
            A = np.zeros((N,N),dtype=int)
            for i in range(N):
                for j in range(N):
                    A[i,j] = np.random.binomial(1,self.func(z[i],z[j]))
        A[np.tril_indices(N)] = A.T[np.tril_indices(N)]
        A[np.eye(N)==1] = 0
        return A,z

def graphon_from_example(idx, size=101,nKnots=10):
    examples = {
                0: lambda u,v: 1/2*(u**2+v**2),
                1: lambda u,v: 1/2*(u+v),
                2: lambda u,v: ((1-u)*(1-v))**(1/1) * 0.8 + (u*v)**(1/1) * 0.85,
                3: lambda u,v: np.exp(-5*np.abs(u-v))
                }
    return Graphon(func=examples[idx],size=size,nKnots=nKnots)

def graphon_from_bspline_coeffs(theta, order=1):
    nKnots = int(np.sqrt(len(theta)))
    tau = np.concatenate(([0],np.linspace(0,1,nKnots),[1]))

    if order==0:
        prob_mat = theta.reshape((nKnots,nKnots))
        def _grad_func(x_eval,y_eval):
            vec_x = np.maximum(np.searchsorted(tau, np.array(x_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            vec_y = np.maximum(np.searchsorted(tau, np.array(y_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            return prob_mat[vec_x][:,vec_y]
    else:
        def _grad_func(x_eval,y_eval):
            x_eval_order = np.argsort(x_eval)
            y_eval_order = np.argsort(y_eval)
            func_eval_order=interpolate.bisplev(x= np.array(x_eval, ndmin=1, copy=False)[x_eval_order], y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order], tck=(tau, tau, theta, order, order), dx=0, dy=0)
            return eval('func_eval_order' + (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if len(x_eval_order) > 1 else ('[' if len(y_eval_order) > 1 else '')) + ('np.argsort(y_eval_order)]' if len(y_eval_order) > 1 else ''))
    Graphon_est = Graphon(func=_grad_func)
    Graphon_est.tau = tau
    Graphon_est.nKnots = nKnots
    Graphon_est.theta = theta
    Graphon_est.order = order
    return Graphon_est

def tuneLambdaSplineRegAIC(A,lambdaMin=0,lambdaMax=1000,z=None,order=1,nKnots=10,canonical=False):
    def opt_func(lmbda):
        Graphon_from_spline = graphon_from_bspline_em(A=A,z=z,order=order,canonical=canonical,lmbda=lmbda,returnAIC=True)
        return Graphon_from_spline
    return optimize.fminbound(func=opt_func,x1=lambdaMin,x2=lambdaMax,xtol=5e-01,maxfun=50)

def graphon_from_graph_bspline(A,z=None,order=1,nKnots=10,canonical=False,lmbda=50,returnAIC=False):
    (N,N) = A.shape
    if z is None:
        deg = np.sum(A,axis=0)
        ord_deg = np.argsort(deg)
        A = A[ord_deg,:][:,ord_deg]
        z = (deg[ord_deg]/max(deg+1))
    k=order
    Us_mult = z.reshape(1,-1)
    m = Us_mult.shape[0]
    nSpline1d = nKnots + k - 1
    nSpline = nSpline1d ** 2
    t = np.linspace(- k / (nKnots - 1), 1 + k / (nKnots - 1), nKnots + 2 * k)
    if k == 0:
        freqVec = np.array([[np.sum(indexVec1 == i) for i in range(nSpline1d)] for indexVec1 in np.maximum(np.ceil(Us_mult * nSpline1d) - 1, 0)])
        freqVecCum = np.array([np.append([0], line1) for line1 in np.cumsum(freqVec, axis=1)])
        indexVecMat = np.array([np.vstack((freqVecCum_i[:-1], freqVecCum_i[1:])).T for freqVecCum_i in freqVecCum])
        def itemset(array, pos_i, pos_j, val):
            array[(pos_i[0]):(pos_i[1])][:, (pos_j[0]):(pos_j[1])] = val
            return (array)
        B = np.array([[itemset(array=np.zeros((N, N)), pos_i=indexVec[i], pos_j=indexVec[j], val=1) for i in range(nSpline1d) for j in range(nSpline1d)] for indexVec in indexVecMat])
        if canonical:
            A_part = (1 / nSpline1d) * np.repeat(1, nSpline1d)
    elif k == 1:
        B = np.array([np.array([interpolate.bisplev(x=np.sort(Us), y=np.sort(Us), tck=(t, t, np.lib.pad([1], (i, nSpline - i - 1), 'constant', constant_values=(0)), k, k), dx=0, dy=0) for i in np.arange(nSpline)])[:, np.argsort(np.argsort(Us)), :][:, :, np.argsort(np.argsort(Us))] for Us in Us_mult])
        if canonical:
            A_part = (1 / (nSpline1d - 1)) * np.concatenate(([1 / 2], np.repeat(1, nSpline1d - 2), [1 / 2]))
    else:
        raise TypeError('B-splines of degree k = ' + k.__str__() + ' have not been implemented yet')
    B_cbind = np.array([np.delete(B[l].reshape(nSpline, N ** 2), np.arange(N) * (N + 1), axis=1) for l in range(m)])
    if canonical:
        A1 = np.vstack((np.array([np.pad(np.append(-A_part, A_part), (nSpline1d * i, nSpline1d * (nSpline1d - i - 2)), 'constant', constant_values=(0, 0)) for i in range(nSpline1d - 1)]), np.identity(nSpline), -np.identity(nSpline)))
    else:
        A1 = np.vstack((np.identity(nSpline), -np.identity(nSpline)))
    A2 = np.array([]).reshape((nSpline, 0))
    for i in range(nSpline1d):
        for j in range(i + 1, nSpline1d):
            NullMat = np.zeros((nSpline1d, nSpline1d))
            NullMat[i, j], NullMat[j, i] = 1, -1
            A2 = np.hstack((A2, NullMat.reshape(nSpline, 1)))
    A2 = A2.T
    L_part = np.identity(nSpline1d)[:-1] - np.hstack((np.zeros((nSpline1d - 1, 1)), np.identity(nSpline1d - 1)))
    I_part = np.identity(nSpline1d)
    penalize = np.dot(np.kron(I_part, L_part).T, np.kron(I_part, L_part)) + np.dot(np.kron(L_part, I_part).T, np.kron(L_part, I_part))
    G_ = cvxopt.matrix(-A1)
    A_ = cvxopt.matrix(A2)
    theta_t = np.repeat(np.mean(np.sum(A,axis=0)) / N, nSpline)
    cvxopt.solvers.options['show_progress'] = False
    differ = 5
    index_marker = 1
    while (differ > 0.01 ** 2):
        Pi = np.minimum(np.maximum(np.sum(B.swapaxes(1, 3) * theta_t, axis=3), 1e-5), 1 - 1e-5)
        mat1 = (B.swapaxes(0, 1) * ((A * (1 / Pi)) - ((1 - A) * (1 / (1 - Pi))))).swapaxes(0, 1)
        score = np.sum(np.sum(np.sum(mat1, axis=0), axis=1), axis=1) - np.sum(np.sum([np.diagonal(mat1[l], axis1=1, axis2=2).T for l in range(m)], axis=0), axis=0)
        mat2 = 1 / (Pi * (1 - Pi))
        fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(N ** 2, ), np.arange(N) * (N + 1)), B_cbind[l].T) for l in range(m)]), 0)
        P_ = cvxopt.matrix(fisher + lmbda * penalize)
        q_ = cvxopt.matrix(-score + lmbda * np.dot(theta_t, penalize))
        if canonical:
            h_ = cvxopt.matrix(np.dot(A1, theta_t) + np.append(np.zeros(nSpline1d - 1 + nSpline, ), np.ones(nSpline, )))
        else:
            h_ = cvxopt.matrix(np.dot(A1, theta_t) + np.append(np.zeros(nSpline, ), np.ones(nSpline, )))
        b_ = cvxopt.matrix(np.dot(-A2, theta_t))
        delta_t = np.squeeze(np.array(cvxopt.solvers.qp(P=P_, q=q_, G=G_, h=h_, A=A_, b=b_)['x']))
        theta_tOld = copy(theta_t)
        theta_t = delta_t + theta_t
        differ = (1 / nSpline) * np.sum((theta_t - theta_tOld) ** 2)
        # print('Iteration of estimating theta:', index_marker)
        index_marker = index_marker + 1
        if index_marker > 10:
            warnings.warn('Fisher scoring did not converge')
            print('UserWarning: Fisher scoring did not converge')
            print(theta_tOld)
            print(theta_t)
            print(np.round(theta_t - theta_tOld, 4))
            break
    if returnAIC:
        Pi = np.minimum(np.maximum(np.sum(B.swapaxes(1, 3) * theta_t, axis=3), 1e-5), 1 - 1e-5)
        mat2 = 1 / (Pi * (1 - Pi))
        fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(N ** 2, ), np.arange(N) * (N + 1)), B_cbind[l].T) for l in range(m)]), 0)
        df_lambda = np.trace(np.dot(np.linalg.inv(fisher + lmbda * penalize), fisher))
        logProbMat = (A * np.log(Pi)) + ((1 - A) * np.log(1 - Pi))
        [np.fill_diagonal(logProbMat_i, 0) for logProbMat_i in logProbMat]
        ret =  -2 * np.sum(logProbMat) + 2 * df_lambda + ((2 * df_lambda * (df_lambda + 1)) / (((N ** 2 - N) * m) - df_lambda - 1))
        return ret
    else:
        if k == 0:
            def fct(x_eval, y_eval):
                vec_x = np.maximum(np.ceil(np.array(x_eval, ndmin=1, copy=False) * nSpline1d) - 1, 0).astype(int)
                vec_y = np.maximum(np.ceil(np.array(y_eval, ndmin=1, copy=False) * nSpline1d) - 1, 0).astype(int)
                return (theta_t.reshape((nSpline1d, nSpline1d))[vec_x][:, vec_y])
        if k == 1:
            def fct(x_eval, y_eval):
                x_eval_order = np.argsort(x_eval)
                y_eval_order = np.argsort(y_eval)
                fct_eval_order = interpolate.bisplev(x=np.array(x_eval, ndmin=1, copy=False)[x_eval_order], y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order], tck=(t, t, theta_t, k, k), dx=0, dy=0)
                return (eval('fct_eval_order' + (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if len(x_eval_order) > 1 else ('[' if len(y_eval_order) > 1 else '')) + ('np.argsort(y_eval_order)]' if len(y_eval_order) > 1 else '')))
        graphonEst = Graphon(func=fct)
        graphonEst.tau = t
        graphonEst.nKnots = nKnots
        graphonEst.theta = theta_t
        graphonEst.order = k
        return graphonEst

def gibbs(W,A,z=None,steps=300,rep=10,proposal='logit_norm',acceptRate=np.array([]),
          sigma_prop=2,returnAllGibbs=False,averageType='mean'):
    (N,N) = A.shape
    if z is None:
        deg = np.sum(A,axis=0)
        ord_deg = np.argsort(deg)
        A = A[ord_deg,:][:,ord_deg]
        z = (deg[ord_deg]/max(deg+1))

    z_MCMC = np.zeros((0, N))
    if returnAllGibbs:
        z_MCMC_all = np.zeros((rep*steps,N))
    U = np.minimum(np.maximum(z,1e-5),1-1e-5)
    for rep_step in range(rep):
        Decision = np.zeros(shape=[steps,N], dtype=bool)
        for step in range(steps):
            for k in np.random.permutation(np.arange(N)):
                if proposal == 'logit_norm':
                    z_star_k=np.random.normal(loc=np.log(U[k]/(1-U[k])),scale=sigma_prop)
                    u_star_k=np.exp(z_star_k)/(1+np.exp(z_star_k))
                if proposal == 'uniform':
                    u_star_k=np.random.uniform(0,1,1)
                u_no_k=np.delete(U,k)
                y_no_k=np.delete(A[k],k)
                w_fct_star_k=np.minimum(np.maximum(W.func(u_star_k,u_no_k), 1e-5), 1-1e-5)
                w_fct_k=np.minimum(np.maximum(W.func(U[k],u_no_k), 1e-5), 1-1e-5)
                prod_k=np.prod(np.squeeze(np.asarray(((w_fct_star_k/w_fct_k)**y_no_k))) * \
                    np.squeeze(np.asarray((((1-w_fct_star_k)/(1-w_fct_k))**(1-y_no_k)))))
                if proposal == 'logit_norm':
                    alpha=min(1,prod_k*((u_star_k*(1-u_star_k))/(U[k]*(1-U[k]))))
                if proposal == 'uniform':
                    alpha=min(1,prod_k)
                Decision[step,k] = (np.random.binomial(n=1,p=alpha)==1)
                if Decision[step,k]:
                    U[k] = np.min([np.max([u_star_k, 1e-5]), 1-1e-5])
                if returnAllGibbs:
                    z_MCMC_all[rep_step * steps + step,k] = U[k]
        z_MCMC = np.vstack((z_MCMC, U))
        new_acceptRate = np.sum(Decision)/(N*steps)
        acceptRate = np.append(acceptRate, new_acceptRate)
    if averageType == 'mean':
        z_new = np.mean(z_MCMC, axis=0)
    if averageType == 'median':
        z_new = np.median(z_MCMC, axis=0)
    z_new_std = (np.linspace(0,1,N+2)[1:-1])[np.argsort(np.argsort(z_new))]
    z_MCMC_std = np.array([(np.linspace(0,1,N+2)[1:-1])[np.argsort(np.argsort(z_MCMC[i]))] for i in range(z_MCMC.shape[0])])
    if returnAllGibbs:
        return z_new,z_new_std,z_MCMC,z_MCMC_all
    else:
        return z_new,z_new_std

def iterateEM(A,z=None,order=1,nKnots=10,canonical=False,
              n_steps=30,proposal='logit_norm',sigma_prop=2,averageType='mean',use_stdVals=True,
              n_eval=3,n_iter=15,rep_start=1,rep_end=25,it_rep_grow=5,rep_forPost=15,
              raiseLabNb=False,
              lambda_start=50, lambda_skip1=10, lambda_lim1=None, lambda_skip2=None, lambda_lim2=None,lambda_last_m=3
             ):
    result = type('', (), {})()
    (N,N) = A.shape
    if z is None:
        deg = np.sum(A,axis=0)
        ord_deg = np.argsort(deg)
        A = A[ord_deg,:][:,ord_deg]
        z = (deg[ord_deg]/max(deg+1))

    lmbda = lambda_start
    lmbdas = np.array([])
    trajMat = np.zeros((0,n_eval,n_eval))

    # EM based algorithm
    for index in range(1,n_iter+1):
        labNb = index + raiseLabNb
        # Graph update:
        if index > 1:
            if use_stdVals:
                z = z_new_std
            else:
                z = z_new
        # Graphon update:
        if index in (list(range(lambda_skip1+1,lambda_lim1+1)) + list(range(lambda_skip2+1,lambda_lim2+1)) + list(range(n_iter-(lambda_last_m-1), n_iter+1))):
            lmbda = tuneLambdaSplineRegAIC(A=A,lambdaMin=0,lambdaMax=1000,z=z,order=order,nKnots=nKnots,canonical=canonical)
            if index in range(lambda_skip2+1,lambda_lim2+1):
                lmbdas = np.append(lmbdas,lmbda)
            if index == lambda_lim2:
                lmbda = lmbdas.mean()
        Graphon_est = graphon_from_bspline_em(A=A,z=None,order=order,nKnots=nKnots,canonical=canonical,lmbda=lmbda,returnAIC=False)
        trajMat = np.append(trajMat,Graphon_est.func(np.arange(1, n_eval+1) / (n_eval+1), np.arange(1, n_eval+1) / (n_eval+1)).reshape(1, n_eval, n_eval), axis=0)
        # zeta update:
        if (index < n_iter) or (rep_forPost > 0):
            rep = rep_start if (index < it_rep_grow) else (int(np.round((rep_start**(n_iter-it_rep_grow) / rep_end)**(1/(n_iter-it_rep_grow-1)) * np.exp(np.log(rep_start / (rep_start**(n_iter-it_rep_grow) / rep_end)**(1/(n_iter-it_rep_grow-1))) * (index-it_rep_grow+1)))) if (index < n_iter) else (rep_forPost))
            z_new,z_new_std = gibbs(W=Graphon_est,A=A,z=z,steps=n_steps,rep=rep,proposal=proposal,sigma_prop=sigma_prop,returnAllGibbs=False,averageType=averageType)
    result.A = A
    result.W = Graphon_est
    result.z = z
    result.lmbda = lmbda
    result.trajMat = trajMat
    return result

def graphon_from_em_bspline(A,z=None,order=1,nKnots=10,canonical=False,
                       n_eval=3,ks_abs=None,ks_rel=np.array([0,.25,.75,1]),n_steps=30,proposal='logit_norm',
                       sigma_prop=2,averageType='mean',
                       use_stdVals=True,
                       n_iter=25,rep_start=1,rep_end=25,it_rep_grow=5,rep_forPost=15,
                       lambda_start=50,lambda_skip1=10,lambda_skip2=None,lambda_lim1=None,lambda_lim2=None,lambda_last_m=3
                      ):
    if lambda_lim1==None:
        lambda_lim1 = (3) + lambda_skip1
    if lambda_skip2==None:
        lambda_skip2 = (2) + lambda_lim1
    if lambda_lim2==None:
        lambda_lim2 = (3) + lambda_skip2

    (N,N) = A.shape
    if z is None:
        deg = np.sum(A,axis=0)
        ord_deg = np.argsort(deg)
        A = A[ord_deg,:][:,ord_deg]
        z = (deg[ord_deg]/max(deg+1))

    lmbda = tuneLambdaSplineRegAIC(A=A,lambdaMin=0,lambdaMax=1000,z=z,order=order,nKnots=nKnots,canonical=canonical)
    Graphon_est0 = graphon_from_bspline_em(A=A,z=z,order=order,nKnots=nKnots,canonical=canonical,lmbda=lmbda,returnAIC=False)
    trajMat = Graphon_est0.func(np.arange(1, n_eval + 1) / (n_eval + 1), np.arange(1, n_eval + 1) / (n_eval + 1)).reshape(1, n_eval, n_eval)

    if ks_abs is None:
        ks_abs = np.unique(np.minimum(np.maximum(np.round(ks_rel * N).astype('int') - 1, 0), N - 1))  # absolute k's for which the posterior is calculated

    z_new,z_new_std = gibbs(W=Graphon_est0,A=A,z=z,steps=n_steps,rep=rep_forPost,proposal=proposal,sigma_prop=sigma_prop,averageType=averageType)
    if use_stdVals:
        z = z_new_std
    else:
        z = z_new

    EM_obj = iterateEM(A=A,z=z,order=order,nKnots=nKnots,canonical=canonical,n_steps=n_steps,
                       proposal=proposal,sigma_prop=sigma_prop,averageType=averageType,use_stdVals=use_stdVals,
                       n_eval=n_eval,n_iter=n_iter,rep_start=rep_start,rep_end=rep_end,it_rep_grow=it_rep_grow,
                       rep_forPost=rep_forPost,raiseLabNb=True,
                       lambda_start=lambda_start, lambda_skip1=lambda_skip1, lambda_lim1=lambda_lim1, lambda_skip2=lambda_skip2, lambda_lim2=lambda_lim2, lambda_last_m=lambda_last_m
                      )
    return EM_obj.W

#---------------------------------------------------------------
'''
Reference:
Chan, Stanley and Airoldi, Edoardo.
"A Consistent Histogram Estimator for Exchangeable Graph Models."
ICML, PMLR 32(1) (2014): 208-216.
'''
def graphon_from_sas(A,z=None,num_bins=5,nKnots=10):
    (N,N) = A.shape
    if z is None:
        deg = np.sum(A,axis=0)
        ord_deg = np.argsort(deg)
        A = A[ord_deg,:][:,ord_deg]
        z = (deg[ord_deg]/max(deg+1))

    num_bins = int(np.minimum(N,num_bins))
    F = np.kron(np.eye(num_bins),np.ones((int(np.ceil(N/num_bins)),int(np.ceil(N/num_bins))))-np.eye(int(np.ceil(N/num_bins))))[:N,:N]
    F[np.sum(F,axis=0)>0,:]/=(np.sum(F,axis=0)[np.sum(F,axis=0)>0]).reshape(-1,1)

    sbm = F@A@F.T
    W_est = Graphon(sbm=sbm,nKnots=nKnots)
    return W_est

