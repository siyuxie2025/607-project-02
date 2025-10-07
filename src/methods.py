# class definition
import numpy as np
import pandas as pd
from quantes.linear import low_dim
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os

class RiskAwareBandit:
    '''
    A class for the Risk-Aware Bandit algorithm with forced sampling and all-sample estimators.
    '''
    def __init__(self, q, h, tau, d, K, beta_real_value, alpha_real_value):
        '''
        Initialize the Risk-Aware Bandit algorithm.

        Parameters:
        q (int) : Number of forced samples per arm in each round
        h (float) : Difference between optimal and suboptimal arms
        tau (float) : Quantile level for quantile regression
        d (int) : Dimension of the context vectors
        K (int) : Number of arms
        beta_real_value (np.ndarray) : True coefficient values for each arm
        alpha_real_value (np.ndarray) : True intercept values for each arm
        '''
        self.Tx = np.empty((K, 0)).tolist()
        self.Sx = np.empty((K, 0)).tolist()
        self.Tr = np.empty((K, 0)).tolist()
        self.Sr = np.empty((K, 0)).tolist()

        self.q = q
        self.h = h
        self.tau = tau
        self.d = d
        self.K = K

        self.beta_t = np.random.uniform(0., 2., (K, d)) # forced sample estimator
        self.beta_a = np.random.uniform(0., 2., (K, d)) # all sample estimator
        self.alpha_t = np.random.uniform(0., 2., K) # forced sample intercept
        self.alpha_a = np.random.uniform(0., 2., K) # all sample intercept
        self.n = 0

        self.beta_real_value = beta_real_value
        self.alpha_real_value = alpha_real_value

        self.beta_error_a = np.zeros(K)
        self.beta_error_t = np.zeros(K)

    
    def choose_a(self,t,x): #x is d-dim vector
        # if t is the first time of the new round
        '''
        
        '''
        if t == ((2**self.n - 1)*self.K*self.q + 1):
            self.set = np.arange(t, t+self.q*self.K)
            self.n += 1
        if t in self.set: # forced sampling
            ind = list(self.set).index(t)
            self.action = ind // self.q
            self.Tx[self.action].append(x)
        else:
            forced_est = np.dot(self.beta_t, x) + self.alpha_t
            max_forced_est = np.amax(forced_est)
            K_hat = np.where(forced_est > max_forced_est - self.h/2.)[0]
            all_est = [np.dot(self.beta_a[k_hat], x) + self.alpha_a[k_hat] for k_hat in K_hat]
            print(f"K_hat: {K_hat}, all_est: {all_est}")  ## print the result
            self.action = K_hat[np.argmax(all_est)]

        self.Sx[self.action].append(x)

        return self.action
    
    # update beta
    def update_beta(self, rwd, t):
        if np.array(self.Tx[self.action]).shape[0] > self.d:
            if t in self.set:
                self.Tr[self.action].append(rwd)
                forced_qr = low_dim(np.array(self.Tx[self.action]), 
                                    np.array(self.Tr[self.action]), 
                                    intercept=True).fit(tau=self.tau)
                self.beta_t[self.action] = forced_qr['beta'][1:]
                self.alpha_t[self.action] = forced_qr['beta'][0]

            self.Sr[self.action].append(rwd)
            all_qr = low_dim(np.array(self.Sx[self.action]), 
                            np.array(self.Sr[self.action]), 
                            intercept=True).fit(tau=self.tau)
            self.beta_a[self.action] = all_qr['beta'][1:]
            self.alpha_a[self.action] = all_qr['beta'][0]
            # print(f"Shapes - Tx: {np.array(self.Tx[self.action]).shape}, Tr: {np.array(self.Tr[self.action]).shape}")
            self.beta_error_a[self.action] = np.linalg.norm(self.beta_a[self.action] - self.beta_real_value[self.action])
            self.beta_error_t[self.action] = np.linalg.norm(self.beta_t[self.action] - self.beta_real_value[self.action])
            return
        else:
            self.Tr[self.action].append(rwd)
            self.Sr[self.action].append(rwd)
            self.beta_t[self.action] = np.random.uniform(0., 1., self.d)
            self.beta_a[self.action] = np.random.uniform(0., 1., self.d)
            self.alpha_t[self.action] = np.random.uniform(0., 1.)
            self.alpha_a[self.action] = np.random.uniform(0., 1.)

            self.beta_error_a[self.action] = np.linalg.norm(self.beta_a[self.action] - self.beta_real_value[self.action])
            self.beta_error_t[self.action] = np.linalg.norm(self.beta_t[self.action] - self.beta_real_value[self.action])
            return 
        

