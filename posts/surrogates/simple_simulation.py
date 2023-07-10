from typing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import scipy.stats as stats
import scipy.special as sp
import graphviz as gv
import statsmodels.api as sm
SEED=99


class Transmitter:
    
    def __init__(
        self, 
        Zdims: int, 
        Xdims: int, Sdims: int, 
        logit_link: bool = False,
        seed: int = None
    ) -> None:

        self.rng = np.random.default_rng(seed=seed)
        self.logit_link = logit_link
        self.Zdims, self.Xdims, self.Sdims = Zdims, Xdims, Sdims
        self.params = self._set_true_parameters()
            
    def _set_true_parameters(self):

        ground_truth = dict(

            # bias terms
            # Decided to remove bias terms for X and S since it makes them 0 centered, which also
            # makes interpretation easy: HTE terms end up cancelling out, and dont need to be included when
            # calculating an overall average treatment effect
            alphaX = np.zeros(self.Xdims), 
            alphaS = np.zeros(self.Sdims),
            alphaY = np.zeros(1),

            # Effects
            bZX = self.rng.normal(0,1,size=(self.Zdims, self.Xdims)),
            bXS =  self.rng.normal(0,1,size=(self.Xdims, self.Sdims)),
            bSY = self.rng.normal(0,1,size=(self.Sdims, 1))
        )    
        
        return ground_truth    
    
    def simulate(
        self,
        n_users: int = 1,
        add_treatment: bool = False,
        bTS = None,
        bXTS = None,
    ) -> pd.DataFrame:
        '''Simulates outcomes based on some ground truth parameters. 

        Parameters
        -----------
            params: The ground truth parameters (effects and biases) to simulate based off of
            add_treatment: adds a randomly allocated treatment when true, with effect `bTS`
            n_users: The number of users to simulate
            logit_link: whether the data generating process is a bernoulli outcome or not
            rng: A numpy random generator 

        Returns
        --------
            A pandas dataframe with simulated data, including pre-treatment covariates,
             surrogate outcomes, a treatment indicator, and a long term outcome, Y
        '''
        
        def eps(n_users, scale=0.25):
            return np.random.normal(0, scale, size=n_users)
        
        # unpack params
        alphaX,alphaS,alphaY  = self.params['alphaX'], self.params['alphaS'], self.params['alphaY'] # bias terms
        bZX,bXS,bSY = self.params['bZX'], self.params['bXS'],self.params['bSY'] # causal relationships
        bTS, bXTS = self._simulate_tx(bTS, bXTS) # tx effects

        # unobserved variable Z representing latent customer traits
        Z = np.random.normal(0,.25, size=(n_users, self.Zdims))

        # Some observed pre-TX measures
        X = alphaX[:,None].T + (Z @ bZX) + eps((n_users, self.Xdims))

        # Intermediate outcomes
        S = alphaS[:,None].T + (X @ bXS) + eps((n_users, self.Sdims))
        
        # Add in treatment effect if applicable
        T = np.random.binomial(1,0.5,size=n_users) if add_treatment else np.zeros(n_users)        
        avg_tx_term = (bTS * T[:,None])  
        hetergeneous_tx_term = ((X*T[:,None]) @ bXTS.T)
        S += avg_tx_term + hetergeneous_tx_term 

        # long term outcome
        eta = (alphaY[:,None] + (S @ bSY))
            
        # Calculate true treatment effects
        bTY = (bTS @ bSY).ravel()
        eta_scale = eta.std() # TODO: dont estimate eta_scale, calculate it analytically
        ATE = (
            self._approx_expit_expectation(bTY, eta_scale ) 
            - self._approx_expit_expectation(0, eta_scale )
        ) if self.logit_link else bTY
        ground_truth = {"bTS":bTS,"bXTS":bXTS, "bTY":bTY, "ATE":ATE}

        # Long term outcome
        if self.logit_link:
            Y = np.random.binomial(1, sp.expit( eta )) 
        else:
            Y = np.random.normal(eta, .1)

        # Output as dataframe
        Xdf = pd.DataFrame(X, columns=[f'X{i}' for i in range(self.Xdims)]).assign(T=T)
        Sdf = pd.DataFrame(S, columns=[f'S{i}' for i in range(self.Sdims)])
        Ydf = pd.DataFrame(Y.ravel(), columns=['Y'])
        return pd.concat((Xdf, Sdf, Ydf),axis=1), ground_truth

    def _approx_expit_expectation(self, mu, scale):
        '''Approximates the expectation of a RV transformed by an expit function,
        which is highly dependent on the variance of the RV.
        
        https://math.stackexchange.com/questions/207861/expected-value-of-applying-the-sigmoid-function-to-a-normal-distribution/1994805
        '''
        gamma_sq = 0.61**2 #np.pi/8
        phi = stats.norm(0,1).cdf
        val = mu / (np.sqrt( gamma_sq**-1 + scale**2 ))
        return phi(val)

    def _simulate_tx(self, bTS=None, bXTS=None):
        '''Used to simulate a new treatment effect for each simulation. 
        Can be overwritten by supplying either bTS or bXTS
        '''
        check_shape = lambda shape, arr: arr.shape[0] == shape[0] and arr.shape[1] == shape[1]
        bTS_shape, bXTS_shape = (1, self.Sdims), (self.Sdims, self.Xdims)        
        
        bTS = bTS if bTS is not None else np.random.normal(0,1,size=bTS_shape)
        bXTS = bXTS if bXTS is not None else np.random.normal(0,0.015/self.Sdims,size=bXTS_shape)
        
        assert check_shape(bTS_shape, bTS), f'bTS should have shape ({bTS_shape})'
        assert check_shape(bXTS_shape, bXTS), f'bXTS should have shape ({bXTS_shape})'
        return bTS, bXTS
