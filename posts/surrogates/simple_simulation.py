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

numpy_rng = np.random._generator.Generator
rng = np.random.default_rng(seed=SEED)

def _get_rng():
	seed = np.random.choice(range(1000))
	rng = np.random.default_rng(seed=seed)
	return rng

def set_true_parameters(
	Zdims: int, 
	Xdims: int, 
	Sdims: int, 
	logit_link: bool = False, 
	rng: numpy_rng = None
) -> Dict:
    '''Initializes ground truth parameters to be used for simulation. Includes effects and bias terms.

    Parameters
    -----------
        Zdims: The number of dimensions representing the customer feature latent space
        Xdims: The number of pre treatment covariates to include in the simulation
        Sdims: the number of surrogates to include in the simulation
        logit_link: whether the data generating process is a bernoulli outcome or not
        rng: A numpy random generator 

    Returns
    --------
        GROUND_TRUTH: A dictionary storing the true parameters and effects used for simulation
    '''
    rng = rng if rng is not None else _get_rng()

    GROUND_TRUTH = dict(

        # bias terms
        # Decided to remove bias terms for X and S since it makes them 0 centered, which also
        # makes interpretation easy: HTE terms end up cancelling out, and dont need to be included when
        # calculating an overall average treatment effect
        alphaX = np.zeros(Xdims), 
        alphaS = np.zeros(Sdims),
        alphaY = np.array([0.5]),

        # Effects
        bZX = rng.normal(0,0.25,size=(Xdims, Zdims)),
        bXS =  rng.normal(0,0.25,size=(Sdims, Xdims)),
        bSY = rng.normal(0,0.5,size=(1, Sdims)),

        # Treatment Effects
        bTS = rng.normal(0,0.5,size=(1, Sdims)),
        bXTS = rng.normal(0,0.015/Sdims ,size=(Sdims, Xdims))
    )

    # Average treatment effect is the effect of the Treatment on the Intermediate outcomes
    # multiplied by the effect of the intermediate outcomes on the Long-Term Outcome, Y
    GROUND_TRUTH['ATE'] = (GROUND_TRUTH["bTS"] @ GROUND_TRUTH["bSY"].T).ravel()
    if logit_link:
        GROUND_TRUTH['ATE'] = sp.expit( sp.logit(GROUND_TRUTH["alphaY"]) + GROUND_TRUTH['ATE']) - GROUND_TRUTH["alphaY"]
 
    return GROUND_TRUTH



def transmitter(
    params: Dict,
    Zdims: int = 20, 
    Xdims: int = 5, 
    Sdims: int = 3,
    add_treatment: bool = False,
    n_users: int = 1,
    logit_link: bool = False,
    rng: numpy_rng = None
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
    rng = rng if rng is not None else _get_rng()

    # unpack params
    Zdims, Xdims, Sdims =  params['bZX'].shape[1],  params['bZX'].shape[0],  params['bXS'].shape[0]
    alphaX,alphaS,alphaY  = params['alphaX'], params['alphaS'], params['alphaY'] # bias terms
    bZX,bXS,bSY = params['bZX'], params['bXS'],params['bSY'] # causal relationships
    bTS,bXTS = params['bTS'], params['bXTS'] # tx effects

    # unobserved variable Z representing latent customer traits
    Z = rng.normal(0,1, size=(Zdims, n_users))

    # Some observed pre-TX measures
    X = alphaX[:,None] + (bZX @ Z)

    # Intermediate outcomes
    S = alphaS[:,None] + (bXS @ X) 

    # Add in treatment effect if applicable
    T = rng.binomial(1,0.5,size=n_users) if add_treatment else np.zeros(n_users)        
    avg_tx_term = (bTS * T[:,None])        
    hetergeneous_tx_term = (bXTS @ (X*T))
    S += avg_tx_term.T + hetergeneous_tx_term

    # expectation of long term outcome
    eta = 0 + (bSY @ S)

    # Long term outcome
    if logit_link:
        Y = rng.binomial(1, sp.expit(eta) )
    else:
        Y = rng.normal(eta, 0.025)

    # Output as dataframe
    Xdf = pd.DataFrame(X.T, columns=[f'X{i}' for i in range(Xdims)]).assign(T=T)
    Sdf = pd.DataFrame(S.T, columns=[f'S{i}' for i in range(Sdims)])
    Ydf = pd.DataFrame(Y.ravel(), columns=['Y'])
    return pd.concat((Xdf, Sdf, Ydf),axis=1)


