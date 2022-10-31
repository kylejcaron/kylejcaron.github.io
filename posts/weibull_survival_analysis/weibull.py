from typing import *
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.special as sp

class Weibull:

    def __init__(self, k, lambd):
        self.k = k
        self.lambd = lambd

    def sample(self, n: int = 1, left_trunc: float = None) -> np.array:
        '''Samples from the weibull distribution
        '''

        dist = stats.weibull_min(self.k, scale=self.lambd)
        if not left_trunc:
            return stats.weibull_min(self.k, scale=self.lambd).rvs(n)
        
        U = np.random.uniform(dist.cdf(left_trunc), 1, size=n)
        return dist.ppf(U)

    def cdf(self, t: np.array) -> np.array:
        '''The cumulative density function evaluated at t
        '''
        return 1 - np.exp(-(t/self.lambd)**self.k)

    def expectation(self) -> np.array:
        '''Calculates the expectation of the weibull distribution
        '''
        return self.lambd * sp.gamma(1 + 1/self.k)

    def survival(self, t: np.array, curr_time: Optional[int] = None) -> np.array:
        '''Outputs the survival probability at each time point T. This is done with the survival function, 
        the complement of the Weibull Distribution's PDF.

        Can also be used to calculate conditional survival with the `curr_time` argument.

        Parameters
        -----------
            t: A numpy array with time points to calculate the survival curve,      
                utilizing the distributions parameters
            curr_time: Used to calculate the survival curve given already reaching 
                some point in time, `curr_time`.
        
        Returns
        -------
            St: A survival curve calculated over the inputted time period
        '''
        # Normalizing constant used for conditional survival
        norm = 1 if curr_time is None else self.survival(curr_time)
        
        # check inputs
        t = self._normalize_to_array(t)
        if curr_time is not None and (t < curr_time).sum() > 1:
            raise ValueError('t<curr_time. t must be greater than or equal to curr_time')
        
        St = (1 - self.cdf(t))/norm
        return St

    def hazard(self, t: np.array) -> np.array:
        '''Outputs the hazard rate at each time point T.

        Parameters
        -----------
            t: A numpy array with time points to calculate the survival curve,      
                utilizing the distributions parameters
        
        Returns
        -------
            St: A survival curve calculated over the inputted time period
        '''
        t = self._normalize_to_array(t)
        ht = (self.k/self.lambd)*(t/self.lambd)**(self.k-1)
        return ht

    def mean_residual_life(self, t: np.array) -> np.array:
        '''
        '''
        t = self._normalize_to_array(t)
        St = self.survival(t)
        numerator = (
            self.lambd 
            * sp.gammaincc(1+1/self.k, (t/self.lambd)**self.k)
            * sp.gamma(1+1/self.k))

        result = np.divide(
            numerator,
            St,
            out=np.zeros_like(numerator),
            where=St!=0
        ) - t[:,None]
        
        # The code above returns negative values when St=0. This clipping corrects those cases
        return np.clip(result, 0, np.inf)
    
    def _normalize_to_array(self, inpt: Union[np.array, List, int, float, pd.Series]) -> np.array:
        if isinstance(inpt, (np.ndarray,)):
            return inpt
        elif isinstance(inpt, (int, float)):
            return np.array([inpt])
        elif isinstance(inpt, (list,)):
            return np.array(inpt)
        elif isinstance(inpt, (pd.Series,)):
            return inpt.values
        else: 
            raise TypeError("Input is not a supported type. See type hints")