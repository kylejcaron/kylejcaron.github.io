from typing import Dict, Tuple
from functools import partial

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan


class RentalInventory:
	"""A model of rental inventory, modeling stock levels as returns and rentals occur each day.
	Currently supports a single product
	"""
	def __init__(self, n_products: int = 1, policies: np.ndarray = None):
		self.n_products = n_products
		self.policies = policies if policies is not None else jnp.zeros((n_products, 10000))
		# Rentals that are out with customers are stored as an array, where the index corresponds with time, 
		# and the value corresponds with the number of rentals from that time that are still out with customers
		# max_periods is the total number of periods to log
		self.max_periods = 10000

	def model(self, init_state: Dict, start_time: int, end_time: int) -> jnp.array:
		"""The Rental Inventory model. Each day returns occur as represented by a lognormal time to event distribution,
		and rentals occur as simulated by a poisson distribution and constrained physically by stock levels.

		Args:
			init_state (Dict): The initial inventory state
			start_time (int): the start time (represented as an integer)
			end_time (int): the end time (represented as an integer)

		Returns:
			jnp.array: An array of rentals at each date
		"""
		_, ys = scan(
			self.model_single_day, 
			init=init_state, 
			xs=jnp.arange(start_time, end_time)
		)
		return ys

	def model_single_day(self, prev_state: Dict, time: int) -> Tuple[Dict, jnp.array]:
		"""Models a single day of inventory activity, including returns, rentals, and stock changes

		Args:
			state (dict): _description_
			time (int): _description_

		Returns:
			Tuple[Dict, jnp.array]: Returns the following inventory state and rentals that occurred in the current state
		"""
		curr_state = dict()

		# Simulate Returns
		returns = self.returns_model(prev_state['existing_rentals'], time)
		curr_state['starting_stock'] = numpyro.deterministic("starting_stock", prev_state['ending_stock'] + returns.sum(1) + self.apply_policy(time))

		# Simulate Rentals, incorporate them into the next state
		rentals = self.demand_model(available_stock=curr_state['starting_stock'], time=time)
		curr_state['ending_stock'] = numpyro.deterministic("ending_stock", curr_state['starting_stock'] - rentals.sum(1))
		curr_state['existing_rentals'] = numpyro.deterministic("existing_rentals", prev_state['existing_rentals'] - returns + rentals)
		return curr_state, rentals

	def returns_model(self, existing_rentals: jnp.array, time: int) -> jnp.array:
		"""Models the number of returns each date
		"""
		# Distribution of possible rental durations
		theta = numpyro.sample("theta", dist.Normal(2.9, 0.01))
		sigma = numpyro.sample("sigma", dist.TruncatedNormal(0.7, 0.01, low=0))
		return_dist = dist.LogNormal(theta, sigma)

		# Calculate the discrete hazard of rented out inventory from previous time-points being returned
		discrete_hazards = self.survival_convolution(dist=return_dist, time=time)

		# Simulate returns from hazards
		returns = numpyro.sample("returns", dist.Binomial(existing_rentals.astype("int32"), probs=discrete_hazards))
		_ = numpyro.deterministic("total_returns", returns.sum(1))
		return returns

	def survival_convolution(self, dist, time: int) -> jnp.array:
		"""Calculates the probability of a return happening (discrete hazard rate) from all past time periods, returning an array where each index is a previous time period,
		and the value is the probability of a rental from that time being returned at the current date.
		"""
		rental_durations = (time-jnp.arange(self.max_periods))
		discrete_hazards = jnp.where(
			# If rental duration is nonnegative,
			rental_durations>0,
			# Use those rental durations to calculate a return rate, using a discrete interval hazard function
			RentalInventory.hazard_func(jnp.clip(rental_durations, a_min=0), dist=dist ),
			# Otherwise, return rate is 0
			0
		)
		return discrete_hazards

	def demand_model(self, available_stock, time):
		"""Models the true demand each day.
		"""
		raise NotImplementedError()

	@staticmethod
	def hazard_func(t, dist):
		"""Discrete interval hazard function - aka the probability of a return occurring on a single date
		"""
		return (dist.cdf(t+1)-dist.cdf(t))/(1-dist.cdf(t))

	def apply_policy(self, time):
		return self.policies[:,time]

	@staticmethod
	def censored_multinomial(n, U_j, stock_j):
		"""This implements a series of iterative multinomial choices under inventory constraints
		"""
		
		stock = stock_j.copy()
		results = jnp.zeros(stock_j.shape[0])
		key = jax.random.PRNGKey(1)
		# Dont take more rentals then the total amount of stock
		n = jnp.where(stock.sum() < n, stock.sum(), n)
		state = (key, n, U_j, stock, results)
		
		def while_choices_left(state): 
			_,n,_,_,_ = state
			return n > 0
		
		@partial(jax.jit, static_argnums=(0))
		def sim_choices(state):
			eps=1e-10
			key,n,U_j,stock,results = state
			new_key, subkey = jax.random.split(key)
			avl_idx = jnp.where(stock>0, 1, 0)
			p_j = jax.nn.softmax(U_j, where=avl_idx, initial=0)
			# min_stock = jnp.where(avl_idx==1, stock, jnp.inf).min()
			# nchoices = jnp.minimum(min_stock, n).astype(int)
			choices = dist.Multinomial(total_count=1, probs=p_j, total_count_max=1).sample(subkey)
			results += choices
			stock -= choices            
			n -= choices.sum()
			return (new_key,n,U_j,stock,results)

		state = jax.lax.while_loop(while_choices_left, sim_choices, init_val=state)
		return state[-1]
	


class PoissonDemandInventory(RentalInventory):
	"""A model of rental inventory, modeling stock levels as returns and rentals occur each day.
	Currently supports a single product
	"""
	def __init__(self, n_products: int = 1, policies: np.ndarray = None):
		super().__init__(n_products, policies)
		rng = np.random.default_rng(seed=99)

		# Heterogeneity in demand is lambd ~ Exp(5) distributed when using this class to simulate data from scratch
		# When simulating demand based on an existinc dataset, this can be overwritten
		# i.e. `numpyro.do(inventory.demand_model, {"lambd": U_hat})`
		self.U = jnp.log( 5 * rng.exponential( size=n_products) )

	def demand_model(self, available_stock, time):
		"""Models the true demand each day.
		"""
		with numpyro.plate("n_products", self.n_products) as ind:
			lambd = numpyro.sample("lambd", dist.Normal(jnp.exp(self.U[ind]), 0.001))

		unconstrained_rentals = numpyro.sample("unconstrained_rentals", dist.Poisson(lambd))
		rentals = numpyro.deterministic("rentals", jnp.clip(unconstrained_rentals, a_min=0, a_max=available_stock ))
		rentals_as_arr = ( time == jnp.arange(self.max_periods) )*rentals[:,None]
		return rentals_as_arr


class MultinomialDemandInventory(RentalInventory):
	"""A model of rental inventory, modeling stock levels as returns and rentals occur each day.
	Currently supports a single product
	"""
	def __init__(self, n_products: int = 1, policies: np.ndarray = None):
		super().__init__(n_products, policies)
		rng = np.random.default_rng(seed=99)
		self.X = rng.normal(0,1,size=(n_products, 5))
		self.beta = np.array([0.248, -0.845, -0.0385, -0.00045, 0.2795])
		self.U = self.X.dot(self.beta) + rng.gumbel(0,0.2)

		self.total_rental_rate = 5000


	def demand_model(self, available_stock, time):
		"""Models the true demand each day.
		"""
		# Hyperparameters
		lambd_total = numpyro.sample("lambd", dist.Normal(self.total_rental_rate, 0.01))
		utility = numpyro.deterministic("utlity",self.U)

		# Generative model
		total_rentals = numpyro.sample("total_rentals", dist.Poisson(lambd_total))

		# Log measures of unconstrained demand
		avl_idx = jnp.where(available_stock>0, 1, 0)
		p_j = jax.nn.softmax(utility, where=avl_idx, initial=0)
		_ = numpyro.deterministic("unconstrained_demand", self.total_rental_rate * p_j)
		_ = numpyro.sample("unconstrained_rentals", dist.Multinomial(self.total_rental_rate, p_j))

		# Simulate observed rentals from a censored multinomial distribution
		rentals = numpyro.deterministic("rentals", 
			RentalInventory.censored_multinomial(n=total_rentals, U_j=utility, stock_j=available_stock)
		)
		rentals_as_arr = ( time == jnp.arange(self.max_periods) )*rentals[:,None]
		return rentals_as_arr.astype(int)