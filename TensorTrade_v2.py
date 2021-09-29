"""Don't forget to change Environments! - conda activate dlenv """
#Step 1: Create Trading environment
from tensortrade.exchanges.simulated import FBMExchange
import tensortrade
from tensortrade import exchanges

from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfit
from tensortrade.actions import DiscreteActions
from tensortrade.environments import TradingEnvironment

normalize_price = MinMaxNormalizer(["open", "high", "low", "close"])
difference = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalize_price, 
                                          difference])
exchange = FBMExchange(timeframe='1h',
                       base_instrument='BTC',
                       feature_pipeline=feature_pipeline)
reward_scheme = SimpleProfit()
action_scheme = DiscreteActions(n_actions=20, 
                                instrument_symbol='ETH/BTC')
environment = TradingEnvironment(exchange=exchange,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme,
                                 feature_pipeline=feature_pipeline)


#Step 2: Create the learning Agent
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2
model = PPO2
policy = MlpLnLstmPolicy
params = { "learning_rate": 1e-5 }

#Step 3: Create a Trading Strategy by plugging in the agent and enviroment we just created
from tensortrade.strategies import StableBaselinesTradingStrategy

strategy = StableBaselinesTradingStrategy(environment=environment,
                                          model=model,
                                          policy=policy,
                                          model_kwargs=params)

#Step 4: Train the strategy
performance = strategy.run(steps=100000,
                           episode_callback=stop_early_callback)
                           