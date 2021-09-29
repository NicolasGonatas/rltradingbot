from sys import float_repr_style
from gym.spaces import Discrete
from numpy import testing
from numpy.core.fromnumeric import size, take
from numpy.lib.npyio import save
#from tensortrade.agents import agent
from tensortrade.agents.dqn_agent import DQNAgent
from tensortrade.env import default
import os
import tensorflow as tf
from tensortrade.env.default.actions import TensorTradeActionScheme, ManagedRiskOrders, BSH

from tensortrade.data.cdd import CryptoDataDownload

from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.core import Clock
from tensortrade.oms.exchanges.exchange import Exchange
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.orders import (Order, proportion_order,TradeSide,TradeType)
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio
from tensortrade.oms.wallets.wallet import Wallet

from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnRewardThreshold
from stable_baselines3.ppo import PPO
from torch._C import device
#Data
cdd = CryptoDataDownload()
data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")

#Create features with the feed module
def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal

features = []
for c in data.columns[1:]:
    s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
    features += [s]

closing_price = Stream.select(features, lambda s: s.name == "close")

features = [
    closing_price.log().diff().rename("lr"),
    rsi(closing_price, period=20).rename("rsi"),
    macd(closing_price, fast=10, slow=50, signal=5).rename("macd")]

feed = DataFeed(features)

feed.compile()
#Exchange
bitfinex = Exchange("bitfinex", service=execute_order)(Stream.source(list(data["close"]), dtype="float").rename("USD-BTC"))

#Portfolio
portfolio = Portfolio(USD,[
    Wallet(bitfinex, 1000*USD), 
    Wallet(bitfinex, 1 * BTC)

] )

renderer_feed = DataFeed([
    Stream.source(list(data["date"])).rename("date"),
    Stream.source(list(data["open"]), dtype="float").rename("open"),
    Stream.source(list(data["high"]), dtype="float").rename("high"),
    Stream.source(list(data["low"]), dtype="float").rename("low"),
    Stream.source(list(data["close"]), dtype="float").rename("close"),
    Stream.source(list(data["volume"]), dtype="float").rename("volume")
])

#actions = ManagedRiskOrders(stop= [0.02,0.04,0.06],take = [0.01,0.02,0.03],trade_sizes='4',)

env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed = feed,
    renderer_feed = renderer_feed,
    renderer=default.renderers.MatplotlibTradingChart(),#save_format = 'png', path = 'charts'
    window_size=20
)
obs = env.action_space.sample()
#print("Obs: ", obs)

agent = PPO('MlpPolicy', env, verbose = 1,device = 'auto',tensorboard_log='./PPO_Tensorboard')
#open conda prompt & type in 'code'


agent.learn(total_timesteps=200000)
#agent = DQNAgent(env)
#agent.train(n_steps=0.8*len(data), n_episodes=2, save_path=save_model_path)
print("Predict Obs: ",agent.predict(observation=env.reset()))

#log_path = os.path.join('Training', 'Logs')
#training_log_path = os.path.join(log_path, 'PPO')
save_model_path = os.path.join('Training','Saved_Models','PPO')



agent.save(save_model_path)
del agent
agent = PPO.load(save_model_path, env=env)


#Test Model
# episodes = 5

# for episode in range(1,episodes +1):
#     obs = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action, _states = agent.predict(obs)
#         obs, reward, done, info = env.step(action)
#         score += reward
#     print("Episone: {} Score: {} ".format(episode, score))
# env.close()


#Callbacks
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
eval_callback = EvalCallback(env, 
                             callback_on_new_best=stop_callback, 
                             eval_freq=10000, 
                             best_model_save_path=save_model_path, 
                             verbose=1)