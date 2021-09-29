import tensortrade.env.default as default

from tensortrade.feed.core import Stream, DataFeed
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH, LTC

#Load Data from exchanges

cdd = CryptoDataDownload()
bitfinex_btc = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
bitfinex_eth = cdd.fetch("Bitfinex", "USD", "ETH", "1h")

bitstamp_btc = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
bitstamp_eth = cdd.fetch("Bitstamp", "USD", "ETH", "1h")
bitstamp_ltc = cdd.fetch("Bitstamp", "USD", "LTC", "1h")


#Inspect transactions
bitfinex = Exchange("bitfinex", service=execute_order)(
    Stream.source(list(bitfinex_btc['close'][-100:]), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitfinex_eth['close'][-100:]), dtype="float").rename("USD-ETH")
)

bitstamp = Exchange("bitstamp", service=execute_order)(
    Stream.source(list(bitstamp_btc['close'][-100:]), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitstamp_eth['close'][-100:]), dtype="float").rename("USD-ETH"),
    Stream.source(list(bitstamp_ltc['close'][-100:]), dtype="float").rename("USD-LTC")
)

portfolio = Portfolio(USD, [
    Wallet(bitfinex, 10000 * USD),
    Wallet(bitfinex, 10 * BTC),
    Wallet(bitfinex, 5 * ETH),
    Wallet(bitstamp, 1000 * USD),
    Wallet(bitstamp, 5 * BTC),
    Wallet(bitstamp, 20 * ETH),
    Wallet(bitstamp, 3 * LTC)
])

feed = DataFeed([
    Stream.source(list(bitstamp_eth['volume'][-100:]), dtype="float").rename("volume:/USD-ETH"),
    Stream.source(list(bitstamp_ltc['volume'][-100:]), dtype="float").rename("volume:/USD-LTC")
])

env = default.create(
    portfolio=portfolio,
    action_scheme=default.actions.SimpleOrders(),
    reward_scheme=default.rewards.SimpleProfit(),
    feed=feed
)

done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

print(portfolio.ledger.as_frame().head(7))

#Inspect Transactions of ManagedRiskOrders
portfolio = Portfolio(USD, [
    Wallet(bitfinex, 10000 * USD),
    Wallet(bitfinex, 0 * BTC),
    Wallet(bitfinex, 0 * ETH),
])

env = default.create(
    portfolio=portfolio,
    action_scheme=default.actions.ManagedRiskOrders(),
    reward_scheme=default.rewards.SimpleProfit(),
    feed=feed
)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

print(portfolio.ledger.as_frame().head(20))