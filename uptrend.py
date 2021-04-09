import zipline.api as algo
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import AverageDollarVolume

from zipline.pipeline import Pipeline

from zipline.api import (
    order_target_percent,
    order_target_value,
    order_target,
    record,
    set_commission,
    set_slippage,
    date_rules,
    time_rules
)

from zipline.pipeline.data import USEquityPricing, EquityPricing, master

# import any built-in factors and filters being used  
from zipline.pipeline.filters import StaticAssets  
from zipline.pipeline.factors import SimpleMovingAverage as SMA  
from zipline.pipeline.factors import CustomFactor, Returns

# import any needed datasets  
from zipline.pipeline.data.sharadar import Fundamentals as shfd
from zipline.pipeline.data.reuters import Financials as refun
from zipline.pipeline.data.reuters import Estimates as reest

from zipline.pipeline.factors import  AnnualizedVolatility
from zipline.pipeline.factors import SimpleMovingAverage, AverageDollarVolume, Latest
from zipline.pipeline.filters import AllPresent, All

from zipline.pipeline.data.master import SecuritiesMaster

from zipline.research import run_pipeline
import numpy as np
import scipy.stats as stats

import pandas as pd

from zipline.pipeline.data.sharadar import Fundamentals as shfd

import logging
from quantrocket.flightlog import FlightlogHandler
import datetime

logger = logging.getLogger('chris_uptrend')
logger.setLevel(logging.INFO)
handler = FlightlogHandler()
logger.addHandler(handler)

def _slope(ts, x=None):
    if x is None:
        x = np.arange(len(ts))
    log_ts = np.log(ts)
    # print(log_ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    return (slope, r_value, intercept, p_value, std_err)

class MomentumScore(CustomFactor):
    """
    12 months Momentum
    Run a linear regression over one year (momentum_window trading days) stocks log returns
    and the slope will be the factor value
    """
    inputs = [USEquityPricing.close]
    params = {'annualization_factor': 252}
    window_length = 125
           
    def compute(self, today, assets, out, close, annualization_factor):
        x = np.arange(len(close))
        # print(assets)
        results = np.apply_along_axis(_slope, 0, close, x.T)
        slopes = results[0,:]
        r_values = results[1,:]
        # Annualize percent
        annualized_slopes = (np.power(np.exp(slopes), annualization_factor) - 1) * 100
        # Adjust for fitness
        score = np.multiply(annualized_slopes, (r_values ** 2))
        out[:] = score

def TradableStocksUS(sector):
    # Equities listed as common stock (not preferred stock, ETF, ADR, LP, etc)
    common_stock = master.SecuritiesMaster.usstock_SecurityType2.latest.eq('Common Stock') 

    # Filter for primary share equities; primary shares can be identified by a
    # null usstock_PrimaryShareSid field (i.e. no pointer to a primary share)
    is_primary_share = master.SecuritiesMaster.usstock_PrimaryShareSid.latest.isnull()

    in_sector = master.SecuritiesMaster.usstock_Sector.latest.eq(sector)

    # combine the security type filters to begin forming our universe
    tradable_stocks = common_stock & is_primary_share & in_sector

    # also require high dollar volume
    tradable_stocks = AverageDollarVolume(window_length=200, mask=tradable_stocks).percentile_between(90, 100)

    # also require price > $5. Note that we use Latest(...) instead of EquityPricing.close.latest
    # so that we can pass a mask
    tradable_stocks = Latest([USEquityPricing.close], mask=tradable_stocks) > 10
    
    # also require no missing data for 200 days
    tradable_stocks = AllPresent(inputs=[USEquityPricing.close], window_length=200, mask=tradable_stocks)
    tradable_stocks = All([USEquityPricing.volume.latest > 0], window_length=200, mask=tradable_stocks)
    
    return tradable_stocks

def initialize(context):
    """
    Called once at the start of a backtest, and once per day at
    the start of live trading. In live trading, the stored context
    will be loaded *after* this function is called.
    """
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.month_start(),
        algo.time_rules.market_open(hours=1),
    )

    algo.schedule_function(
        bonds,
        algo.date_rules.month_start(days_offset=1),
        algo.time_rules.market_open(hours=1),
    )

    algo.set_benchmark(algo.sid("FIBBG000BDTBL9"))

    # Create a pipeline to select stocks each day.
    algo.attach_pipeline(make_pipeline(), 'pipeline')
    
    # algo.set_min_leverage(0, datetime.timedelta(30))
    # algo.set_max_leverage(1.2)

    context.trend_filter = False


def make_pipeline():
    """
    Create a pipeline to select stocks each day.
    """
    universe = TradableStocksUS('Real Estate') | TradableStocksUS('Utilities') |                     \
        TradableStocksUS('Consumer Staples') | TradableStocksUS('Technology') |                      \
        TradableStocksUS('Financials') | TradableStocksUS('Energy') |                                \
        TradableStocksUS('Materials') | TradableStocksUS('Health Care') |                            \
        TradableStocksUS('Industrials') | TradableStocksUS('Consumer Discretionary') |               \
        TradableStocksUS('Communications')

    roic = shfd.slice(dimension='MRT', period_offset=0).ROIC.latest
    ebit = shfd.slice(dimension='MRQ', period_offset=0).EBIT.latest
    ev = shfd.slice(dimension='MRQ', period_offset=0).EV.latest
    volatility = AnnualizedVolatility(window_length=100)
    value = ebit / ev

    roic_rank = roic.rank(mask=universe)
    value_rank = value.rank(mask=universe)
    volatility_rank = volatility.rank(mask=universe, ascending=False)

    spy_ma100_price = SMA(inputs=[USEquityPricing.close],  
                          window_length=100)[algo.sid("FIBBG000BDTBL9")]
    spy_price       = USEquityPricing.close.latest[algo.sid("FIBBG000BDTBL9")]

    momentum_score = MomentumScore()

    overall_rank = roic_rank + value_rank + volatility_rank

    # seven_month_returns = Returns(window_length=148, mask=universe,)
    # one_month_returns = Returns(window_length=30, mask=universe,)

    pipeline = Pipeline(
        columns={
            'stock' : master.SecuritiesMaster.Symbol.latest,
            'sid': master.SecuritiesMaster.Sid.latest,
            'sector' : master.SecuritiesMaster.usstock_Sector.latest,
            'average_dollar_volume': AverageDollarVolume(window_length=200),
            'price': EquityPricing.close.latest,
            'volume': EquityPricing.volume.latest,
            'roic' : roic,
            'value' : value,
            'volatility': volatility,
            'roic_rank' : roic_rank,
            'value_rank' : value_rank,
            'momentum': momentum_score,
            'momentum_decile': momentum_score.deciles(),
            'volatility_decile' : volatility.deciles(),
            'overall_rank' : overall_rank,
            'overall_rank_decile': overall_rank.deciles(),
            'trend_filter': spy_price > spy_ma100_price,
            # 'returns' : one_month_returns - seven_month_returns
        },                                                 
        screen = universe
    )

    return pipeline

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = algo.pipeline_output('pipeline')

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index

def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    logger.debug('rebalancing on: %s', algo.get_datetime())

    context.trend_filter = False

    # new_portfolio = algo.pipeline_output('pipeline').dropna(subset=['overall_rank']).sort_values('momentum', ascending=False)

    new_portfolio = algo.pipeline_output('pipeline').dropna(subset=['overall_rank']).sort_values('momentum', ascending=False)

    for equity, row in new_portfolio.iterrows():
        logger.debug('new portfolio (before filtering) - equity: %s', equity)

    # print(new_portfolio)

    # new_portfolio = new_portfolio[new_portfolio['overall_rank'].notna() & new_portfolio['momentum'] > 40][:20]
    
    # new_portfolio = new_portfolio[(new_portfolio['momentum_decile'] > 8)][:20]

    new_portfolio = new_portfolio.nlargest(20, ['overall_rank', 'momentum'])    #<- $600K PL in 10 years

    # new_portfolio = new_portfolio.nlargest(20, ['momentum', 'overall_rank'])  #<- 1M PL in 10 years

    if logger.level is logging.DEBUG:
        for equity, row in new_portfolio.iterrows():
            logger.debug('new portfolio - (after filtering) equity: %s', equity)
    

    # print(len(new_portfolio.index))

    # volatility driven weights
    # new_portfolio['inverse_volatility'] = new_portfolio['volatility'].apply(lambda x: 1 / x)
    # inv_vola_sum = new_portfolio['inverse_volatility'].sum()
    # new_portfolio['target_weight'] =  new_portfolio['inverse_volatility'].apply(lambda x: x / inv_vola_sum)

    # portfolio size driven weights
    # num_equities = len(new_portfolio.index)
    # new_portfolio['target_weight'] =  1 / num_equities\

    # logger.info('len existing portfolio: %s', len(context.portfolio.positions))

    if logger.level is logging.DEBUG:
        for equity, values in context.portfolio.positions.items():
            logger.debug('context.portfolio.positions - equity: %s, amount: %s, cost_basis: %s, sold_on: %s, sold_at_price: %s', equity, values.amount, values.cost_basis, values.last_sale_date, values.last_sale_price)

    
    order_target(algo.sid('FIBBG000NTFYM5'), 0)
    logger.debug('selling all bonds')

    for equity in context.portfolio.positions:
        if equity is algo.sid('FIBBG000NTFYM5'): 
            continue
        if equity not in set(new_portfolio.index.tolist()):
            # logger.info('selling %s', equity)
            order_target_percent(equity, 0)

    stock_weights = 1.0 / max(len(context.portfolio.positions), len(new_portfolio.index))

    logger.debug('len existing portfolio (afer ejection): %s', len(context.portfolio.positions))
    logger.debug('len new portfolio: %s', len(new_portfolio.index))
    logger.debug('stock_weights: %s', stock_weights)

    # print(context.portfolio.positions.get(algo.sid('FIBBG000NTFYM5')))

    # spy = context.portfolio.positions.get(algo.sid('FIBBG000NTFYM5'))

    # if (spy is not None) and (spy.amount > 0):
    #     order_target_percent(algo.sid('FIBBG000NTFYM5'), 0)

    for equity, row in new_portfolio.iterrows():
        if row.trend_filter is True:
            # logger.info('buying %s', equity)
            context.trend_filter = True
            order_target_percent(equity, stock_weights)
        else:
            context.trend_filter = False
            
    logger.debug('cash: %s', context.portfolio.cash)
    logger.debug('portfolio_value: %s', context.portfolio.portfolio_value)
    logger.debug('num_positions: %s', len(context.portfolio.positions))
    logger.debug('positions: %s', context.portfolio.positions)

def bonds(context, data):
    logger.debug('buying bonds on: %s', algo.get_datetime())
    logger.debug('num open orders: %s', len(algo.get_open_orders()))
    logger.debug('len existing portfolio (afer ejection): %s', len(context.portfolio.positions))
    logger.debug('cash: %s', context.portfolio.cash)
    logger.debug('portfolio_value: %s', context.portfolio.portfolio_value)
    logger.debug('num_positions: %s', len(context.portfolio.positions))
    logger.debug('positions: %s', context.portfolio.positions)

    if logger.level is logging.DEBUG:
        for equity, values in context.portfolio.positions.items():
            logger.debug('context.portfolio.positions - equity: %s, amount: %s, cost_basis: %s, sold_on: %s, sold_at_price: %s', equity, values.amount, values.cost_basis, values.last_sale_date, values.last_sale_price)

    if context.portfolio.cash > 0 and context.trend_filter is False:
        logger.debug('converting all cash to bonds') 
        order_target_value(algo.sid('FIBBG000NTFYM5'), context.portfolio.cash)
