import multiprocessing as mp
import os
import pandas as pd
import numpy as np
import time

from system import Input, Output
from data import Data
from strategy import Strategy


def _process_position(args):

    symbol, strategy = args

    prices = Data.util.read_from_parquet('market_close')[symbol]

    if strategy['type'] == 'sma':
        short = strategy['short']
        long = strategy['long']
        position = Strategy.trader.sma_position(symbol, short, long, prices)

        short = str(short).zfill(3)
        long = str(long).zfill(3)
        msg = f'{symbol:<7} {"sma":<3} {short:<3} {long:<3} ' f'{position:=2}'

    elif strategy['type'] == 'momentum':
        momentum = strategy['momentum']
        position = Strategy.trader.momentum_position(symbol, momentum, prices)

        momentum = str(momentum).zfill(3)
        msg = f'{symbol:<7} {"m":<3} {momentum:<7} {position:=2}'

    elif strategy['type'] == 'mean_reversion':
        sma = strategy['sma']
        thr = strategy['thr']
        strategy_func = Strategy.trader.mean_reversion_position
        position = strategy_func(symbol, sma, thr, prices)

        sma = str(sma).zfill(3)
        thr = str(thr).zfill(3)
        msg = f'{symbol:<7} {"mr":<3} {sma:<3} {thr:<3} {position:=2}'

    elif strategy['type'] == 'linear_regression':
        lag = strategy['lag']
        strategy_func = Strategy.trader.linear_regression_position
        position = strategy_func(symbol, lag, prices)

        lag = str(lag).zfill(3)
        msg = f'{symbol:<7} {"lr":<3} {lag:<7} {position:=2}'

    Output.log.message(msg)

    return {'symbol': symbol, 'position': position}


class Simulation:

    def initialize(self):
        Data.download.stocks(init=True)
        os.makedirs('data/csv')
        os.makedir('data/html')
        os.makedir('data/json')
        os.makedir('data/parquet')
        os.makedir('data/parquet/lag')

    def log_account_status(self):
        total = Strategy.account.balance
        Output.log.message(f'capital: {total:>10.2f}')
        for position in Strategy.account.data['portfolio']:
            symbol = position['symbol']
            equity, cost_basis, units = Strategy.account.get_position(symbol)
            cost_basis_per_share = cost_basis / units
            msg = (f'{symbol:>7}: {equity:>10.2f} '
                   f'({cost_basis_per_share:.2f} x {units})')
            Output.log.message(msg)
            total += equity
        Output.log.message(f'  total: {total:>10.2f}')

    def download_data(self):
        Output.log.message('extracting data')
        Data.download.stocks(init=False)

    def transform_data(self):
        Output.log.message('transforming data')
        Data.transform.stocks()

    def set_strategies(self):
        Output.log.message('applying optimal strategies')

        unique_symbols = set()

        if Input.shell.args.momentum:
            momentum_strategies = Strategy.optimizer.momentum()
            unique_symbols = \
                unique_symbols.union(set(momentum_strategies.keys()))

        if Input.shell.args.sma:
            sma_strategies = Strategy.optimizer.sma()
            unique_symbols = unique_symbols.union(set(sma_strategies.keys()))

        if Input.shell.args.mean_reversion:
            mean_reversion_strategies = Strategy.optimizer.mean_reversion()
            unique_symbols = \
                unique_symbols.union(set(mean_reversion_strategies.keys()))

        if Input.shell.args.linear_regression:
            linear_regression_strategies = \
                Strategy.optimizer.linear_regression()
            unique_symbols = \
                unique_symbols.union(set(linear_regression_strategies.keys()))

        # remove expired strategies
        existing_strategies = {s['symbol']
                               for s in Strategy.account.strategies}

        for symbol in existing_strategies:
            if symbol not in unique_symbols:
                updated_strategies = [s for s in Strategy.account.strategies
                                      if s['symbol'] != symbol]
                Strategy.account.strategies = updated_strategies

        # save the new set of strategies
        for symbol in unique_symbols:
            strategies = list()

            if Input.shell.args.momentum:
                strategies.append(momentum_strategies.get(symbol))

            if Input.shell.args.sma:
                strategies.append(sma_strategies.get(symbol))

            if Input.shell.args.mean_reversion:
                strategies.append(mean_reversion_strategies.get(symbol))

            if Input.shell.args.linear_regression:
                strategies.append(linear_regression_strategies.get(symbol))

            strategies = [s for s in strategies if s]
            strategies = {'symbol': symbol,
                          'strategies': strategies}
            current_strategies = [s for s in Strategy.account.strategies
                                  if s['symbol'] != symbol]
            current_strategies.append(strategies)
            Strategy.account.strategies = current_strategies

    def strategize(self):
        Output.log.message('running strategies')

        linear_reg_strategies = list()
        other_strategies = list()
        for strategy in Strategy.account.strategies:
            symbol, strategy = strategy['symbol'], strategy['strategies']
            if len(strategy) > 1:
                strategy = max(strategy, key=lambda s: s['operf'])
            else:
                strategy = strategy[0]

            if strategy['type'] == 'linear_regression':
                linear_reg_strategies.append((symbol, strategy))
            else:
                other_strategies.append((symbol, strategy))

        # process non-linear regression strategies with a process pool
        with mp.Pool() as pool:
            positions = list()

            kwargs = {'func': _process_position,
                      'iterable': other_strategies,
                      'chunksize': 16}

            for position in pool.imap_unordered(**kwargs):
                positions.append(position)

        for symbol, strategy in linear_reg_strategies:
            positions.append(_process_position((symbol, strategy)))

        Strategy.account.data['positions'] = positions

    def apply_positions(self):
        Output.log.message('applying positions')

        positions = pd.DataFrame(Strategy.account.data['positions'])

        sales = positions[positions['position'] == -1][['symbol']]
        purchases = positions[positions['position'] == 1][['symbol']]

        Output.log.message('processing sales')
        kwargs = {'func': lambda s: Strategy.account.get_position(s['symbol']),
                  'axis': 1, 'result_type': 'expand'}

        current_positions = sales.apply(**kwargs)
        sales = pd.concat([sales, current_positions], axis=1)
        sales.columns = ('symbol', 'equity', 'cost_basis', 'units')
        sales = sales.dropna()

        current_positions = purchases.apply(**kwargs)
        purchases = pd.concat([purchases, current_positions], axis=1)
        purchases = \
            np.where(purchases[0].isna(), purchases['symbol'], np.nan)
        purchases = pd.Series(purchases).dropna()

        for _, (symbol, equity, cost_basis, units) in sales.iterrows():
            price = Data.download.current_price(symbol)

            if cost_basis / units >= price:
                continue

            profit = equity - cost_basis
            Strategy.account.data['profit'] += profit

            msg = (f'selling {units} units of {symbol} '
                   f'at {price:.2f}/share (${profit:.2f})')
            Output.log.message(msg)

            # place the sale
            Strategy.account.balance += equity
            updated_portfolio = [s for s in Strategy.account.portfolio
                                 if s != symbol]

            Strategy.account.portfolio = updated_portfolio

        # maximum allowable amount per purchase
        Output.log.message('assessing risk')
        purchase_budget = Strategy.account.balance / 20
        session_budget = Strategy.account.balance / 10

        daily_returns = Data.util.read_from_parquet('returns_daily')

        def kelly_func(symbol):
            return Strategy.risk.get_kelly_criterion(symbol, daily_returns)

        kelly_scores = purchases.apply(kelly_func)

        purchases = pd.concat([purchases, kelly_scores], axis=1)
        purchases.columns = ('symbol', 'kelly_score')
        purchases = purchases.sort_values('kelly_score', ascending=False)

        Output.log.message('processing purchases')
        for _, (symbol, kelly_score) in purchases.iterrows():
            if 0 >= session_budget:
                break

            riskable_capital = Strategy.risk.parse_kelly_score(kelly_score)
            units = riskable_capital * purchase_budget
            units = 1 if 1 > units >= 0.5 else int(units)

            if units == 0:
                continue

            price = Data.download.current_price(symbol)
            if price == np.nan:
                continue

            total_price = units * price
            if 0 >= session_budget - total_price:
                continue

            msg = (f'bought {units} units of {symbol} '
                   f'at ${price:.2f}/share (${total_price:.2f}, '
                   f'{riskable_capital:.1%})')
            Output.log.message(msg)

            # buy the stocks
            Strategy.account.balance -= total_price
            session_budget -= total_price

            unit_data = {'price': price, 'quantity': units,
                         'date': int(time.time())}

            try:
                portfolio_record = Strategy.account.portfolio[symbol]
                portfolio_record['shares'].append(unit_data)
            except KeyError:
                portfolio_record = {'symbol': symbol, 'shares': [unit_data]}

            Strategy.account.data['portfolio'].append(portfolio_record)


class Main:
    simulation = Simulation()


def run_simulation():
    try:
        Input.shell.args.download and Data.download.stocks()
        Input.shell.args.transform and Data.transform.stocks()
        Input.shell.args.optimize and Main.simulation.set_strategies()
        Input.shell.args.strategize and Main.simulation.strategize()
        Input.shell.args.apply and Main.simulation.apply_positions()
        Input.shell.args.print and Main.simulation.log_account_status()
        (not Input.shell.args.rollback) and Strategy.account.save_account()
    except KeyboardInterrupt:
        Output.log.message('keyboard interrupt detected - exiting')
    except Exception as e:
        Output.log.message(e)

        if Input.shell.args.verbose:
            raise


if __name__ == '__main__':
    run_simulation()
