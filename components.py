import multiprocessing as mp
import itertools as it
import threading as t
import configparser
import pandas as pd
import numpy as np
import requests
import zipfile
import queue
import json
import time
import bs4
import os
import re

from system import Output
from const import Account, Periods, Config

config = configparser.ConfigParser()
config.read(Config.path)


class UtilComponent:
    @staticmethod
    def write_to_parquet(df, name):
        path = f'data/parquet/{name}.parquet.gzip'
        df.to_parquet(path, compression='gzip')

    @staticmethod
    def read_from_parquet(name):
        return pd.read_parquet(f'data/parquet/{name}.parquet.gzip')

    @staticmethod
    def convert_dense_to_sparse(df):
        new_df = df.apply(lambda s: pd.Series(pd.arrays.SparseArray(s)))
        new_df.index = df.index
        return new_df

    @staticmethod
    def convert_sparse_to_dense(df):
        new_df = df.apply(lambda s: s.values.to_dense())
        new_df.index = df.index
        return new_df


class DownloadComponent:
    util = UtilComponent()

    @staticmethod
    def current_price(symbol):
        marketwatch_endpoint = ('https://www.marketwatch.com/'
                                f'investing/stock/{symbol}')

        try:
            marketwatch_response = requests.get(marketwatch_endpoint)
        except requests.exceptions.ConnectionError:
            return np.nan

        marketwatch_html = marketwatch_response.text
        marketwatch_soup = bs4.BeautifulSoup(marketwatch_html, 'html.parser')

        quote_channel_pattern = (r'(\/zigman2\/quotes\/\d+\/composite,'
                                 r'\/zigman2\/quotes\/\d+\/lastsale|'
                                 r'\/zigman2\/quotes\/\d+\/delayed)')
        quote_attributes = {
            'field': 'Last',
            'channel': re.compile(quote_channel_pattern)
        }

        try:
            ticker = marketwatch_soup.find_all('bg-quote', quote_attributes)
            price = float(ticker[0].text.replace(',', ''))
        except AttributeError:
            price = np.nan
        except IndexError:
            try:
                intraday_price_attr = {'class': 'intraday__price'}
                close_price = marketwatch_soup.find('h3', intraday_price_attr)
                close_price = close_price.find('span', {'class': 'value'})
                price = float(close_price.text.replace(',', ''))
            except AttributeError:
                price = np.nan

        return price

    @staticmethod
    def stock_exists(symbol):
        endpoint = ('https://www.marketwatch.com/tools/quotes/lookup.asp?'
                    f'siteID=mktw&Lookup={symbol}&Country=us&Type=All')

        response = requests.get(endpoint)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')

        title = soup.find('title').text

        return title == 'Stock Ticker Symbol Lookup - MarketWatch'

    @staticmethod
    def nyse_list():
        page_number = 1
        records = list()

        total_records = 1

        while total_records > len(records):

            response = requests.post(
                'https://www.nyse.com/api/quotes/filter',
                json={
                    'instrumentType': 'EQUITY',
                    'pageNumber': page_number,
                    'sortColumn': 'NORMALIZED_TICKER',
                    'sortOrder': 'ASC',
                    'maxResultsPerPage': 1000,
                    'filterToken': '',
                },
            )

            response_data = json.loads(response.text)
            total_records = response_data[0]['total']

            records += list(response_data)
            page_number += 1

        nyse_df = pd.DataFrame(records)
        nyse_df.to_csv(Config.nyse, index=False)

    @staticmethod
    def quotemedia_list():
        quandl_key = config['quandl']['api_key']
        endpoint = (f'https://www.quandl.com/api/v3/databases/EOD/metadata?'
                    f'api_key={quandl_key}')

        zipped_data = requests.get(endpoint)
        with open(Config.quotemedia + '.zip', 'wb') as zip_file:
            zip_file.write(zipped_data.content)

        with zipfile.ZipFile(Config.quotemedia + '.zip') as zip_file:
            zip_file.extract('EOD_metadata.csv', 'data/csv/')

        os.remove(Config.quotemedia)
        os.rename('data/csv/EOD_metadata.csv', Config.quotemedia)

    @staticmethod
    def stock_data(symbol):
        symbol = symbol.replace('-', '_').replace('.', '_')
        quandl_key = config['quandl']['api_key']
        endpoint = (f'https://www.quandl.com/api/v3/datasets/EOD/{symbol}'
                    f'?api_key={quandl_key}')

        quandl_response = requests.get(endpoint)

        accumulated_wait = 0
        while quandl_response.status_code == 429:
            time.sleep(30)
            accumulated_wait += 0.5

            quandl_response = requests.get(endpoint)

            if accumulated_wait == 11:
                break

        quandl_json = json.loads(quandl_response.text)

        if 'quandl_error' in quandl_json:
            quandl_json = {'dataset': {'column_names': [], 'data': []}}

        dataset = quandl_json['dataset']
        history_df = pd.DataFrame(dataset['data'])

        if len(history_df) > 0:
            history_df.columns = [c.lower() for c in dataset['column_names']
                                  if len(history_df) > 0]
            history_df.set_index('date')
            history_df.index = pd.DatetimeIndex(history_df.index)

        return history_df

    def quotemedia_history(self, all_companies=False):
        history_list = list()
        thread_count = os.cpu_count() * 3
        thread_count = int(thread_count)
        symbol_queue = queue.PriorityQueue(thread_count * 1.5)

        def _thread_func():
            keeping_threading = True
            while keeping_threading:
                priority, symbol = symbol_queue.get()

                if priority != 0:
                    try:
                        history = self.stock_data(symbol)
                        if len(history) != 0:
                            history['symbol'] = symbol
                            history_list.append(history)
                    except Exception as e:
                        Output.log.message(e)
                        symbol_queue.task_done()
                        break

                keeping_threading = bool(priority)
                symbol_queue.task_done()

        Output.log.message('starting threads')
        for _ in range(thread_count):
            thread = t.Thread(target=_thread_func, daemon=True)
            thread.start()

        if all_companies:
            companies_df = pd.read_csv(Config.quotemedia, index_col=0)
            companies_df = companies_df[companies_df['refreshed_at'].notnull()]
            companies_df = pd.DataFrame(companies_df)
        else:
            companies_df = pd.read_csv(Config.nyse, index_col=5)
            companies_df.index = companies_df.index.str.replace('.', '_')
            companies_df.index = companies_df.index.str.replace('-', '_')

        try:
            last_symbol = str(companies_df.index[0])

            for priority, symbol in enumerate(companies_df.index):
                priority = -(priority + 1)
                if priority % 500 == 0:
                    Output.log.message((f'{last_symbol:>12} - {symbol:<12} '
                                        f'({len(history_list)} downloaded)'))
                    last_symbol = symbol

                symbol_queue.put((priority, symbol))

            symbol_queue.join()
        except KeyboardInterrupt:
            raise
        finally:
            Output.log.message('closing threads')
            for _ in range(thread_count):
                symbol_queue.put((0, None))

        Output.log.message('unioning datasets')
        history_df = pd.concat(history_list)

        Output.log.message('rebuilding index')
        history_df = history_df.reset_index()
        history_df = history_df.set_index(['symbol', 'date'])

        return history_df

    def stocks(self, init=False):
        Output.log.message('downloading NYSE list')
        self.nyse_list()

        Output.log.message('downloading Quotemedia list')
        self.quotemedia_list()

        Output.log.message('downloading Quotemedia history')
        history_df = self.quotemedia_history()

        Output.log.message('writing data')
        self.util.write_to_parquet(history_df, 'quotemedia_history')


class TransformComponent:
    util = UtilComponent()

    def create_process_function(self, process_queue):
        def process_lags():
            while True:
                symbol, period, data = process_queue.get()

                data = pd.Series(data)

                lag_df = pd.DataFrame()
                lags = (lag for lag in range(period + 1))

                for lag in lags:
                    lag_column = f'{symbol}_{lag}'
                    lag_df[lag_column] = data.shift(lag).values.to_dense()

                lag_df.index = data.index

                target_file = f'lag/{symbol}_{period}'

                self.util.write_to_parquet(lag_df, target_file)

                process_queue.task_done()

        return process_lags

    @staticmethod
    def quotemedia_descriptions():
        quotemedia_df = pd.read_csv(Config.quotemedia)

        html_text = ('<html>\n'
                     '<head>\n'
                     '\t<title>Quotemedia Stock Descriptions</title>\n'
                     '</head>\n'
                     '<body>\n')

        for desc in quotemedia_df['description']:
            if desc == 'This dataset has no description.':
                continue

            html_text += desc + '</br>\n'

        html_text += '</body>'

        with open(Config.quotemedia_desc, 'w') as file:
            file.write(html_text)

    def stocks(self):
        history_df = self.util.read_from_parquet('quotemedia_history')

        Output.log.message('building close price matrix')
        history_df['close'] = history_df['close'] * history_df['split']
        mkt_close = history_df[['close']].unstack('symbol')
        mkt_close = pd.DataFrame(mkt_close['close'])
        mkt_close.index = pd.DatetimeIndex(mkt_close.index)
        mkt_close = mkt_close.loc['1999-01-01':]
        self.util.write_to_parquet(mkt_close, 'market_close')

        mkt_close = self.util.convert_dense_to_sparse(mkt_close)

        Output.log.message('processing returns')
        # daily statistics
        daily_change = mkt_close - mkt_close.shift(1)
        daily_change = self.util.convert_sparse_to_dense(daily_change)
        daily_returns = daily_change.fillna(0).cumsum()
        self.util.write_to_parquet(daily_returns, 'returns_daily')

        del daily_change

        daily_returns = self.util.convert_dense_to_sparse(daily_returns)

        Output.log.message('processing momentum')
        # momentum - daily
        for period in Periods.momentum:
            momentum = np.sign(daily_returns.rolling(period).mean())
            period_df = pd.DataFrame(momentum)
            self.util.write_to_parquet(period_df, f'momentum_daily_{period}')

        Output.log.message('processing moving averages')
        # simple moving average - daily
        for period in Periods.sma:
            period_df = mkt_close.rolling(period).mean()
            self.util.write_to_parquet(period_df, f'sma_daily_{period}')

        del period_df

        Output.log.message('processing lagged returns')
        sorted_lags = sorted(Periods.lags, reverse=True)

        for period in sorted_lags:
            Output.log.message(f'processing {period}-day lags')
            process_queue = mp.JoinableQueue(9)

            process_func = self.create_process_function(process_queue)
            processes = list()
            for _ in range(3):
                process = mp.Process(target=process_func, daemon=True)
                process.start()
                processes.append(process)

            for symbol in daily_returns.columns:
                values = daily_returns[symbol]
                process_queue.put((symbol, period, values))

            process_queue.join()

            for process in processes:
                process.terminate()
                process.join()
                process.close()


class AccountComponent:
    util = UtilComponent()
    download = DownloadComponent()

    def __init__(self):
        Output.log.message('loading account')
        try:
            self.data = json.load(open(Account.path, 'r'))
        except FileNotFoundError:
            self.data = {}

    @property
    def balance(self):
        return self.data['balance']

    @property
    def strategies(self):
        return self.data['strategies']

    @property
    def portfolio(self):
        return {p['symbol']: p['shares'] for p in self.data['portfolio']}

    @balance.setter
    def balance(self, val):
        Output.log.message(f'setting balance: ${val:.2f}')
        self.data['balance'] = val

    @strategies.setter
    def strategies(self, val):
        self.data['strategies'] = val

    def init_account(self):
        Output.log.message('initializing portfolio')
        self.data = dict()

        self.data['date_modified'] = 0
        self.data['balance'] = 0
        self.data['profit'] = 0
        self.data['portfolio'] = []
        self.data['strategies'] = []
        self.data['positions'] = []

        self.save_account()

    def reset_portfolio(self):
        Output.log.message('resetting portfolio')
        self.data['portfolio'] = []

    def reset_strategies(self):
        Output.log.message('resetting strategies')
        self.data['strategies'] = []

    def reset_positions(self):
        Output.log.message('resetting positions')
        self.data['positions'] = []

    def save_account(self):
        Output.log.message('saving account')
        self.data['date_modified'] = int(time.time())
        json.dump(self.data, open(Account.path, 'w'))

    def get_position(self, symbol):
        portfolio = list(self.data['portfolio'])

        if len(portfolio) == 0:
            return None, None, None

        # search for position, exiting if not found
        while True:
            position = portfolio.pop(0)
            if position['symbol'] != symbol:
                if len(portfolio) == 0:
                    return None, None, None
                continue

            shares = position['shares']
            break

        equity = 0
        cost_basis = 0
        unit_quantity = 0

        current_price = self.download.current_price(symbol)
        if current_price == np.nan:
            raise Exception('current price is NaN')

        for unit in shares:
            returns = current_price - unit['price']
            returns = returns * unit['quantity']

            unit_quantity += unit['quantity']
            cost_basis += unit['price'] * unit['quantity']
            equity += unit['price'] * unit['quantity'] + returns

        return equity, cost_basis, unit_quantity


class RiskManagementComponent:
    util = UtilComponent()

    def get_kelly_criterion(self, symbol, daily_returns=None):
        if daily_returns is None:
            daily_returns = self.util.read_from_parquet('returns_daily')

        daily_returns = daily_returns[symbol]
        mu = daily_returns.mean()
        sigma = daily_returns.var()

        kelly_criterion = (mu / sigma**2)

        return kelly_criterion

    def get_kelly_criterion_multi(self, symbols):
        daily_returns = self.util.read_from_parquet('returns_daily')
        daily_returns = daily_returns[symbols]

        returns_mean = daily_returns.mean()
        returns_covariance = daily_returns.cov()

        kwargs = {'index': daily_returns.columns,
                  'columns': daily_returns.columns}

        precision = pd.DataFrame(np.linalg.inv(returns_covariance), **kwargs)
        kelly_criterion_percent = precision.dot(returns_mean)

        return kelly_criterion_percent

    @staticmethod
    def parse_kelly_score(kelly_criterion):
        if kelly_criterion > 1:
            riskable_capital = 0.10
        elif kelly_criterion > 0.5:
            riskable_capital = 0.05
        elif kelly_criterion > 0.1:
            riskable_capital = 0.025
        elif kelly_criterion > 0.01:
            riskable_capital = 0.01
        elif kelly_criterion > 0.005:
            riskable_capital = 0.005
        else:
            # don't chance it
            riskable_capital = 0.0

        return riskable_capital


class BacktestComponent:
    util = UtilComponent()

    @staticmethod
    def evaluate_performance(buy_signals, equity, daily_returns):
        market = daily_returns.iloc[-1]
        strategy = equity.iloc[-1]

        # custom failure logic
        lost_money = 0 > equity
        poor_performance = market > strategy

        for symbol in equity:
            symbol_daily_returns = daily_returns[symbol].dropna()
            trades_per_month = buy_signals[symbol].dropna().abs()
            trades_per_month = trades_per_month.resample('M').sum()

            # trades infrequently
            if 1 > trades_per_month.mean():
                strategy[symbol] = -np.inf

            # too short of a history (< 2 years)
            if 504 > len(symbol_daily_returns):
                strategy[symbol] = -np.inf

            # outperformed by the market
            if poor_performance[symbol]:
                strategy[symbol] = -np.inf

            # lost money
            if lost_money[symbol].any():
                strategy[symbol] = -np.inf

        return market, strategy

    def sma(self, short, long, data=None):
        if data is None:
            close_price = self.util.read_from_parquet('market_close')
            daily_returns = self.util.read_from_parquet('returns_daily')
        else:
            close_price = data['market_close']
            daily_returns = data['returns_daily']

        short_sma = self.util.read_from_parquet(f'sma_daily_{short}')
        long_sma = self.util.read_from_parquet(f'sma_daily_{long}')

        daily_returns = daily_returns.replace(0, np.nan)
        daily_returns = pd.DataFrame(np.log(daily_returns))

        returns_percent = daily_returns / daily_returns.shift(1)

        buy_signals = np.where(short_sma > long_sma, 1, -1)
        buy_signals = pd.DataFrame(buy_signals, columns=close_price.columns)
        buy_signals = buy_signals[buy_signals != buy_signals.shift(1)]
        buy_signals.index = close_price.index

        equity = buy_signals * close_price
        equity = equity.fillna(0).cumsum() * returns_percent

        performance_measures = (buy_signals, equity, daily_returns)
        market, strategy = self.evaluate_performance(*performance_measures)

        # "success" as defined by not failing the tests and getting slammed
        # with the dreaded -np.inf
        strategy_success = strategy.replace([np.inf, -np.inf], np.nan).dropna()
        market_success = market[strategy_success.index]

        absolute_performance = \
            strategy_success.median() - market_success.median()

        return strategy, strategy - market, absolute_performance

    def momentum(self, day_period, data=None):
        if data is None:
            close_price = self.util.read_from_parquet('market_close')
            daily_returns = self.util.read_from_parquet('returns_daily')
        else:
            close_price = data['market_close']
            daily_returns = data['returns_daily']

        momentum = self.util.read_from_parquet(f'momentum_daily_{day_period}')

        daily_returns = daily_returns.replace(0, np.nan)
        daily_returns = pd.DataFrame(np.log(daily_returns))

        returns_percent = daily_returns / daily_returns.shift(1)

        buy_signals = np.where(momentum == 1, 1, -1)
        buy_signals = pd.DataFrame(buy_signals, columns=close_price.columns)
        buy_signals = buy_signals[buy_signals != buy_signals.shift(1)]
        buy_signals.index = close_price.index

        equity = buy_signals * close_price
        equity = equity.fillna(0).cumsum() * returns_percent

        performance_measures = (buy_signals, equity, daily_returns)
        market, strategy = self.evaluate_performance(*performance_measures)

        # "success" as defined by not failing the tests and getting slammed
        # with a fat np.inf
        strategy_success = strategy.replace([np.inf, -np.inf], np.nan).dropna()
        market_success = market[strategy_success.index]

        absolute_performance = \
            strategy_success.median() - market_success.median()

        return strategy, strategy - market, absolute_performance

    def mean_reversion(self, sma_period, threshold, data=None):
        if data is None:
            close_price = self.util.read_from_parquet('market_close')
            daily_returns = self.util.read_from_parquet('returns_daily')
        else:
            close_price = data['market_close']
            daily_returns = data['returns_daily']

        sma = self.util.read_from_parquet(f'sma_daily_{sma_period}')

        daily_returns = daily_returns.replace(0, np.nan)
        daily_returns = pd.DataFrame(np.log(daily_returns))

        returns_percent = daily_returns / daily_returns.shift(1)

        distance = close_price - sma

        buy_signals = np.where(-threshold > distance, 1, np.nan)
        buy_signals = np.where(distance > threshold, -1, buy_signals)

        # if crossing from -distance to +distance, or vice versa,
        # hold until a sustained trend forms
        unsustained_trend = distance * distance.shift(1) < 0
        buy_signals = np.where(unsustained_trend, 0, buy_signals)

        buy_signals = pd.DataFrame(buy_signals, columns=close_price.columns)
        buy_signals = buy_signals[buy_signals != buy_signals.shift(1)]
        buy_signals.index = close_price.index

        equity = buy_signals * close_price
        equity = equity.fillna(0).cumsum() * returns_percent

        performance_measures = (buy_signals, equity, daily_returns)
        market, strategy = self.evaluate_performance(*performance_measures)

        # "success" as defined by not failing the tests and getting slammed
        # with a fat np.inf
        strategy_success = strategy.replace([np.inf, -np.inf], np.nan).dropna()
        market_success = market[strategy_success.index]

        absolute_performance = \
            strategy_success.median() - market_success.median()

        return strategy, strategy - market, absolute_performance

    def linear_regression(self, lag, data=None):
        if data is None:
            close_price = self.util.read_from_parquet('market_close')
            daily_returns = self.util.read_from_parquet('returns_daily')
        else:
            close_price = data['market_close']
            daily_returns = data['returns_daily']

        daily_returns = daily_returns.replace(0, np.nan)
        daily_returns = pd.DataFrame(np.log(daily_returns))

        returns_percent = daily_returns / daily_returns.shift(1)

        buy_signals_list = list()

        for symbol in close_price.columns:
            lagged_returns = self.util.read_from_parquet(f'lag/{symbol}_{lag}')
            lags = [f'{symbol}_{n}' for n in range(lag + 1)]
            lags_df = pd.DataFrame(lagged_returns[lags].dropna())

            train_percentage = int(len(lags_df.index) * 0.666)
            train_slice = lags_df.iloc[:train_percentage]
            test_slice = lags_df.iloc[train_percentage:]

            symbol_returns = \
                daily_returns[symbol].loc[train_slice.index].dropna()
            train_slice = train_slice.loc[symbol_returns.index]

            train_results = np.sign(symbol_returns)
            regression = np.linalg.lstsq(train_slice, train_results,
                                         rcond=None)[0]

            buy_signals = np.sign(np.dot(test_slice, regression))
            buy_signals = pd.DataFrame(buy_signals, columns=[symbol])
            buy_signals = buy_signals[buy_signals != buy_signals.shift(1)]
            buy_signals.index = test_slice.index

            buy_signals_list.append(buy_signals)

        buy_signals = pd.concat(buy_signals_list, axis=1)
        close_price = close_price.loc[buy_signals.index]
        returns_percent = returns_percent.loc[buy_signals.index]

        equity = buy_signals * close_price
        equity = equity.fillna(0).cumsum() * returns_percent

        performance_measures = (buy_signals, equity, daily_returns)
        market, strategy = self.evaluate_performance(*performance_measures)

        # "success" as defined by not failing the tests and getting slammed
        # with a fat np.inf
        strategy_success = strategy.replace([np.inf, -np.inf], np.nan).dropna()
        market_success = market[strategy_success.index]

        absolute_performance = \
            strategy_success.median() - market_success.median()

        return strategy, strategy - market, absolute_performance


class OptimizerComponent:
    util = UtilComponent()
    backtest = BacktestComponent()

    def sma(self):
        Output.log.message('optimizing simple moving average strategies')
        optimal_smas = dict()
        sma_permutations = it.permutations(Periods.sma, 2)
        sma_permutations = ((n, i) for n, i in sma_permutations if i > n)

        data = {'market_close': self.util.read_from_parquet('market_close'),
                'returns_daily': self.util.read_from_parquet('returns_daily')}

        nyse = pd.read_csv(Config.nyse, index_col=5)
        nyse.index = nyse.index.str.replace('.', '_', regex=False)
        nyse.index = nyse.index.str.replace('-', '_', regex=False)
        nyse = nyse.index

        for short, long in sma_permutations:
            _, operf, aperf = self.backtest.sma(short, long, data)
            aperf = np.array([aperf] * len(operf))

            performance = pd.DataFrame(zip(operf, aperf))
            performance.columns = ('operf', 'aperf')
            performance.index = operf.index

            for symbol in nyse:
                if symbol not in performance.index:
                    continue

                try:
                    operf = float(performance['operf'].loc[symbol])
                    aperf = int(performance['aperf'].loc[symbol])

                    if 0 > operf:
                        continue

                    if np.isnan(operf):
                        continue

                    # if an existing strategy is better performer, move on
                    if optimal_smas[symbol]['operf'] > operf:
                        continue

                    if optimal_smas[symbol]['aperf'] > aperf:
                        continue

                except KeyError:
                    # there wasn't an existing strategy
                    pass

                # either the prior strategies were more weak
                # or there weren't existing strategies, so
                # save this strategy
                optimal_smas[symbol] = {'type': 'sma',
                                        'short': short,
                                        'long': long,
                                        'operf': operf,
                                        'aperf': aperf}

        return optimal_smas

    def momentum(self):
        Output.log.message('optimizing momentum strategies')
        optimal_momentum = dict()

        data = {'market_close': self.util.read_from_parquet('market_close'),
                'returns_daily': self.util.read_from_parquet('returns_daily')}

        nyse = pd.read_csv(Config.nyse, index_col=5)
        nyse.index = nyse.index.str.replace('.', '_', regex=False)
        nyse.index = nyse.index.str.replace('-', '_', regex=False)
        nyse = nyse.index

        for period in Periods.momentum:
            _, operf, aperf = self.backtest.momentum(period, data)
            aperf = np.array([aperf] * len(operf))

            performance = pd.DataFrame(zip(operf, aperf))
            performance.columns = ('operf', 'aperf')
            performance.index = operf.index

            for symbol in nyse:
                if symbol not in performance.index:
                    continue

                try:
                    operf = float(performance['operf'].loc[symbol])
                    aperf = float(performance['aperf'].loc[symbol])

                    if 0 > operf:
                        continue

                    if np.isnan(operf):
                        continue

                    # if an existing strategy is better performer, move on
                    if optimal_momentum[symbol]['operf'] > operf:
                        continue

                    if optimal_momentum[symbol]['aperf'] > aperf:
                        continue

                except KeyError:
                    # there wasn't an existing strategy
                    pass

                # either the prior strategies were more weak
                # or there weren't existing strategies, so
                # save this strategy
                optimal_momentum[symbol] = {'type': 'momentum',
                                            'momentum': period,
                                            'operf': operf,
                                            'aperf': aperf}

        return optimal_momentum

    def mean_reversion(self):
        Output.log.message('optimizing mean reversion strategies')
        optimal_mean_reversion = dict()
        mean_reversion_permutations = ((n, i / 2) for n in Periods.sma
                                       for i in range(1, 20))

        data = {'market_close': self.util.read_from_parquet('market_close'),
                'returns_daily': self.util.read_from_parquet('returns_daily')}

        nyse = pd.read_csv(Config.nyse, index_col=5)
        nyse.index = nyse.index.str.replace('.', '_', regex=False)
        nyse.index = nyse.index.str.replace('-', '_', regex=False)
        nyse = nyse.index

        for sma, thr in mean_reversion_permutations:
            _, operf, aperf = self.backtest.mean_reversion(sma, thr, data)
            aperf = np.array([aperf] * len(operf))

            performance = pd.DataFrame(zip(operf, aperf))
            performance.columns = ('operf', 'aperf')
            performance.index = operf.index

            for symbol in nyse:
                if symbol not in performance.index:
                    continue

                try:
                    operf = float(performance['operf'].loc[symbol])
                    aperf = int(performance['aperf'].loc[symbol])

                    if 0 > operf:
                        continue

                    if np.isnan(operf):
                        continue

                    # if an existing strategy is better performer, move on
                    if optimal_mean_reversion[symbol]['operf'] > operf:
                        continue

                    if optimal_mean_reversion[symbol]['aperf'] > aperf:
                        continue

                except KeyError:
                    # there wasn't an existing strategy
                    pass

                # either the prior strategies were more weak
                # or there weren't existing strategies, so
                # save this strategy
                optimal_mean_reversion[symbol] = {'type': 'mean_reversion',
                                                  'sma': sma,
                                                  'thr': thr,
                                                  'operf': operf,
                                                  'aperf': aperf}

        return optimal_mean_reversion

    def linear_regression(self):
        Output.log.message('optimizing linear regression strategies')
        optimal_linear_regression = dict()

        data = {'market_close': self.util.read_from_parquet('market_close'),
                'returns_daily': self.util.read_from_parquet('returns_daily')}

        nyse = pd.read_csv(Config.nyse, index_col=5)
        nyse.index = nyse.index.str.replace('.', '_', regex=False)
        nyse.index = nyse.index.str.replace('-', '_', regex=False)
        nyse = nyse.index

        for lag in Periods.lags[::-1]:
            _, operf, aperf = self.backtest.linear_regression(lag, data)
            aperf = np.array([aperf] * len(operf))

            performance = pd.DataFrame(zip(operf, aperf))
            performance.columns = ('operf', 'aperf')
            performance.index = operf.index

            for symbol in nyse:
                if symbol not in performance.index:
                    continue

                try:
                    operf = float(performance['operf'].loc[symbol])
                    aperf = int(performance['aperf'].loc[symbol])

                    if 0 > operf:
                        continue

                    if np.isnan(operf):
                        continue

                    # if an existing strategy is better performer, move on
                    if optimal_linear_regression[symbol]['operf'] > operf:
                        continue

                    if optimal_linear_regression[symbol]['aperf'] > aperf:
                        continue

                except KeyError:
                    # there wasn't an existing strategy
                    pass

                # either the prior strategies were more weak
                # or there weren't existing strategies, so
                # save this strategy
                optimal_linear_regression[symbol] = {
                    'type': 'linear_regression',
                    'lag': lag,
                    'operf': operf,
                    'aperf': aperf
                }

        return optimal_linear_regression


class TraderComponent:
    util = UtilComponent()
    download = DownloadComponent()

    def sma_position(self, symbol, short, long, close_price=None):
        if close_price is None:
            close_price = self.util.read_from_parquet('market_close')
            close_price = close_price[symbol]

        current_price = self.download.current_price(symbol)
        if current_price == np.nan:
            return 0

        today_index = close_price.index.max() + pd.Timedelta(1, 'day')
        close_price.loc[today_index] = current_price

        long_sma = close_price.iloc[-long:].mean()
        short_sma = close_price.iloc[-short:].mean()

        position = 1 if short_sma > long_sma else -1

        return position

    def momentum_position(self, symbol, momentum, close_price=None):
        if close_price is None:
            close_price = self.util.read_from_parquet('market_close')
            close_price = close_price[symbol]

        current_price = self.download.current_price(symbol)
        if current_price == np.nan:
            return 0

        today_index = close_price.index.max() + pd.Timedelta(1, 'day')
        close_price.loc[today_index] = current_price

        returns = close_price / close_price.shift(1)
        returns = pd.Series(np.log(close_price.dropna()))
        momentum = np.sign(returns.iloc[-momentum:].mean())

        position = 1 if momentum == 1 else -1

        return position

    def mean_reversion_position(self, symbol, sma, thr, close_price=None):
        if close_price is None:
            close_price = self.util.read_from_parquet('market_close')
            close_price = close_price[symbol]

        current_price = self.download.current_price(symbol)
        if current_price == np.nan:
            return 0

        today_index = close_price.index.max() + pd.Timedelta(1, 'day')
        close_price.loc[today_index] = current_price

        sma = close_price.iloc[-sma:].mean()
        distance = current_price - sma
        prior_distance = close_price.iloc[-2] - sma

        position = -1 if distance > thr else 0
        position = 1 if -thr > distance else 0
        position = 0 if 0 > distance * prior_distance else position

        return position

    def linear_regression_position(self, symbol, lags, close_price=None):
        if close_price is None:
            close_price = self.util.read_from_parquet('market_close')
            close_price = close_price[symbol]

        current_price = self.download.current_price(symbol)
        if current_price == np.nan:
            return 0

        lagged_returns = self.util.read_from_parquet(f'lag/{symbol}_{lags}')
        lag_columns = [f'{symbol}_{n}' for n in range(lags)]
        lagged_returns = lagged_returns[lag_columns].dropna()

        close_price = close_price.loc[lagged_returns.index]

        returns = close_price / close_price.shift(1)
        returns = returns.dropna()

        lagged_returns = lagged_returns.loc[returns.index]
        outcome = np.sign(returns)
        regression = np.linalg.lstsq(lagged_returns, outcome, rcond=None)

        return_lags = [returns.shift(n).iloc[-1] for n in range(lags - 1)]
        return_lags = [current_price - close_price.iloc[-1]] + return_lags

        projection = np.sign(np.dot(return_lags, regression[0]))

        if np.isnan(projection):
            projection = 0

        return int(projection)
