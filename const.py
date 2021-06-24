

class Account:
    path = 'data/json/account.json'


class System:
    log = 'algotrader.log'


class Config:
    path = 'algotrader.cfg'
    nyse = 'data/csv/NYSE.csv'
    quotemedia = 'data/csv/quotemedia.csv'
    quotemedia_desc = 'data/html/quotemedia_desc.html'
    watchlist = 'data/csv/watchlist.csv'


class Periods:
    sma = [14, 21, 42, 64, 126, 252]
    momentum = [3, 14, 21, 64]
    lags = [3, 5, 8, 13, 21]

    @staticmethod
    def add_sma(*new_smas):
        new_list = Periods.sma + new_smas
        Periods.sma = list(set(new_list))

    @staticmethod
    def add_momentum(*new_momentums):
        new_list = Periods.momentum + new_momentums
        Periods.momentum = list(set(new_list))

    @staticmethod
    def add_lags(*new_lags):
        new_list = Periods.lags + new_lags
        Periods.momentum = list(set(new_list))


class Argument:
    flag = None
    alias = None
    help_text = None


class Group:
    title = None


class Arguments:
    class Util(Group):
        title = 'utility'

        class Initialize(Argument):
            flag = 'init'
            alias = 'i'
            help_text = ('initialize folder structure and download history for'
                         ' all companies')

    class Data(Group):
        title = 'data processing'

        class Download(Argument):
            flag = 'download'
            alias = 'd'
            help_text = 'download NYSE close price history'

        class Transform(Argument):
            flag = 'transform'
            alias = 't'
            help_text = ('reshape downloaded NYSE data into matrices'
                         ' for easier processing')

    class Backtest(Group):
        title = 'backtesting'

        class Optimize(Argument):
            flag = 'optimize'
            alias = 'o'
            help_text = 'find the most optimal strategy per stock'

    class Position(Group):
        title = 'position'

        class Strategize(Argument):
            flag = 'strategize'
            alias = 's'
            help_text = ('determine position by running optimal strategies'
                         ' against current close prices')

        class Apply(Argument):
            flag = 'apply'
            alias = 'a'
            help_text = 'apply selected positions (places transactions)'

    class Testing(Group):
        title = 'testing'

        class Verbose(Argument):
            flag = 'verbose'
            alias = 'v'

        class Rollback(Argument):
            flag = 'rollback'
            alias = 'r'
            help_text = 'prevent changes to account from being saved'
