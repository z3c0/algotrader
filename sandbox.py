from components import Data
from const import Config
import pandas as pd


quotemedia_df = pd.read_csv(Config.quotemedia)

quotemedia_columns = ['code', 'refreshed_at', 'from_date', 'to_date']
quotemedia_records = quotemedia_df[quotemedia_columns].iterrows()

close_matrix = pd.DataFrame()
for idx, (symbol, refreshed, _, _) in quotemedia_records:
    if not refreshed:
        continue

    stock_df = Data.download.stock_data(symbol)

    close_price = stock_df['Close'] * stock_df['Split']
    close_matrix[symbol] = close_price

print(close_matrix)
