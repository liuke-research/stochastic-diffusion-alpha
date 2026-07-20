import pandas as pd
import numpy as np


class MarketLoader:

    def __init__(self, path="data/HS300.csv"):
        self.path = path

    def load(self):

        df = pd.read_csv(self.path)

        df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values("date")

        df["return"] = np.log(
            df["close"] / df["close"].shift(1)
        )

        df = df.dropna()

        return df