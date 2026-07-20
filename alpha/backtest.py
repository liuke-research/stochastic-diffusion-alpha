class AlphaSignal:
    def __init__(self, feature_builder):
        self.fb = feature_builder

    def generate(self, x0, xT):
        return self.fb.build_signal(x0, xT)


class Backtester:
    def run(self, signal, returns):
        pnl = signal * returns
        equity = pnl.cumsum()
        return equity