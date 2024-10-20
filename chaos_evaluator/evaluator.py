import numpy as np
import pandas as pd
from scipy.stats import norm

class ChaosEvaluator:
    def __init__(self, visualize=True):
        self.visualize = visualize
        if visualize:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output
            plt.style.use("dark_background")
            self.plt = plt
            self.clear_output = clear_output

    def evaluate(self, df: pd.DataFrame, normalize: bool = False) -> dict:
        """
        Evaluate the most stable span for 1 or multiple time series

        Args:
            df (pd.DataFrame): a data frame of 1 or multiple time series
            normalize (bool): whether or not to normalize the data first

        Returns:
            dict: a dictionary of the scores for stability (0-1) and best spans; 1 means perfectly stable.
        """
        self._input_validation(df)

        if normalize:
            arr = df.pct_change()[1:].copy()
        else:
            arr = df.copy()

        best_ps = []
        accumulative_ps = 0
        n_init_points = len(arr) // 2
        for i in range(n_init_points):
            all_spans = self._get_all_spans(arr, start=i)
            ps = [self._evaluate(array) for array in all_spans]
            accumulative_ps += np.array(ps)
            best_ps.append(np.argmax(ps, axis=0))

            if self.visualize:
                self._train_visualize(accumulative_ps, i, labels=df.columns)

        # evaluate the best span(s)
        acc_ps = accumulative_ps / len(accumulative_ps)
        out_df = pd.DataFrame(acc_ps, columns=df.columns)
        out_df.index = out_df.index + 2
        best_spans = out_df.idxmax().to_dict()

        # evaluate stability
        best_ps = np.array(best_ps)
        all_counts = [np.unique(best_ps[:, i], return_counts=True)[1] for i in range(best_ps.shape[1])]
        scores = {name: np.mean(counts) / np.sum(counts) for name, counts in zip(df.columns, all_counts)}

        # output record
        out_record = {}
        for i in df.columns:
            out_record[i] = {"best_span": best_spans[i],
                             "chaos_score": np.round(scores[i], 4)}

        return out_record

    def _evaluate(self, df: np.array) -> float:
        reals = np.fft.fft(df, axis=0).real
        iqd = np.subtract(*np.percentile(reals, [75, 25], axis=0))
        iqd = np.where(iqd == 0, 1e-20, iqd)  # avoid 0 division
        z = -np.absolute((reals - np.median(reals, axis=0)) / iqd)
        p = np.mean(norm(0, 1).cdf(z) * 2, axis=0)
        return p

    def _get_all_spans(self, df: pd.DataFrame, start: int) -> list:
        length = len(df) // 2
        max_span = len(df) // 2
        all_spans = [df.iloc[start: start + i].to_numpy() for i in range(length) if (start + i > start + 1)]
        return all_spans

    def _train_visualize(self, acc_ps, j, labels):
        self.clear_output(wait=True)
        n_plots = min(acc_ps.shape[1], 4)
        if n_plots >= 2:
            fig, ax = self.plt.subplots(nrows=n_plots, ncols=1, sharex=True, figsize=(5, 7))
            for index in range(n_plots):
                counts, bins = np.histogram(acc_ps[:, index], 20)
                counts = np.array(counts) / np.sum(counts)
                ax[index].hist(bins[:-1], bins, weights=counts)
                ax[index].set_xticks([])
                ax[index].set_title(labels[index], pad=15, fontsize=20)
        else:
            counts, bins = np.histogram(acc_ps, 20)
            counts = np.array(counts) / np.sum(counts)
            fig = self.plt.figure(figsize=(6, 6))
            self.plt.hist(bins[:-1], bins, weights=counts)
            self.plt.title(labels[0], pad=15, fontsize=20)
            self.plt.xticks([])

        self.plt.xlabel(f"\nIteration {j}")
        self.plt.tight_layout()
        self.plt.show()

    def _input_validation(self, df):
        assert isinstance(df, pd.DataFrame), "Input needs to be a Pandas DataFrame."
        assert not df.isnull().any().any(), "There are NaN values in your input DataFrame."
