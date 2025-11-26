import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import typing

def plot_twodim(x = None, y = None, xlabel: str = 'x-label', ylabel: str = 'y-label'):
    """Plot x and y in a diagram with given labels, xlabel and ylabel.

    Args:
        x (_type_, optional): the X. Defaults to None.
        y (_type_, optional): the Y. Defaults to None.
        xlabel (str, optional): Label for X. Defaults to 'x-label'.
        ylabel (str, optional): Label for Y. Defaults to 'y-label'.
    """
    if x is None or y is None:
        print(f'No value given for either x {x} or y {y}.')
        return
    try:
        plt.plot(x, y)
        plt.grid()
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.show()
    except Exception as e:
        print(f'Error plotting x {xlabel} and y {ylabel}: {e}')


def save_fig(df: pd.DataFrame, figurename: str, filetype: str):
    fig = df.plot()
    fig.grid()
    plt.savefig(f'{figurename}.{filetype}')


def plot_kde(df: pd.DataFrame, hue: str, diag_kind: str = 'kde', height: float = 1.0):

    # ensure numeric columns are used for the grid
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if hue not in df.columns:
        raise ValueError(f"Hue column '{hue}' does not exist in dataframe.")

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to plot with pairplot.")

    # build final df for plotting
    plot_df = df[numeric_cols + [hue]]

    sns.set(style='ticks')
    sns.pairplot(plot_df, hue=hue, diag_kind=diag_kind, height=height)



def scatter_matrix(df: pd.DataFrame, alpha: float = 0.2, figsize: typing.Tuple[int, int] = (6, 6), diagonal: str = 'kde'):
    pd.plotting.scatter_matrix(df, alpha=alpha, figsize=figsize, diagonal=diagonal)
