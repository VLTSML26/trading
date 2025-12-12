from __future__ import annotations
import pandas as pd
import numpy as np
from matplotlib import axes, cm
from matplotlib import pyplot as plt
from typing import Optional, Dict, Tuple, Any, List, Hashable, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Portfolio

class BasePlotter:
    """
    Base astratta per plotter. Offre routing di figure/axes tramite 'group key'.
    Le subclass definiscono come generare la group key e cosa plottare.
    """
    def __init__(self, auto_legend: bool = True):
        """
        Costruttore di BasePlotter.
        """
        self.auto_legend = auto_legend
    
        # dizionario che associa una chiave (hashable) a una coppia (fig, ax)
        # serve per riutilizzare lo stesso axes se la chiave coincide.
        self.groups: Dict[Hashable, Tuple[plt.Figure, axes.Axes]] = {}

        # memorizza l'ultima figura utilizzata (vedi anche relativo metodo get)
        self.last_fig: Optional[plt.Figure] = None

    def reset(self) -> None:
        """
        Pulisce lo stato interno (non chiude le figure correnti).
        """
        self.groups.clear()
        self.last_fig = None

    def get_last_figure(self) -> plt.Figure | None:
        return self.last_fig

    def get_all_figures(self) -> List[plt.Figure]:
        return [
            fig for (fig, _ax) in self.groups.values()
            if fig is not None and plt.fignum_exists(fig.number)
        ]

    def _get_group_axes(self, group_key: Optional[Hashable]) -> Tuple[plt.Figure, axes.Axes]:
        """
        Se 'group_key' è None: restituisce SEMPRE una nuova figura/axes (nessuna condivisione).
        Altrimenti: riutilizza o crea (fig, ax) associati a quella chiave.
        """
        if group_key is None:
            fig, ax = plt.subplots()
            self.last_fig = fig
            return fig, ax

        pair = self.groups.get(group_key)
        if pair:
            fig, ax = pair
            if fig is not None and plt.fignum_exists(fig.number):
                self.last_fig = fig
                return fig, ax

        fig, ax = plt.subplots()
        self.groups[group_key] = (fig, ax)
        self.last_fig = fig
        return fig, ax

class PortfolioPlotter(BasePlotter):
    """
    Plotter della classe Portfolio.
    Decide la 'group key' in funzione dell'indice del portfolio e del tipo di grafico da plottare.
    Condivide l'axes solo se l'indice è DatetimeIndex (e coincide) e se il tipo di grafico è lo stesso.
    """
    def __init__(self, *, style: Optional[str]='ggplot', auto_legend: bool=True):
        """
        Costruttore di PortfolioPlotter che eredita da BasePlotter.
        """
        super().__init__(auto_legend=auto_legend)
        if style:
            plt.style.use(style)

    @staticmethod
    def _key_for_index(idx: pd.Index) -> Hashable | None:
        """
        Restituisce una chiave hashable per la condivisione.

        :param idx: Indice
        :type idx: pd.Index
        :return: Chiave hashable per la condivisione basata su tuple degli int64 (ns)
            oppure None se l'indice non è un DatetimeIndex.
        :rtype: Hashable | None
        """
        if isinstance(idx, pd.DatetimeIndex):
            # uguaglianza stretta delle date/ordine
            return ("datetime64[ns]", tuple(idx.asi8.tolist()))
        # nessuna condivisione automatica
        return None

    def _axes_for_portfolio(
        self,
        portfolio: "Portfolio",
        ax: Optional[axes.Axes],
        *,
        plot_kind: str
    ) -> axes.Axes:
        """
        Restituisce l'axes da usare: se ax è None, decide la group key includendo:
        - index_key (condivisione ax solo se indice è DateTime e coincide)
        - plot_kind (non mescola diversi tipi di grafico tipo returns e drawdown)
        ... altrimenti usa ax esplicito.
        """
        if ax is not None:
            return ax

        # chiave di indice
        idx = portfolio.comp_returns.index # FIXME: va bene ma forse sarebbe meglio index più generico tra tutti i dataframe di Portfolio
        index_key = self._key_for_index(idx)

        # se index_key è None, BasePlotter genererà sempre una figura nuova (group_key sarà None)
        group_key: Hashable | None = (plot_kind, index_key) if index_key is not None else None
        _, ax2 = self._get_group_axes(group_key)
        return ax2

    def plot_returns(
        self,
        portfolio: "Portfolio",
        ax: Optional[axes.Axes] = None,
        *,
        rescale: bool = False,
        legend_loc: str = 'best',
        **kwargs
    ) -> axes.Axes:
        """
        Funzione di realizzazione grafica dei rendimenti del portafoglio.
        """
        # creazione ax e titolo della figura
        ax = self._axes_for_portfolio(portfolio, ax, plot_kind='returns')
        ax.set_title("Portfolio returns")

        # dati da plottare (e rinormalizzare se caso)
        series = portfolio.comp_returns
        data = series / series.iloc[0] if rescale else series

        # plot
        data.plot(ax=ax, label=getattr(portfolio, 'name', None), **kwargs)
        if self.auto_legend:
            ax.legend(loc=legend_loc)
        return ax

    def plot_drawdown(
        self,
        portfolio: "Portfolio",
        ax: Optional[axes.Axes] = None,
        *,
        legend_loc: str = 'best',
        **kwargs
    ) -> axes.Axes:
        """
        Funzione di realizzazione grafica dei drawdown del portafoglio.
        """
        # creazione ax e titolo della figura
        ax = self._axes_for_portfolio(portfolio, ax, plot_kind='drawdown')
        ax.set_title("Portfolio drawdowns")

        # plot
        series = -portfolio.drawdown
        series.plot(ax=ax, label=getattr(portfolio, 'name', None), **kwargs)
        if self.auto_legend:
            ax.legend(loc=legend_loc)
        return ax

    def plot_weights(
        self,
        portfolios: list["Portfolio"],
        ax: Optional[axes.Axes] = None,
        *,
        legend_loc: str = 'best',
        colormap: str = 'tab20', # colormap per i tickers
        sort_tickers: bool = False # se True, ordina i tickers per nome
    ) -> axes.Axes:
        """
        Grafico a colonne impilate dei pesi per una lista di portafogli.
        """
        # check input
        if not portfolios:
            raise ValueError("Must provide at least one Portfolio.")
        if ax is None:
            # group_key separata: ('weights', None) non dipende dall'indice temporale
            fig, ax = self._get_group_axes(('weights', None))

        # matrice dei pesi
        all_tickers = []
        for p in portfolios:
            all_tickers.extend(list(p.weights.index))
        tickers_unique = sorted(set(all_tickers)) if sort_tickers else list(dict.fromkeys(all_tickers))

        # costruzione DataFrame con 0 dove l'asset non è presente in un portfolio
        data = pd.DataFrame(
            0.,
            index=[getattr(p, 'name', f'Portfolio {i}') for i, p in enumerate(portfolios)],
            columns=tickers_unique
        )
        for i, p in enumerate(portfolios):
            name = getattr(p, 'name', f'Portfolio {i}')
            # allineiamo sulla union dei tickers, riempiendo con 0
            w = p.weights.reindex(tickers_unique).fillna(0.)
            # normalizziamo per sicurezza (non dovrebbe servire)
            s = w.sum()
            if s > 0 and not np.isclose(s, 1.):
                w = w / s
            data.loc[name, :] = w

        # colori consistenti per tickers
        cmap = cm.get_cmap(colormap, len(tickers_unique))
        colors = {t: cmap(i) for i, t in enumerate(tickers_unique)}

        # disegno delle barre impilate
        x = np.arange(len(data.index))
        width = 0.6

        bottom = np.zeros(len(data.index))
        for t in tickers_unique:
            heights = data[t].values
            ax.bar(x, heights, width=width, bottom=bottom, color=colors[t], label=t)
            bottom += heights

        # ulteriori abbellimenti
        ax.set_xticks(x)
        ax.set_xticklabels(list(data.index), rotation=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Weights")
        ax.set_title("Portfolio weights")

        if self.auto_legend:
            ax.legend(loc=legend_loc, ncol=2, frameon=True)

        # griglia leggera sull'asse y
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        return ax

GLOBAL_PLOTTER = PortfolioPlotter(style='ggplot', auto_legend=True)
