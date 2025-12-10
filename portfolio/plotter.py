from __future__ import annotations
import pandas as pd
from matplotlib import axes
from matplotlib import pyplot as plt
from typing import Optional, Dict, Tuple, Any, List, Hashable

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

    def _get_group_axes(self, group_key: Hashable | None) -> Tuple[plt.Figure, axes.Axes]:
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
    def __init__(
        self,
        *,
        style: Optional[str] = 'ggplot',
        auto_legend: bool = True
    ):
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
        portfolio: Any,
        ax: axes.Axes | None,
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
        portfolio: Any,
        ax: axes.Axes | None = None,
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
        portfolio: Any,
        ax: axes.Axes | None = None,
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

GLOBAL_PLOTTER = PortfolioPlotter(style='ggplot', auto_legend=True)
