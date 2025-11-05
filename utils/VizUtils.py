# utils/VizUtils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

class VizUtils:
    DEFAULT_SCALE = px.colors.sequential.Blues

    @staticmethod
    def _theme_colors(dark: bool = False) -> dict:
        """Dark/Light temaya göre bar ve çizgi renkleri."""
        return {
            "template": "plotly_dark" if dark else "plotly_white",
            "paper": "rgba(0,0,0,0)",
            "plot": "rgba(30,30,30,1)" if dark else "rgba(255,255,255,1)",
            "bar": "rgba(59,130,246,0.85)" if dark else "rgba(37,99,235,0.85)",  # mavi
            "bar_line": "rgba(17,24,39,0.60)" if dark else "rgba(255,255,255,0.45)",  # bar kenarı
            "curve": "rgba(239,246,255,0.9)" if dark else "rgba(17,24,39,0.70)",  # KDE çizgisi
        }

    @staticmethod
    def pretty_histogram(series: pd.Series, bins: int = 40, title: str = "Histogram",
                         height: int = 260, dark: bool = False) -> go.Figure:
        """Tema-duyarlı: tek renk bar + belirgin kenar + yumuşak eğri."""
        C = VizUtils._theme_colors(dark)
        x = pd.to_numeric(series, errors="coerce").dropna().values
        fig = go.Figure()
        if x.size == 0:
            fig.update_layout(template=C["template"], height=height,
                              margin=dict(l=10, r=10, t=40, b=10),
                              paper_bgcolor=C["paper"], plot_bgcolor=C["plot"])
            return fig

        counts, edges = np.histogram(x, bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)

        # Tek, doygun bar rengi (gradyan yok -> kontrast net)
        fig.add_bar(
            x=edges[:-1],
            y=counts,
            width=widths,
            marker=dict(color=C["bar"], line=dict(color=C["bar_line"], width=1)),
            hovertemplate="Range: %{x} – %{customdata}<br>Count: %{y:,}<extra></extra>",
            customdata=np.round(edges[1:], 3),
            name="count",
        )

        # KDE-benzeri pürüzsüz eğri
        if counts.sum() > 0:
            k = max(3, int(0.04 * counts.size) * 2 + 1)
            t = np.linspace(-2.5, 2.5, k)
            kernel = np.exp(-0.5 * t ** 2);
            kernel /= kernel.sum()
            smooth = np.convolve(counts, kernel, mode="same")
            fig.add_scatter(x=mids, y=smooth, mode="lines",
                            line=dict(width=2, color=C["curve"]),
                            name="smooth",
                            hovertemplate="x: %{x}<br>Smoothed: %{y:.0f}<extra></extra>")

        fig.update_layout(
            template=C["template"], height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            bargap=0.08, showlegend=False,
            title=dict(text=title, x=0.5, xanchor="center", y=0.92),
            xaxis=dict(title=None, zeroline=False, showgrid=True),
            yaxis=dict(title=None, showgrid=True),
            paper_bgcolor=C["paper"], plot_bgcolor=C["plot"],
        )
        return fig

    @staticmethod
    def top_categories_bar(df: pd.DataFrame, col: str, top: int = 6, height: int = 260,
                           title: str = "Top categories", dark: bool = False) -> go.Figure:
        C = VizUtils._theme_colors(dark)

        s = df[col].astype("string").fillna("NA")
        total = int(s.shape[0])
        vc = s.value_counts(dropna=False)
        top_df = vc.head(top).rename_axis("value").reset_index(name="count")
        other = int(vc.iloc[top:].sum()) if vc.shape[0] > top else 0
        if other > 0:
            top_df.loc[len(top_df)] = ["Other values …", other]
        top_df["freq_pct"] = (top_df["count"] / max(1, total)) * 100.0
        top_df = top_df.iloc[::-1].reset_index(drop=True)

        fig = px.bar(top_df, x="count", y="value", orientation="h", text="count",
                     labels={"value": "", "count": "", "freq_pct": "Frequency (%)"},
                     title=title, color_discrete_sequence=[C["bar"]])
        fig.update_traces(
            marker=dict(line=dict(color=C["bar_line"], width=1)),
            texttemplate="%{text:,}", insidetextanchor="start",
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        )
        fig.update_layout(
            template=C["template"], height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=True, zeroline=False, title=None),
            yaxis=dict(showgrid=False, title=None),
            showlegend=False,
            paper_bgcolor=C["paper"], plot_bgcolor=C["plot"],
        )
        return fig

    @staticmethod
    def time_count_bar(series: pd.Series, freq: str = "D", height: int = 320,
                       title: str = "Time distribution", dark: bool = False) -> go.Figure:
        C = VizUtils._theme_colors(dark)
        sd = pd.to_datetime(series, errors="coerce")
        grp = sd.dropna().dt.to_period(freq).value_counts().sort_index()
        fig = go.Figure()
        if not grp.empty:
            fig = px.bar(x=grp.index.to_timestamp(), y=grp.values,
                         labels={"x": "", "y": "Count"}, title=title,
                         color_discrete_sequence=[C["bar"]])
            fig.update_traces(marker=dict(line=dict(color=C["bar_line"], width=1)),
                              hovertemplate="%{x|%Y-%m-%d}<br>Count: %{y:,}<extra></extra>")
        fig.update_layout(template=C["template"], height=height, margin=dict(l=10, r=10, t=30, b=10),
                          showlegend=False, paper_bgcolor=C["paper"], plot_bgcolor=C["plot"])
        return fig

