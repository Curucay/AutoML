import polars as pl
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import networkx as nx

class VizUtils:
    @staticmethod
    def numeric_histogram(df: pl.DataFrame, column: str, dark=False):
        """
        Sayısal sütunlar için küçük histogram grafiği.
        """
        if df[column].dtype not in (pl.Int32, pl.Int64, pl.Float32, pl.Float64):
            return None

        pdf = df.select(column).to_pandas()
        bg = "#111827" if dark else "#FFFFFF"
        txt = "#F5F6F8" if dark else "#111827"

        chart = (
            alt.Chart(pdf, title=f"Dağılım: {column}")
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=20)),
                y=alt.Y("count()", title="Frekans"),
                tooltip=[column, "count()"]
            )
            .configure(background=bg)
            .configure_axis(labelColor=txt, titleColor=txt)
            .configure_title(color=txt, fontSize=13)
            .properties(height=200)
        )
        return chart

    @staticmethod
    def _theme_cfg(dark: bool = False):
        bg = "#1E1E1E" if dark else "#FFFFFF"
        grid = "#2A2A2A" if dark else "#EAEAEA"
        txt = "#F5F6F8" if dark else "#111827"
        return dict(
            view={"stroke": "transparent", "fill": bg},
            background=bg,
            axis=dict(labelColor=txt, titleColor=txt, gridColor=grid, domainColor=grid),
            legend=dict(labelColor=txt, titleColor=txt),
        )

    @staticmethod
    def histogram(series: pl.Series, bins: int = 40, title: str = "Histogram",
                  height: int = 260, dark: bool = False) -> alt.Chart:
        """
        Sayısal kolonun histogram ve densite dağılımını çizer.
        """
        s = series.cast(pl.Float64, strict=False).drop_nulls()
        if s.is_empty():
            return alt.Chart(pl.DataFrame({"x": [], "count": []})).mark_bar()

        _sample_max = 200_000  # Altair için 200k satır fazlasıyla yeterli
        if s.len() > _sample_max:
            s = s.sample(n=_sample_max, shuffle=True)

        df = pl.DataFrame({"x": s})
        cfg = VizUtils._theme_cfg(dark)

        # Histogram verisini Pandas'a çeviriyoruz çünkü Altair Polars ile direkt çalışmaz
        pdf = df.to_pandas()

        hist = (alt.Chart(pdf)
                .mark_bar(size=0, opacity=0.9, color="#2563EB")
                .encode(
                    x=alt.X("x:Q", bin=alt.Bin(maxbins=bins), title=None),
                    y=alt.Y("count()", title=None),
                    tooltip=[alt.Tooltip("count()", title="Count"),
                             alt.Tooltip("x:Q", bin=True, title="Range")]
                ))

        dens = (alt.Chart(pdf)
                .transform_density("x", as_=["x", "density"])
                .mark_line(color="#E5E7EB" if dark else "#111827", strokeWidth=2)
                .encode(x="x:Q", y=alt.Y("density:Q", axis=None)))

        return (hist + dens).properties(title=title, height=height).configure(**cfg)

    @staticmethod
    def top_categories(df: pl.DataFrame, col: str, top: int = 8,
                       title: str = "Top categories", height: int = 260,
                       dark: bool = False) -> alt.Chart:
        """
        Kategorik kolonların en sık görülen değerlerini çizer.
        Polars vektörel ve tip uyumlu sürüm.
        """
        s = df[col].cast(pl.Utf8).fill_null("NA")
        total = s.len()

        # Value counts (Polars native, UInt32 'count' üretir)
        vc = s.value_counts(sort=True)
        main_col = vc.columns[0]  # örn: "province"
        top_df = vc.head(top)

        # Diğer değerlerin toplamı
        top_count_sum = int(top_df["count"].sum()) if top_df.height > 0 else 0
        others_count = int(total - top_count_sum) if vc.height > top else 0

        # Frekans oranlarını hesapla
        top_df = top_df.with_columns(
            (pl.col("count") / max(1, total) * 100).alias("freq_pct")
        )

        # Kolon adlarını normalize et
        top_df = top_df.rename({main_col: "value"})

        # ✅ Tipleri eşitle: UInt64 + Float64
        top_df = top_df.with_columns([
            pl.col("count").cast(pl.UInt64),
            pl.col("freq_pct").cast(pl.Float64)
        ])

        # 'Other values' satırını ekle
        if others_count > 0:
            other_row = pl.DataFrame({
                "value": ["Other values"],
                "count": [others_count],
                "freq_pct": [others_count / max(1, total) * 100]
            })

            # ✅ Aynı tipleri koru
            other_row = other_row.with_columns([
                pl.col("count").cast(pl.UInt64),
                pl.col("freq_pct").cast(pl.Float64)
            ])

            top_df = pl.concat([top_df, other_row])

        # --- Tema ayarları (Altair)
        bg = "#1E1E1E" if dark else "#FFFFFF"
        grid = "#2A2A2A" if dark else "#EAEAEA"
        txt = "#F5F6F8" if dark else "#111827"
        cfg = dict(
            view={"stroke": "transparent", "fill": bg},
            background=bg,
            axis=dict(labelColor=txt, titleColor=txt, gridColor=grid, domainColor=grid),
            legend=dict(labelColor=txt, titleColor=txt),
        )

        # --- Görselleştirme (aynı stil korunur)
        pdf = top_df.to_pandas()
        base = alt.Chart(pdf).encode(
            x=alt.X("count:Q", title=None),
            y=alt.Y("value:N", sort=None, title=None),
            tooltip=[
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip("count:Q", title="Count", format=",.0f"),
                alt.Tooltip("freq_pct:Q", title="Frequency (%)", format=".1f")
            ]
        )

        bars = base.mark_bar(opacity=0.9).encode(
            color=alt.Color("value:N").scale(scheme="tableau10").legend(None)
        )
        txt_layer = base.mark_text(align="left", dx=4,
                                   color="#E5E7EB" if dark else "#111827"
                                   ).encode(text=alt.Text("count:Q", format=",.0f"))

        return (bars + txt_layer).properties(title=title, height=height).configure(**cfg)

    @staticmethod
    def time_count(series: pl.Series, freq: str = "D", title: str = "Time distribution",
                   height: int = 300, dark: bool = False) -> alt.Chart:
        """
        Tarihsel kolonun zaman periyoduna göre dağılımını çizer.
        """
        # (Önceki düzeltmemizdeki tip kontrolü)
        if series.dtype == pl.Datetime:
            s = series.drop_nulls()
        elif series.dtype == pl.Utf8:
            s = series.str.strptime(pl.Datetime, strict=False).drop_nulls()
        else:
            s = pl.Series(values=[], dtype=pl.Datetime)

        if s.is_empty():
            return alt.Chart(pl.DataFrame({"x": [], "count": []})).mark_bar()

        # Tarihe göre grupla
        df = pl.DataFrame({"x": s})

        # 1. Önce veriyi sırala (group_by_dynamic için zorunlu)
        df_sorted = df.sort("x")

        freq_map = {
            "D": "1d",  # Günlük
            "W": "1w",  # Haftalık
            "M": "1mo",  # Aylık (Month-start)
            "Y": "1y"  # Yıllık
        }
        # freq'i Polars formatına çevir, bulamazsa orijinali kullan
        polars_freq = freq_map.get(freq.upper(), freq)
        # ======================================

        # 2. group_by_dynamic ile grupla (çok daha hızlı)
        grp = df_sorted.group_by_dynamic(
            "x",  # Zaman kolonu
            every=polars_freq,  # Gruplama periyodu (DÜZELTİLDİ)
            period=polars_freq,  # Periyot aralığı (DÜZELTİLDİ)
            closed="left"  # Periyodun başlangıcını dahil et
        ).agg(
            pl.count().alias("count")  # 'x' kolonunu say
        )

        grp = grp.rename({"x": "period"})
        # =======================================================

        cfg = VizUtils._theme_cfg(dark)
        # 'grp' zaten sıralı gelir (sort("period") gerekmez)
        pdf = grp.rename({"period": "x"}).to_pandas()

        ch = (alt.Chart(pdf)
              .mark_bar(opacity=0.9, color="#2563EB")
              .encode(
            x=alt.X("x:T", title=None),
            y=alt.Y("count:Q", title=None),
            tooltip=[alt.Tooltip("x:T", title="Date"),
                     alt.Tooltip("count:Q", title="Count", format=",.0f")]
        )
              .properties(title=title, height=height)
              .configure(**cfg))
        return ch

    @staticmethod
    def correlation_heatmap(
            df_corr: pl.DataFrame,
            dark: bool = False,
            title: str = "Korelasyon Matrisi"
    ) -> alt.Chart:
        """
        Altair ile interaktif korelasyon matrisi.
        Geliştirilmiş yazı boyutları, ortalı başlık ve okunabilir legend.
        """

        # 1️⃣ Sayısal değişken yoksa bilgi göster
        if "message" in df_corr.columns:
            base = alt.Chart(pd.DataFrame({"info": ["Sayısal değişken bulunamadı."]})).mark_text(
                text="Sayısal değişken bulunamadı.",
                size=16,
                color="red"
            ).properties(title=title, height=100)
            return base

        # 2️⃣ Polars → Pandas dönüşümü ve uzun forma (melt)
        pdf = df_corr.to_pandas().set_index("column")
        corr_long = (
            pdf.reset_index()
            .melt(id_vars="column", var_name="variable", value_name="correlation")
            .rename(columns={"column": "var1", "variable": "var2"})
        )

        # 3️⃣ Tema renkleri
        bg = "#1E1E1E" if dark else "#FFFFFF"
        txt = "#F5F6F8" if dark else "#111827"

        # 4️⃣ Eksen sıralaması (orijinal sıralama)
        axis_order = list(pdf.columns)

        # 5️⃣ Dinamik yükseklik ve yazı stili
        cell_size = 70
        min_height = 800
        max_height = 1200
        chart_height = max(min_height, min(max_height, len(axis_order) * cell_size))
        text_color_on_strong = "#FFFFFF"
        text_color_on_weak = txt

        # === ALT TEMEL GRAFİK ===
        base = alt.Chart(corr_long).encode(
            x=alt.X(
                "var1:N",
                title=None,
                sort=axis_order,
                axis=alt.Axis(
                    labelAngle=0  # Etiketleri yatay (0 derece) yapar
                )
            ),
            y=alt.Y(
                "var2:N",
                title=None,
                sort=axis_order
            ),
            tooltip=[
                alt.Tooltip("var1:N", title="Değişken 1"),
                alt.Tooltip("var2:N", title="Değişken 2"),
                alt.Tooltip("correlation:Q", title="Korelasyon", format=".3f"),
            ]
        )

        # === HEATMAP ===
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(
                    scheme="redblue",
                    domain=[-1, 1],
                    range="diverging"
                ),
                legend=alt.Legend(
                    title="Korelasyon",
                    titleFontSize=14,
                    titleFontWeight="bold",
                    labelFontSize=14,
                    labelLimit=60,
                    padding=10,
                    gradientLength=chart_height - 200
                )
            )
        )

        # === METİN ETİKETLERİ ===
        text_labels = base.mark_text(baseline="middle", fontSize=15, fontWeight="bold").encode(
            text=alt.Text("correlation:Q", format=".3f"),
            color=alt.condition(
                alt.expr.abs(alt.datum.correlation) > 0.5,
                alt.value(text_color_on_strong),
                alt.value(text_color_on_weak)
            )
        )

        # === FİNAL GRAFİK ===
        final_chart = (heatmap + text_labels).properties(
            title=alt.TitleParams(
                text=title,
                fontSize=26,  # 🔹 Başlık büyütüldü
                fontWeight="bold",  # 🔹 Kalın yapıldı
                anchor="middle",  # 🔹 Ortalandı
                dy=-5  # 🔹 Yukarı biraz taşındı
            ),
            height=chart_height,
            width=chart_height,
            background=bg
        ).configure_axis(
            labelFontSize=15,  # 🔹 Eksen yazıları büyütüldü
            titleFontSize=16,
            labelColor=txt,
            titleColor=txt
        ).configure_title(
            color=txt,
            font="Inter",
            fontWeight="bold"
        ).configure_legend(
            titleColor=txt,
            labelColor=txt,
            labelFontSize=14,
            titleFontSize=16
        ).interactive()

        return final_chart

    @staticmethod
    def correlation_strength_bar(
            df_corr: pl.DataFrame,
            target_col: str,
            dark: bool = False,
            title: str = "Korelasyon Gücü Grafiği"
    ) -> alt.Chart:
        """
        Hedef değişkenle diğer değişkenlerin korelasyon gücünü sıralı çubuk grafikle gösterir.
        """

        if "message" in df_corr.columns or target_col not in df_corr.columns:
            base = alt.Chart(pd.DataFrame({"info": ["Hedef değişken bulunamadı."]})).mark_text(
                text="Hedef değişken bulunamadı.",
                size=14,
                color="red"
            ).properties(title=title, height=100)
            return base

        # Polars → Pandas
        pdf = df_corr.to_pandas().set_index("column")
        correlations = pdf[target_col].drop(target_col, errors="ignore").sort_values(key=abs, ascending=False)
        df_bar = correlations.reset_index()
        df_bar.columns = ["Değişken", "Korelasyon"]

        # Tema renkleri
        bg = "#1E1E1E" if dark else "#FFFFFF"
        txt = "#F5F6F8" if dark else "#111827"

        # Grafik
        chart = (
            alt.Chart(df_bar, title=alt.TitleParams(text=title, fontSize=26, fontWeight="bold", anchor="middle"))
            .mark_bar(size=28)
            .encode(
                x=alt.X("Korelasyon:Q",
                        scale=alt.Scale(domain=[-1, 1]),
                        axis=alt.Axis(
                            title="Korelasyon Gücü",
                            titleFontWeight="bold"
                        )
                ),
                y=alt.Y("Değişken:N",
                        sort="-x",
                        axis=alt.Axis(
                            title="Değişkenler",
                            titleFontWeight="bold"
                        )
                ),
                color=alt.condition(
                    "datum.Korelasyon > 0",
                    alt.value("#E4572E"),  # Pozitif -> Turuncu
                    alt.value("#4B9CD3"),  # Negatif -> Mavi
                ),
                tooltip=[
                    alt.Tooltip("Değişken:N", title="Değişken"),
                    alt.Tooltip("Korelasyon:Q", title="Değer", format=".3f"),
                ],
            )
            .properties(
                width=600,
                height=600,
                background=bg
            )
            .configure_axis(
                labelColor=txt,
                titleColor=txt,
                labelFontSize=18,
                titleFontSize=20,
            )
            .configure_title(color=txt)
        )

        return chart

    # Bar Plot
    @staticmethod
    def missing_bar(df_missing: pl.DataFrame, dark=False):
        pdf = df_missing.to_pandas()
        bg = "#1E1E1E" if dark else "#FFFFFF"
        txt = "#F5F6F8" if dark else "#111827"

        chart = (
            alt.Chart(pdf, title=alt.TitleParams(text="Eksik Değer Dağılımı", fontSize=26, fontWeight="bold", anchor="middle"))
            .mark_bar()
            .encode(
                x=alt.X("missing_pct:Q",
                        axis=alt.Axis(
                            title="Eksik Değer Oranı (%)",
                            titleFontWeight="bold"
                        )
                ),
                y=alt.Y("column:N",
                        sort="-x",
                        axis=alt.Axis(
                            title="Değişkenler",
                            titleFontWeight="bold"
                        )
                ),
                color=alt.Color("missing_pct:Q", scale=alt.Scale(scheme="reds")),
                tooltip=[
                    alt.Tooltip("column:N", title="Kolon"),
                    alt.Tooltip("missing_count:Q", title="Eksik Sayısı", format=",d"),
                    alt.Tooltip("missing_pct:Q", title="Oran (%)", format=".2f")
                ],
            )
            .properties(
                width=600,
                height=600,
                background=bg,
            )
            .configure_axis(
                labelColor=txt,
                titleColor=txt,
                labelFontSize=18,
                titleFontSize=20,
            )
            .configure_legend(  # <-- BU SATIRI EKLEYİN
                titleColor=txt,
                labelColor=txt,
                titleFontSize=16,
                labelFontSize=14,
                gradientLength=400
            )
        )
        return chart

    # Matrix Plot (Missingno)
    @staticmethod
    def missing_matrix(df: pl.DataFrame):
        pdf = df.to_pandas()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        msno.matrix(
            pdf.sample(min(5000,
            len(pdf))),
            ax=ax,
            sparkline=False,
            fontsize=6
        )
        ax.set_title(
            "Eksik Değer Matrisi",
            fontsize=8,
            fontweight="bold"
        )
        return fig

    # Heatmap (Missingno)
    @staticmethod
    def missing_heatmap(df: pl.DataFrame):
        pdf = df.to_pandas()
        fig, ax = plt.subplots(figsize=(10, 8))
        msno.heatmap(pdf.sample(min(10000, len(pdf))), ax=ax)
        ax.set_title("Eksik Değer Korelasyon Haritası", fontsize=13)
        return fig

    # Dendrogram (Missingno)
    @staticmethod
    def missing_dendrogram(df: pl.DataFrame):
        pdf = df.to_pandas()
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        msno.dendrogram(
            pdf.sample(min(2000, len(pdf))),
            ax=ax,
            orientation='top'
        )
        ax.set_title("Eksik Değer Dendrogramı", fontsize=7, pad=7)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_fontsize(8)
        # --- X VE Y EKSEN ETİKETLERİ ---
        ax.set_xlabel("Değişkenler", fontsize=7, labelpad=6)
        ax.set_ylabel("Korelasyon Mesafesi", fontsize=7, labelpad=6)
        # --- TICK FONT BOYUTLARI VE ROTASYONLAR ---
        ax.tick_params(axis='x', labelsize=7, rotation=45)
        ax.tick_params(axis='y', labelsize=7)
        # --- LAYOUT OPTİMİZASYONU ---
        plt.tight_layout(pad=1.0)
        return fig

    # Eksik Korelasyon Plotu (Correlation Plot)
    @staticmethod
    def missing_corr_plot(df: pl.DataFrame):
        pdf = df.select([pl.col(c).is_null().cast(pl.Int8).alias(c) for c in df.columns]).to_pandas()
        corr = pdf.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="Reds", linewidths=0.5, ax=ax)
        ax.set_title("Eksik Değer Korelasyon Grafiği", fontsize=10)
        return fig

