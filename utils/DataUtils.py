# utils/DataUtils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
)

SUPPORTED_EXTS = (".csv", ".xlsx", ".xls", ".parquet")

@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    mem_usage_mb: float
    missing_total: int
    missing_ratio: float
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]
    sample: pd.DataFrame

class DataUtils:
    @staticmethod
    def read_any(file_name: str, file_bytes: bytes, **kwargs) -> pd.DataFrame:
        name = (file_name or "").lower()
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes), **kwargs)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(file_bytes), **kwargs)
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(file_bytes), **kwargs)
        raise ValueError(f"Desteklenmeyen uzantı: {name}")

    @staticmethod
    def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
        # "Unnamed: 0" gibi artifakt kolonları temizle, whitespace kırp
        df = df.copy()
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def infer_dtypes(
            df: pd.DataFrame,
            datetime_guess: bool = True,
            coerce_threshold: float = 0.8,
            normalize_tz_to_naive_utc: bool = False,
    ) -> pd.DataFrame:
        """
        - object sütunlarını datetime/numeric'e olabildiğince çevirir.
        - tz'li datetime'ları (opsiyonel) UTC-naive'e indirger.
        """
        out = df.copy()

        if not datetime_guess:
            return out

        for col in out.columns:
            s = out[col]

            # Sadece object tipleri üzerinde otomatik tahmin yap
            if s.dtype == "object":
                # 1) Datetime dene (coerce ile; başarılı oranı threshold'u geçerse kabul et)
                dt_parsed = pd.to_datetime(
                    s, errors="coerce", infer_datetime_format=True, utc=True
                )
                dt_ratio = dt_parsed.notna().mean()

                if dt_ratio >= coerce_threshold:
                    # İstenirse tz'yi tamamen kaldır (UTC-naive)
                    if normalize_tz_to_naive_utc:
                        try:
                            dt_parsed = dt_parsed.dt.tz_convert("UTC").dt.tz_localize(None)
                        except Exception:
                            # zaten UTC ise veya çevrilemezse sessizce geç
                            try:
                                dt_parsed = dt_parsed.dt.tz_localize(None)
                            except Exception:
                                pass
                    out[col] = dt_parsed
                    continue

                # 2) Numeric dene (opsiyonel): çok metinli alanlarda gerekmez ama faydalı olabilir
                num_parsed = pd.to_numeric(s, errors="coerce")
                num_ratio = num_parsed.notna().mean()
                if num_ratio >= coerce_threshold:
                    out[col] = num_parsed
                    continue

            # Zaten datetime ise (tz’li veya tz’siz), isteğe bağlı normalize et
            if is_datetime64_any_dtype(s) or is_datetime64tz_dtype(s):
                if normalize_tz_to_naive_utc and is_datetime64tz_dtype(s):
                    try:
                        out[col] = s.dt.tz_convert("UTC").dt.tz_localize(None)
                    except Exception:
                        try:
                            out[col] = s.dt.tz_localize(None)
                        except Exception:
                            pass

        return out

    @staticmethod
    def validate(df: pd.DataFrame, max_rows: int = 7_000_000, max_cols: int = 5_000) -> Tuple[bool, str]:
        if df.shape[0] > max_rows:
            return False, f"Satır sayısı çok büyük: {df.shape[0]:,} > {max_rows:,}"
        if df.shape[1] > max_cols:
            return False, f"Sütun sayısı çok büyük: {df.shape[1]:,} > {max_cols:,}"
        return True, "OK"

    @staticmethod
    def suggest_join_keys(df_left: pd.DataFrame, df_right: pd.DataFrame, max_candidates: int = 5):
        """
        Ortak kolon adlarına bakıp, non-null oranı ve benzersiz oranı yüksek olanları önerir.
        """
        commons = [c for c in df_left.columns if c in df_right.columns]
        if not commons:
            return []
        # Heuristik: yüksek doluluk, çok az/çok fazla benzersiz olmayan kolonları ele
        scored = []
        for c in commons:
            sL, sR = df_left[c], df_right[c]
            fill_rate = (1 - sL.isna().mean()) * (1 - sR.isna().mean())
            try:
                nunL = sL.nunique(dropna=True);
                nunR = sR.nunique(dropna=True)
                uniq_score = 1.0 - abs((nunL / max(len(sL), 1)) - (nunR / max(len(sR), 1)))
            except Exception:
                uniq_score = 0.5
            # isim bazlı “id/key” bonusu
            name_bonus = 0.15 if str(c).lower() in {"id", "key", "user_id", "customer_id", "kod", "code"} else 0.0
            score = 0.6 * fill_rate + 0.4 * uniq_score + name_bonus
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:max_candidates]]

    @staticmethod
    def align_dtypes_for_merge(df_left: pd.DataFrame, df_right: pd.DataFrame, on: list[str]):
        L, R = df_left.copy(), df_right.copy()
        for c in on:
            if c not in L.columns or c not in R.columns:
                continue
            sL, sR = L[c], R[c]
            # Her iki taraf datetime'a çevrilebilir mi?
            try:
                dL = pd.to_datetime(sL, errors="coerce", utc=True)
                dR = pd.to_datetime(sR, errors="coerce", utc=True)
                if dL.notna().mean() > 0.8 and dR.notna().mean() > 0.8:
                    L[c] = dL;
                    R[c] = dR
                    continue
            except Exception:
                pass
            # Numerik mi?
            try:
                nL = pd.to_numeric(sL, errors="coerce");
                nR = pd.to_numeric(sR, errors="coerce")
                if nL.notna().mean() > 0.8 and nR.notna().mean() > 0.8:
                    L[c] = nL;
                    R[c] = nR
                    continue
            except Exception:
                pass
            # Ortak payda: string
            L[c] = sL.astype("string")
            R[c] = sR.astype("string")
        return L, R

    @staticmethod
    def merge_safe(
            df_left: pd.DataFrame,
            df_right: pd.DataFrame,
            on: list[str],
            how: str = "inner",
            suffixes: tuple[str, str] = ("_x", "_y")
    ) -> pd.DataFrame:
        if not on:
            raise ValueError("Merge için en az bir 'on' anahtarı seçin.")
        L, R = DataUtils.align_dtypes_for_merge(df_left, df_right, on)
        return pd.merge(L, R, on=on, how=how, suffixes=suffixes)

    @staticmethod
    def concat_safe(
            dfs: list[pd.DataFrame],
            column_mode: str = "union",  # 'union' (outer) | 'intersection' (inner)
            add_source_label: bool = False,
            source_names: list[str] | None = None
    ) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        if column_mode not in {"union", "intersection"}:
            raise ValueError("column_mode 'union' veya 'intersection' olmalı.")
        if column_mode == "intersection":
            common = set(dfs[0].columns)
            for d in dfs[1:]:
                common &= set(d.columns)
            dfs = [d[list(common)].copy() for d in dfs]
            join = "inner"
        else:
            join = "outer"
        if add_source_label:
            if source_names is None or len(source_names) != len(dfs):
                source_names = [f"src_{i + 1}" for i in range(len(dfs))]
            out = []
            for d, nm in zip(dfs, source_names):
                dd = d.copy()
                dd["__source__"] = nm
                out.append(dd)
            return pd.concat(out, axis=0, join=join, ignore_index=True)
        return pd.concat(dfs, axis=0, join=join, ignore_index=True)

    @staticmethod
    def _bytes_to_mb(nbytes: int) -> float:
        return round(nbytes / (1024 ** 2), 3)

    @staticmethod
    def profile(df: pd.DataFrame, sample_rows: int = 1000) -> DataProfile:
        n_rows, n_cols = df.shape
        mem_mb = float(df.memory_usage(index=True, deep=True).sum()) / (1024 ** 2)
        missing_total = int(df.isna().sum().sum())
        missing_ratio = float(missing_total) / float(n_rows * n_cols) if n_rows and n_cols else 0.0

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]

        sample = df.head(sample_rows)
        return DataProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            mem_usage_mb=round(mem_mb, 3),
            missing_total=missing_total,
            missing_ratio=round(missing_ratio, 4),
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            sample=sample
        )

    @staticmethod
    def quick_stats(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        target = df[cols] if cols else df.select_dtypes(include=[np.number])
        return target.describe(include="all").T

    @staticmethod
    def value_counts_frame(df: pd.DataFrame, col: str, top: int = 20) -> pd.DataFrame:
        vc = df[col].astype("object").value_counts(dropna=False).head(top)
        return vc.rename_axis(col).reset_index(name="count")

    @staticmethod
    def _hist_numeric(
            df: pd.DataFrame,
            cols: Optional[list[str]] = None,
            bins: int = 40,
            density: bool = False,
            max_cols: int = 12,
            fig_size: tuple[float, float] = (6.0, 3.5)
    ) -> None:
        nums = df.select_dtypes(include=[np.number]).columns.tolist() if cols is None else list(cols)
        if not nums:
            st.info("Histogram için numerik sütun bulunamadı.")
            return
        nums = nums[:max_cols]  # gereksiz kalabalığı engelle

        st.markdown("### Sayısal Dağılımlar (Histogram)")
        for col in nums:
            s = df[col].dropna()
            if s.empty:
                continue
            # tür güvenliği
            if not np.issubdtype(s.dtype, np.number):
                try:
                    s = s.astype(float)
                except Exception:
                    continue
            fig, ax = plt.subplots(figsize=fig_size)
            ax.hist(s, bins=bins, density=density)
            ax.set_title(f"{col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Yoğunluk" if density else "Frekans")
            ax.grid(alpha=0.2)
            st.pyplot(fig, clear_figure=True)

    @staticmethod
    def _bar_categorical(
            df: pd.DataFrame,
            cols: Optional[list[str]] = None,
            top: int = 20,
            max_cols: int = 12,
            fig_size: tuple[float, float] = (6.0, 3.5)
    ) -> None:
        cats_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cats = cats_all if cols is None else list(cols)
        cats = [c for c in cats if c in cats_all][:max_cols]

        if not cats:
            st.info("Kategorik sütun bulunamadı.")
            return

        st.markdown("### Kategorik Dağılımlar (Top-N)")
        for col in cats:
            vc = df[col].astype("string").fillna("NA").value_counts(dropna=False).head(top)
            if vc.empty:
                continue
            fig, ax = plt.subplots(figsize=fig_size)
            vc.plot(kind="bar", ax=ax)
            ax.set_title(f"{col} · en sık {top}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frekans")
            ax.tick_params(axis='x', labelrotation=45)
            ax.grid(axis='y', alpha=0.2)
            st.pyplot(fig, clear_figure=True)

    @staticmethod
    def _corr_heatmap(
            df: pd.DataFrame,
            method: str = "pearson",
            max_cols: int = 25
    ) -> None:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            st.info("Korelasyon için en az iki numerik sütun gerekli.")
            return

        # Çok geniş matrislerde kolon sınırı (varyansa göre en bilgililer)
        variances = num_df.var(numeric_only=True).sort_values(ascending=False)
        cols = list(variances.dropna().index[:max_cols])
        corr_df = num_df[cols].corr(method=method, numeric_only=True)

        st.markdown(f"### Korelasyon Isı Haritası ({method})")
        if corr_df.empty:
            st.info("Korelasyon hesaplanamadı.")
            return

        # Boyutları, kolon sayısına göre küçük tut
        w = min(10.0, 0.55 * len(cols) + 2.0)
        h = min(8.0, 0.55 * len(cols) + 1.5)

        fig, ax = plt.subplots(figsize=(w, h))
        im = ax.imshow(corr_df.values, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Tabloyu da görmek istersen:
        st.dataframe(corr_df, use_container_width=True, height=min(320, 24 * len(cols) + 80))

    @staticmethod
    def _datetime_overview(df: pd.DataFrame, max_cols: int = 8) -> None:
        dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        if not dt_cols:
            return
        st.markdown("### Tarih/Saat Sütun Özeti")
        for col in dt_cols[:max_cols]:
            s = df[col].dropna()
            if s.empty:
                st.write(f"**{col}**: boş")
                continue
            rng = s.max() - s.min()
            rng_days = rng.days if hasattr(rng, "days") else "NA"
            st.write(f"**{col}** → min: `{s.min()}` | max: `{s.max()}` | aralık (gün): `{rng_days}`")

    @staticmethod
    def detect_target_type(s: pd.Series, max_unique_categorical: int = 20) -> str:
        """
        Heuristik:
          - Numeric dtype ise ve benzersiz değer sayısı çok az değilse -> 'numeric'
          - Object/category ise veya numeric ama çok az unique varsa -> 'categorical'
        """
        if is_numeric_dtype(s):
            nunique = int(s.dropna().nunique())
            return "categorical" if nunique <= max_unique_categorical else "numeric"
        return "categorical"

    @staticmethod
    def target_profile_categorical(s: pd.Series) -> dict:
        vc = s.astype("string").fillna("NA").value_counts(dropna=False)
        n = int(s.shape[0])
        k = int(vc.shape[0])
        ratios = (vc / max(n, 1)).tolist()
        maj_ratio = float(vc.iloc[0] / max(n, 1)) if k > 0 else 0.0
        # Shannon entropy (bit)
        p = vc / max(n, 1)
        entropy = float(-(p * np.log2(p.replace(0, np.nan))).sum(skipna=True))
        return {
            "n": n, "k_classes": k, "majority_ratio": maj_ratio,
            "entropy_bits": round(entropy, 4),
            "value_counts": vc.reset_index().rename(columns={"index": "class", 0: "count"})
        }

    @staticmethod
    def target_profile_numeric(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce")
        desc = s.describe()
        out = {
            "n": int(desc.get("count", 0)),
            "mean": float(desc.get("mean", np.nan)),
            "std": float(desc.get("std", np.nan)),
            "min": float(desc.get("min", np.nan)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(desc.get("max", np.nan)),
        }
        return out

    @staticmethod
    def corr_with_target_numeric(df: pd.DataFrame, target_col: str, method: str = "pearson") -> pd.DataFrame:
        """
        Sadece numerik özellikler ile hedef (numeric) arasındaki korelasyon.
        """
        if target_col not in df.columns:
            return pd.DataFrame()
        num_df = df.select_dtypes(include=[np.number])
        if target_col not in num_df.columns:
            return pd.DataFrame()
        corr = num_df.corr(method=method, numeric_only=True)[target_col].drop(labels=[target_col])
        return corr.dropna().sort_values(ascending=False).reset_index().rename(
            columns={"index": "feature", target_col: "corr"})

    @staticmethod
    def anova_like_scores(df: pd.DataFrame, target_col: str, max_features: int = 50) -> pd.DataFrame:
        """
        Kategorik hedef için sayısal özelliklerde 'between/within variance' oranı (ANOVA-benzeri skor).
        Yüksek skor = sınıflar arası fark daha belirgin.
        """
        if target_col not in df.columns:
            return pd.DataFrame()
        y = df[target_col]
        X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in [target_col] if c in df.columns],
                                                       errors="ignore")
        if X.shape[1] == 0:
            return pd.DataFrame()

        groups = y.astype("string").fillna("NA")
        scores = []
        overall_means = X.mean(numeric_only=True)
        for col in X.columns[:max_features]:
            x = X[col].dropna()
            if x.empty:
                continue
            # grup istatistikleri
            joined = pd.DataFrame({"x": X[col], "g": groups}).dropna()
            if joined.empty:
                continue
            g_means = joined.groupby("g")["x"].mean()
            g_sizes = joined.groupby("g")["x"].size()
            overall = joined["x"].mean()
            # between variance (ağırlıklı)
            between = float(((g_means - overall) ** 2 * g_sizes).sum() / max(g_sizes.sum(), 1))
            # within variance
            within = float(joined.groupby("g")["x"].var(ddof=1).fillna(0).mean())
            score = between / (within + 1e-12)
            scores.append((col, score))

        if not scores:
            return pd.DataFrame()
        out = pd.DataFrame(scores, columns=["feature", "anova_like"])
        return out.sort_values("anova_like", ascending=False).reset_index(drop=True)

    @staticmethod
    def cramers_v(x: pd.Series, y: pd.Series) -> float:
        """
        Kategorik-kategorik ilişki gücü (0..1). Bias-correction uygulanır.
        """
        a = x.astype("string").fillna("NA")
        b = y.astype("string").fillna("NA")
        tbl = pd.crosstab(a, b)
        n = tbl.values.sum()
        if n == 0:
            return np.nan

        # beklenen frekanslar
        row_sum = tbl.sum(axis=1).values.reshape(-1, 1)
        col_sum = tbl.sum(axis=0).values.reshape(1, -1)
        expected = (row_sum @ col_sum) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((tbl.values - expected) ** 2 / np.where(expected == 0, np.nan, expected))

        phi2 = chi2 / n
        r, k = tbl.shape
        # bias correction (Bergsma)
        phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
        rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
        kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
        denom = max(1e-12, min(rcorr - 1, kcorr - 1))
        return float(np.sqrt(phi2corr / denom))

    @staticmethod
    def cate_cate_assoc(df: pd.DataFrame, target_col: str, max_features: int = 50) -> pd.DataFrame:
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cats = [c for c in cats if c != target_col]
        out = []
        for col in cats[:max_features]:
            v = DataUtils.cramers_v(df[col], df[target_col])
            kx = int(df[col].astype("string").nunique())
            ky = int(df[target_col].astype("string").nunique())
            out.append((col, v, kx, ky, int(len(df))))
        if not out:
            return pd.DataFrame()
        res = pd.DataFrame(out, columns=["feature", "cramers_v", "k_x", "k_target", "n"])
        return res.sort_values("cramers_v", ascending=False).reset_index(drop=True)

    @staticmethod
    def group_diff_scores_for_numeric_target(
            df: pd.DataFrame, target_col: str, max_features: int = 50
    ) -> pd.DataFrame:
        """
        Sayısal hedef ile KATEGORİK özelliklerin ayrıştırma gücü (between/within variance oranı).
        """
        if target_col not in df.columns:
            return pd.DataFrame()
        y = pd.to_numeric(df[target_col], errors="coerce")
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cats = [c for c in cats if c != target_col]
        scores = []
        for col in cats[:max_features]:
            joined = pd.DataFrame({"y": y, "g": df[col].astype("string")}).dropna()
            if joined.empty:
                continue
            g_means = joined.groupby("g")["y"].mean()
            g_sizes = joined.groupby("g")["y"].size()
            overall = joined["y"].mean()
            between = float(((g_means - overall) ** 2 * g_sizes).sum() / max(g_sizes.sum(), 1))
            within = float(joined.groupby("g")["y"].var(ddof=1).fillna(0).mean())
            score = between / (within + 1e-12)
            scores.append((col, score, int(len(g_means))))
        if not scores:
            return pd.DataFrame()
        res = pd.DataFrame(scores, columns=["feature", "anova_like", "n_categories"])
        return res.sort_values("anova_like", ascending=False).reset_index(drop=True)

    @staticmethod
    def normalized_contingency(df: pd.DataFrame, feat: str, target_col: str) -> pd.DataFrame:
        tbl = pd.crosstab(
            df[feat].astype("string").fillna("NA"),
            df[target_col].astype("string").fillna("NA")
        )
        if tbl.empty:
            return tbl
        row_pct = tbl.div(tbl.sum(axis=1).replace(0, np.nan), axis=0)
        return row_pct.fillna(0.0)
