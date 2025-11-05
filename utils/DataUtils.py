
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
)

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
            protected_cols: list[str] | None = None,  # <<< YENİ
            protect_id_like_names: bool = True,  # <<< YENİ
    ) -> pd.DataFrame:
        out = df.copy()
        if not datetime_guess:
            return out

        # id benzeri kolon isimleri (heuristic)
        id_like = {"id", "key", "user_id", "customer_id", "kod", "code"}
        protected = set((protected_cols or []))
        if protect_id_like_names:
            for c in out.columns:
                if str(c).strip().lower() in id_like:
                    protected.add(c)

        for col in out.columns:
            s = out[col]

            # Korumalı kolonları hiç ellemeyelim
            if col in protected:
                continue

            if s.dtype == "object":
                dt_parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=True)
                if dt_parsed.notna().mean() >= coerce_threshold:
                    if normalize_tz_to_naive_utc:
                        try:
                            dt_parsed = dt_parsed.dt.tz_convert("UTC").dt.tz_localize(None)
                        except Exception:
                            try:
                                dt_parsed = dt_parsed.dt.tz_localize(None)
                            except Exception:
                                pass
                    out[col] = dt_parsed
                    continue

                num_parsed = pd.to_numeric(s, errors="coerce")
                if num_parsed.notna().mean() >= coerce_threshold:
                    out[col] = num_parsed
                    continue

            # varolan dt’leri normalize et (opsiyonel)
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
    def _cast_series_pair(sL: pd.Series, sR: pd.Series, mode: str) -> tuple[pd.Series, pd.Series]:
        m = (mode or "auto").lower()
        if m == "string":
            return sL.astype("string"), sR.astype("string")
        if m == "numeric":
            return pd.to_numeric(sL, errors="coerce"), pd.to_numeric(sR, errors="coerce")
        if m == "datetime":
            return pd.to_datetime(sL, errors="coerce", utc=True), pd.to_datetime(sR, errors="coerce", utc=True)

        # auto: önce datetime, sonra numeric, değilse string
        try:
            dL = pd.to_datetime(sL, errors="coerce", utc=True)
            dR = pd.to_datetime(sR, errors="coerce", utc=True)
            if dL.notna().mean() > 0.8 and dR.notna().mean() > 0.8:
                return dL, dR
        except Exception:
            pass
        try:
            nL = pd.to_numeric(sL, errors="coerce")
            nR = pd.to_numeric(sR, errors="coerce")
            if nL.notna().mean() > 0.8 and nR.notna().mean() > 0.8:
                return nL, nR
        except Exception:
            pass
        return sL.astype("string"), sR.astype("string")

    @staticmethod
    def align_dtypes_for_merge_lr(
            df_left: pd.DataFrame,
            df_right: pd.DataFrame,
            left_on: list[str],
            right_on: list[str],
            key_cast_seq: list[str] | None = None,  # her eşleşme için: 'auto'|'string'|'numeric'|'datetime'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(left_on) != len(right_on):
            raise ValueError("left_on ve right_on uzunlukları eşit olmalı.")
        L, R = df_left.copy(), df_right.copy()
        if key_cast_seq is None:
            key_cast_seq = ["auto"] * len(left_on)
        if len(key_cast_seq) != len(left_on):
            raise ValueError("key_cast_seq uzunluğu anahtar sayısıyla eşit olmalı.")

        for (lc, rc, mode) in zip(left_on, right_on, key_cast_seq):
            if lc not in L.columns or rc not in R.columns:
                # eksik kolon varsa string’e çevirip yine de ilerleyelim
                if lc in L.columns:
                    L[lc] = L[lc].astype("string")
                if rc in R.columns:
                    R[rc] = R[rc].astype("string")
                continue
            sL, sR = L[lc], R[rc]
            cL, cR = DataUtils._cast_series_pair(sL, sR, mode)
            L[lc], R[rc] = cL, cR
        return L, R

    @staticmethod
    def merge_safe_lr(
            df_left: pd.DataFrame,
            df_right: pd.DataFrame,
            left_on: list[str],
            right_on: list[str],
            how: str = "inner",
            suffixes: tuple[str, str] = ("_x", "_y"),
            key_cast_seq: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Farklı isimli anahtar kolonları eşleştirerek güvenli merge.
        - left_on / right_on: eşleşen kolon listeleri (aynı sıradaki elemanlar eşleştirilir)
        - key_cast_seq: her eşleşme için tip stratejisi ('auto'|'string'|'numeric'|'datetime')
        """
        if not left_on or not right_on:
            raise ValueError("left_on / right_on boş olamaz.")
        if len(left_on) != len(right_on):
            raise ValueError("left_on ve right_on uzunlukları eşit olmalı.")

        L, R = DataUtils.align_dtypes_for_merge_lr(df_left, df_right, left_on, right_on, key_cast_seq)
        return pd.merge(L, R, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes)

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
    def value_counts_frame(df: pd.DataFrame, col: str, top: int = 20) -> pd.DataFrame:
        vc = df[col].astype("object").value_counts(dropna=False).head(top)
        return vc.rename_axis(col).reset_index(name="count")

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

    # --- Tek kolon profili (Variables paneli için) -------------------------------
    @staticmethod
    def variable_profile(df: pd.DataFrame, col: str, bins: int = 40) -> dict:
        s = df[col]
        n = int(s.shape[0])
        dtype = str(s.dtype)
        mem_mb = float(s.memory_usage(deep=True)) / (1024 ** 2)

        # ortak metrikler
        missing = int(s.isna().sum())
        missing_pct = float(missing) / max(n, 1) * 100.0
        distinct = int(s.nunique(dropna=True))
        distinct_pct = float(distinct) / max(n, 1) * 100.0

        out = {
            "dtype": dtype,
            "n": n,
            "missing": missing,
            "missing_pct": round(missing_pct, 4),
            "distinct": distinct,
            "distinct_pct": round(distinct_pct, 4),
            "mem_mb": round(mem_mb, 3),
            "min": None, "max": None, "mean": None, "std": None,
            "zeros": None, "zeros_pct": None,
            "neg": None, "neg_pct": None,
            "hist": None, "hist_edges": None
        }

        # numerik ise ek metrikler
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            out["min"] = None if x.dropna().empty else float(x.min())
            out["max"] = None if x.dropna().empty else float(x.max())
            out["mean"] = None if x.dropna().empty else float(x.mean())
            out["std"] = None if x.dropna().empty else float(x.std(ddof=1))

            zeros = int((x == 0).sum(skipna=True))
            neg = int((x < 0).sum(skipna=True))
            out["zeros"] = zeros
            out["neg"] = neg
            out["zeros_pct"] = float(zeros) / max(n, 1) * 100.0
            out["neg_pct"] = float(neg) / max(n, 1) * 100.0

            # histogram verisi (küçük grafik için)
            h, edges = np.histogram(x.dropna(), bins=bins)
            out["hist"], out["hist_edges"] = h.tolist(), edges.tolist()

        # datetime ise min / max
        elif pd.api.types.is_datetime64_any_dtype(s):
            sd = pd.to_datetime(s, errors="coerce")
            if not sd.dropna().empty:
                out["min"] = sd.min()
                out["max"] = sd.max()

        return out

    @staticmethod
    def variable_common_values(df: pd.DataFrame, col: str, top: int = 20) -> pd.DataFrame:
        """
        Kolondaki değerlerin tekrar sayısını ve yüzde frekansını döndürür.
        En sonda 'Other values' satırında kalan tüm değerler toplanır.
        """
        s = df[col].astype("string").fillna("NA")
        total = int(s.shape[0])

        # En sık 'top' değer
        vc = s.value_counts(dropna=False)  # Series (value -> count)
        top_vc = vc.head(top)

        # Others
        others_count = int(vc.iloc[top:].sum()) if vc.shape[0] > top else 0

        # Tablo
        out = top_vc.rename_axis("value").reset_index(name="count")
        out["freq_pct"] = (out["count"] / max(1, total)) * 100.0

        # 'Other values' satırını ekle
        if others_count > 0:
            out.loc[len(out)] = ["Other values", others_count, (others_count / max(1, total)) * 100.0]

        # Türleri netleştir
        out["count"] = out["count"].astype(int)
        out["freq_pct"] = out["freq_pct"].astype(float)

        return out

    # --- Tek kolon için tablo hazır istatistikler (Variables/Statistics sekmesi) ---
    @staticmethod
    def variable_quantile_table(s: pd.Series) -> pd.DataFrame:
        """Sayısal seriler için quantile özet tablosu."""
        s = pd.to_numeric(s, errors="coerce")
        q = {
            "Minimum": s.min(),
            "5-th percentile": s.quantile(0.05),
            "Q1": s.quantile(0.25),
            "median": s.median(),
            "Q3": s.quantile(0.75),
            "95-th percentile": s.quantile(0.95),
            "Maximum": s.max(),
        }
        q["Range"] = q["Maximum"] - q["Minimum"]
        q["Interquartile range (IQR)"] = q["Q3"] - q["Q1"]
        df = pd.DataFrame({"value": q}).reset_index().rename(columns={"index": ""})
        return df

    @staticmethod
    def variable_descriptive_table(s: pd.Series) -> pd.DataFrame:
        """Sayısal seriler için tanımlayıcı istatistik tablosu."""
        s = pd.to_numeric(s, errors="coerce")
        std = s.std(ddof=1)
        mean = s.mean()
        desc = {
            "Standard deviation": std,
            "Coefficient of variation (CV)": (std / mean) if pd.notna(std) and pd.notna(mean) and mean != 0 else np.nan,
            "Kurtosis": s.kurtosis(),
            "Mean": mean,
            "Median Absolute Deviation (MAD)": (s.mad() if hasattr(s, "mad") else (s - s.median()).abs().median()),
            "Skewness": s.skew(),
            "Sum": s.sum(),
            "Variance": s.var(ddof=1),
            "Monotonicity": (
                "Monotonic increasing" if s.is_monotonic_increasing else
                ("Monotonic decreasing" if s.is_monotonic_decreasing else "Not monotonic")
            ),
        }
        df = pd.DataFrame({"value": desc}).reset_index().rename(columns={"index": ""})
        return df

    @staticmethod
    def variable_top_categories(df: pd.DataFrame, col: str, top: int = 5) -> pd.DataFrame:
        """
        Kategorik bir kolonda en sık görülen top-N değerleri, count ve yüzde ile döndürür.
        En sona 'Other values' satırını ekler (varsa).
        """
        s = df[col].astype("string").fillna("NA")
        total = int(s.shape[0])
        vc = s.value_counts(dropna=False)  # value -> count (azalan)
        top_vc = vc.head(top)
        others = int(vc.iloc[top:].sum()) if vc.shape[0] > top else 0

        out = top_vc.rename_axis("value").reset_index(name="count")
        out["freq_pct"] = (out["count"] / max(1, total)) * 100.0
        if others > 0:
            out.loc[len(out)] = ["Other values …", others, (others / max(1, total)) * 100.0]

        out["count"] = out["count"].astype(int)
        out["freq_pct"] = out["freq_pct"].astype(float)
        return out

