
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import io
import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
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
    def read_any(file_name: str, file_bytes: bytes, **kwargs) -> pl.DataFrame:
        name = (file_name or "").lower()
        buffer = io.BytesIO(file_bytes)

        if name.endswith(".csv"):
            return pl.read_csv(buffer, **kwargs)
        if name.endswith(".parquet"):
            return pl.read_parquet(buffer, **kwargs)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            import openpyxl
            import pandas as pd
            df = pd.read_excel(buffer, **kwargs)
            return pl.from_pandas(df)
        raise ValueError(f"Desteklenmeyen uzantı: {name}")

    @staticmethod
    def sanitize_df(df: pl.DataFrame) -> pl.DataFrame:
        """
        "Unnamed" ile başlayan kolonları temizler ve tüm kolon adlarındaki
        gereksiz boşlukları kaldırır.
        """
        # Kolon isimlerini düzenle
        clean_cols = [str(c).strip() for c in df.columns]

        # Yeni kolon isimlerini uygula
        df = df.rename({old: new for old, new in zip(df.columns, clean_cols)})

        # "Unnamed" ile başlayanları filtrele
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(unnamed_cols)

        return df

    @staticmethod
    def infer_dtypes(
            df: pl.DataFrame,
            datetime_guess: bool = True,
            coerce_threshold: float = 0.8,
            normalize_tz_to_naive_utc: bool = False,
            protected_cols: list[str] | None = None,
            protect_id_like_names: bool = True,
    ) -> pl.DataFrame:
        """
        Kolon veri tiplerini otomatik olarak tahmin eder ve dönüştürür.
        - Tarih benzeri string kolonları datetime tipine çevirir.
        - Sayısal değerlere benzer kolonları float tipine çevirir.
        - 'id', 'code' gibi kolonlar koruma altındadır.
        """
        out = df.clone()

        if not datetime_guess:
            return out

        id_like = {"id", "key", "user_id", "customer_id", "kod", "code"}
        protected = set((protected_cols or []))

        # id benzeri kolonları koru
        if protect_id_like_names:
            for c in out.columns:
                if str(c).strip().lower() in id_like:
                    protected.add(c)

        for col in out.columns:
            if col in protected:
                continue

            s = out[col]

            # sadece Utf8 tipli kolonları dönüştürmeyi dene
            if s.dtype == pl.Utf8:
                # --- Tarih tipine dönüştürme ---
                dt_parsed = None
                common_formats = [
                    "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y",
                    "%Y.%m.%d", "%d.%m.%Y", "%Y%m%d",
                    "%d-%b-%Y", "%d %b %Y", "%b %d %Y",
                    "%Y-%m-%dT%H:%M:%S"
                ]

                for fmt in common_formats:
                    try:
                        dt_parsed = s.str.strptime(pl.Datetime, format=fmt, strict=False)
                        # başarı oranını ölç (%50 üzeri olursa kabul et)
                        if dt_parsed.drop_nulls().len() / max(1, s.len()) > 0.5:
                            break
                    except Exception:
                        continue

                if dt_parsed is not None and dt_parsed.drop_nulls().len() / max(1, s.len()) >= coerce_threshold:
                    if normalize_tz_to_naive_utc:
                        try:
                            dt_parsed = dt_parsed.dt.replace_time_zone(None)
                        except Exception:
                            pass
                    out = out.with_columns(dt_parsed.alias(col))
                    continue

                # --- Sayısal tahmin ---
                num_parsed = s.cast(pl.Float64, strict=False)
                num_valid_ratio = num_parsed.drop_nulls().len() / max(1, s.len())
                if num_valid_ratio >= coerce_threshold:
                    out = out.with_columns(num_parsed.alias(col))
                    continue

        return out

    @staticmethod
    def validate(df: pl.DataFrame, max_rows: int = 7_000_000, max_cols: int = 5_000) -> tuple[bool, str]:
        """
        DataFrame boyutlarını kontrol eder.
        Limitleri aşan durumlarda False ve hata mesajı döndürür.
        """
        n_rows, n_cols = df.height, len(df.columns)

        if n_rows > max_rows:
            return False, f"Satır sayısı çok büyük: {n_rows:,} > {max_rows:,}"
        if n_cols > max_cols:
            return False, f"Sütun sayısı çok büyük: {n_cols:,} > {max_cols:,}"

        return True, "OK"

    @staticmethod
    def _cast_series_pair(sL: pl.Series, sR: pl.Series, mode: str) -> tuple[pl.Series, pl.Series]:
        """
        İki serinin tipini belirtilen moda göre hizalar.
        - 'string': her iki seriyi de string'e çevirir.
        - 'numeric': sayısal tipe çevirir.
        - 'datetime': tarih formatına çevirir.
        - 'auto': önce datetime, sonra numeric, olmazsa string.
        """
        m = (mode or "auto").lower()

        if m == "string":
            return sL.cast(pl.Utf8, strict=False), sR.cast(pl.Utf8, strict=False)

        if m == "numeric":
            return sL.cast(pl.Float64, strict=False), sR.cast(pl.Float64, strict=False)

        if m == "datetime":
            return (
                sL.str.strptime(pl.Datetime, strict=False, utc=True),
                sR.str.strptime(pl.Datetime, strict=False, utc=True),
            )

        # AUTO: önce datetime, sonra numeric, değilse string
        try:
            dL = sL.str.strptime(pl.Datetime, strict=False, utc=True)
            dR = sR.str.strptime(pl.Datetime, strict=False, utc=True)
            if dL.drop_nulls().height / max(1, sL.height) > 0.8 and dR.drop_nulls().height / max(1, sR.height) > 0.8:
                return dL, dR
        except Exception:
            pass

        try:
            nL = sL.cast(pl.Float64, strict=False)
            nR = sR.cast(pl.Float64, strict=False)
            if nL.drop_nulls().height / max(1, sL.height) > 0.8 and nR.drop_nulls().height / max(1, sR.height) > 0.8:
                return nL, nR
        except Exception:
            pass

        return sL.cast(pl.Utf8, strict=False), sR.cast(pl.Utf8, strict=False)

    @staticmethod
    def align_dtypes_for_merge_lr(
            df_left: pl.DataFrame,
            df_right: pl.DataFrame,
            left_on: list[str],
            right_on: list[str],
            key_cast_seq: list[str] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        İki DataFrame'deki anahtar kolonların tiplerini hizalar.
        Her eşleşme için belirli bir cast stratejisi (auto/string/numeric/datetime) kullanılabilir.
        """
        if len(left_on) != len(right_on):
            raise ValueError("left_on ve right_on uzunlukları eşit olmalı.")

        L, R = df_left.clone(), df_right.clone()

        if key_cast_seq is None:
            key_cast_seq = ["auto"] * len(left_on)
        if len(key_cast_seq) != len(left_on):
            raise ValueError("key_cast_seq uzunluğu anahtar sayısıyla eşit olmalı.")

        for lc, rc, mode in zip(left_on, right_on, key_cast_seq):
            if lc not in L.columns or rc not in R.columns:
                # Eksik kolon varsa string olarak varsay
                if lc in L.columns:
                    L = L.with_columns(L[lc].cast(pl.Utf8, strict=False))
                if rc in R.columns:
                    R = R.with_columns(R[rc].cast(pl.Utf8, strict=False))
                continue

            sL, sR = L[lc], R[rc]
            cL, cR = DataUtils._cast_series_pair(sL, sR, mode)
            L = L.with_columns(cL.alias(lc))
            R = R.with_columns(cR.alias(rc))

        return L, R

    @staticmethod
    def merge_safe_lr(
            df_left: pl.DataFrame,
            df_right: pl.DataFrame,
            left_on: list[str],
            right_on: list[str],
            how: str = "inner",
            suffixes: tuple[str, str] = ("_x", "_y"),
            key_cast_seq: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Farklı isimli anahtar kolonları eşleştirerek güvenli bir merge işlemi gerçekleştirir.
        - left_on / right_on: eşleşen kolon listeleri
        - key_cast_seq: her eşleşme için tip stratejisi ('auto'|'string'|'numeric'|'datetime')
        """
        if not left_on or not right_on:
            raise ValueError("left_on / right_on boş olamaz.")
        if len(left_on) != len(right_on):
            raise ValueError("left_on ve right_on uzunlukları eşit olmalı.")

        # Önce tipleri hizala
        L, R = DataUtils.align_dtypes_for_merge_lr(df_left, df_right, left_on, right_on, key_cast_seq)

        # Polars join metodu
        # right_on kullanımı Polars’ta doğrudan left_on ile birlikte belirtilir
        joined = L.join(
            R,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffix=suffixes[1] if suffixes else "_right"
        )

        return joined

    @staticmethod
    def convert_column_type(df: pl.DataFrame, column: str, target_type: str) -> pl.DataFrame:
        """
        Seçili kolonu verilen hedef türe dönüştürür.
        Dönüşüm, tipler arası doğrudan (örn: Datetime->Date) veya
        otomatik format tanıma (örn: String->Date) yoluyla yapılır.
        """
        try:
            current_dtype = df[column].dtype

            # 1️⃣ String dönüşümü
            if target_type == "string":
                df = df.with_columns(pl.col(column).cast(pl.Utf8))

            # 2️⃣ Sayısal dönüşümler
            elif target_type == "int":
                df = df.with_columns(pl.col(column).cast(pl.Int64, strict=False))
            elif target_type == "float":
                df = df.with_columns(pl.col(column).cast(pl.Float64, strict=False))

            # 3️⃣ Boolean dönüşümü
            elif target_type == "boolean":
                df = df.with_columns(pl.col(column).cast(pl.Boolean, strict=False))

            # 4️⃣ Date dönüşümü (Sadece Yıl-Ay-Gün)
            elif target_type == "date":
                if current_dtype == pl.Datetime:
                    # 1. Mevcut tip Datetime ise, saati sil (Verimli)
                    df = df.with_columns(pl.col(column).cast(pl.Date))  # Bu zaten doğruydu
                elif current_dtype == pl.Date:
                    # 2. Zaten Date ise, dokunma
                    pass
                else:
                    # 3. Diğer (string, int) tiplerden geliyorsa, OTOMATİK parse et
                    df = df.with_columns(
                        pl.col(column)
                        .cast(pl.Utf8)
                        .str.strptime(pl.Datetime, format=None, strict=False)
                        .dt.date()
                    )

            # 5️⃣ Datetime dönüşümü (Tarih + Saat)
            elif target_type == "datetime":
                if current_dtype == pl.Date:
                    # 1. Mevcut tip Date ise, saat ekle (Verimli)

                    # [HATA DÜZELTMESİ 2]
                    # .dt.datetime() metodu Date tipi üzerinde çalışmaz.
                    # Doğru yöntem .cast(pl.Datetime) kullanmaktır.
                    df = df.with_columns(pl.col(column).cast(pl.Datetime))

                elif current_dtype == pl.Datetime:
                    # 2. Zaten Datetime ise, dokunma
                    pass
                else:
                    # 3. Diğer (string, int) tiplerden geliyorsa, OTOMATİK parse et
                    df = df.with_columns(
                        pl.col(column)
                        .cast(pl.Utf8)
                        .str.strptime(pl.Datetime, format=None, strict=False)
                    )

            return df

        except Exception as e:
            raise ValueError(f"Dönüşüm hatası ({column} -> {target_type}): {e}")

    @staticmethod
    def extract_date_parts(df: pl.DataFrame, column: str) -> pl.DataFrame:
        """
        [BONUS DÜZELTME] Docstring güncellendi.
        Tarih veya Tarih/Saat sütunundan yıl, ay, gün bilgilerini çıkarır.
        Sadece Datetime veya Date türü sütunlarda çalışır.
        """
        dtype = df[column].dtype

        # 1️⃣ Kontrol: Sütun datetime değilse anlamlı uyarı ver
        if dtype not in (pl.Datetime, pl.Date):
            raise TypeError(
                f"'{column}' sütunu {dtype} tipinde. "
                f"Yalnızca Datetime veya Date türlerinde tarih ayrıştırma yapılabilir."
            )

        # 2️⃣ Güvenli dönüşüm işlemleri
        try:
            df = df.with_columns([
                pl.col(column).dt.year().alias(f"{column}_year"),
                pl.col(column).dt.month().alias(f"{column}_month"),
                pl.col(column).dt.day().alias(f"{column}_day"),
            ])
            return df
        except Exception as e:
            raise ValueError(f"Tarih ayrıştırma hatası: {e}")

    @staticmethod
    def _bytes_to_mb(nbytes: int) -> float:
        """
        Byte değerini megabayt (MB) cinsine dönüştürür.
        """
        return round(nbytes / (1024 ** 2), 3)

    @staticmethod
    def profile(df: pl.DataFrame, sample_rows: int = 1000) -> DataProfile:
        """
        Polars DataFrame için profil çıkarımı yapar.
        - Satır/sütun sayısı
        - Bellek kullanımı
        - Eksik veri oranı
        - Kolon türleri (numerik, kategorik, datetime)
        - Örnek satırlar
        """
        n_rows, n_cols = df.height, len(df.columns)
        mem_mb = DataUtils._bytes_to_mb(df.estimated_size())

        # Eksik değer sayımı — güvenli versiyon
        missing_total = int(
            df.select(pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int64) for c in df.columns]))[0, 0])
        missing_ratio = float(missing_total) / float(max(1, n_rows * n_cols))

        # Tip sınıflandırması
        numeric_cols = [c for c, t in zip(df.columns, df.dtypes)
                        if t in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)]

        # pl.Date tipi de Tarih/Zaman olarak sınıflandırılmalı.
        datetime_cols = [c for c, t in zip(df.columns, df.dtypes) if t in (pl.Datetime, pl.Date)]

        categorical_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]

        sample = df.head(sample_rows)

        return DataProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            mem_usage_mb=mem_mb,
            missing_total=int(missing_total),
            missing_ratio=round(missing_ratio, 4),
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            sample=sample
        )

    # --- Tek kolon profili (Variables paneli için) -------------------------------
    @staticmethod
    def variable_profile(df: pl.DataFrame, col: str, bins: int = 40) -> dict:
        """
        Tek bir kolonun istatistiksel profilini çıkarır.
        - Sayısal kolonlar için: min, max, mean, std, histogram
        - Tarih kolonları için: min ve max tarih
        - Kategorik kolonlar için: distinct oranı
        """
        s = df[col]
        n = s.len()
        dtype = str(s.dtype)
        mem_mb = DataUtils._bytes_to_mb(s.estimated_size())

        # Eksik ve distinct metrikleri
        missing = int(s.null_count())
        missing_pct = round((missing / max(1, n)) * 100, 4)
        if s.dtype == pl.Utf8 and n > 500_000:
            distinct = int(s.approx_n_unique())
        else:
            distinct = int(s.n_unique())
        distinct_pct = round((distinct / max(1, n)) * 100, 4)

        out = {
            "dtype": dtype,
            "n": n,
            "missing": missing,
            "missing_pct": missing_pct,
            "distinct": distinct,
            "distinct_pct": distinct_pct,
            "mem_mb": round(mem_mb, 3),
            "min": None, "max": None, "mean": None, "std": None,
            "zeros": None, "zeros_pct": None,
            "neg": None, "neg_pct": None,
            "hist": None, "hist_edges": None
        }

        # --- Sayısal kolonlar ---
        if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
            s_nonnull = s.drop_nulls()
            if s_nonnull.len() > 0:
                out["min"] = float(s_nonnull.min())
                out["max"] = float(s_nonnull.max())
                out["mean"] = float(s_nonnull.mean())
                out["std"] = float(s_nonnull.std())

                zeros = int((s_nonnull == 0).sum())
                neg = int((s_nonnull < 0).sum())
                out["zeros"], out["neg"] = zeros, neg
                out["zeros_pct"] = (zeros / max(1, n)) * 100
                out["neg_pct"] = (neg / max(1, n)) * 100

                # Histogram (numpy uyumlu)
                import numpy as np
                h, edges = np.histogram(s_nonnull.to_numpy(), bins=bins)
                out["hist"], out["hist_edges"] = h.tolist(), edges.tolist()

        # --- Tarih kolonları ---
        elif s.dtype == pl.Datetime:
            s_nonnull = s.drop_nulls()
            if s_nonnull.len() > 0:
                out["min"] = s_nonnull.min()
                out["max"] = s_nonnull.max()

        return out

    @staticmethod
    def variable_common_values(df: pl.DataFrame, col: str, top: int = 20) -> pl.DataFrame:
        """
        Kolondaki en sık görülen 'top' değerleri ve yüzdelik oranlarını döndürür.
        'Other values' satırında kalan tüm değerler özetlenir.
        """
        s = df[col].cast(pl.Utf8).fill_null("NA")
        total = s.len()

        # Değerlerin frekans sayımı
        vc = s.value_counts(sort=True)
        top_vc = vc.head(top)

        top_count_sum = int(top_vc["count"].sum()) if top_vc.height > 0 else 0
        others_count = int(total - top_count_sum) if vc.height > top else 0

        # Frekans oranlarını hesapla
        out = top_vc.with_columns(
            (pl.col("count") / max(1, total) * 100).alias("freq_pct")
        ).rename({col: "value"})

        # ✅ Tüm kolon tiplerini Int64 + Float64 olarak sabitle
        out = out.with_columns([
            pl.col("count").cast(pl.Int64),
            pl.col("freq_pct").cast(pl.Float64)
        ])

        # 'Other values' satırını ekle
        if others_count > 0:
            other_row = pl.DataFrame({
                "value": ["Other values"],
                "count": [others_count],
                "freq_pct": [others_count / max(1, total) * 100]
            })

            # Aynı şemayı korumak için cast et
            other_row = other_row.with_columns([
                pl.col("count").cast(pl.Int64),
                pl.col("freq_pct").cast(pl.Float64)
            ])

            out = pl.concat([out, other_row])

        return out

    # --- Tek kolon için tablo hazır istatistikler (Variables/Statistics sekmesi) ---
    @staticmethod
    def variable_quantile_table(s: pl.Series) -> pl.DataFrame:
        """
        Sayısal seriler için quantile (dağılım) özet tablosu döndürür.
        - Minimum, Q1, Median, Q3, Maksimum
        - Range ve IQR hesapları dahil
        """
        s = s.cast(pl.Float64, strict=False).drop_nulls()

        if s.is_empty():
            return pl.DataFrame({"": [], "value": []})

        q = {
            "Minimum": s.min(),
            "5-th percentile": s.quantile(0.05, interpolation="nearest"),
            "Q1": s.quantile(0.25, interpolation="nearest"),
            "median": s.median(),
            "Q3": s.quantile(0.75, interpolation="nearest"),
            "95-th percentile": s.quantile(0.95, interpolation="nearest"),
            "Maximum": s.max(),
        }

        q["Range"] = q["Maximum"] - q["Minimum"]
        q["Interquartile range (IQR)"] = q["Q3"] - q["Q1"]

        df = pl.DataFrame({
            "": list(q.keys()),
            "value": [float(v) if v is not None else None for v in q.values()]
        })

        return df

    @staticmethod
    def variable_descriptive_table(s: pl.Series) -> pl.DataFrame:
        """
        Sayısal seriler için tanımlayıcı istatistik tablosu oluşturur.
        - Ortalama, varyans, standart sapma, çarpıklık, basıklık vb.
        """
        s = s.cast(pl.Float64, strict=False).drop_nulls()

        if s.is_empty():
            return pl.DataFrame({"": [], "value": []})

        std = s.std()
        mean = s.mean()
        variance = s.var()
        sum_val = s.sum()
        mad = (s - s.median()).abs().median()

        n = s.len()
        if n > 2:
            centered = s - mean
            skew = float((centered ** 3).mean() / (std ** 3)) if std not in (0, None) else None
            kurt = float((centered ** 4).mean() / (std ** 4)) - 3 if std not in (0, None) else None
        else:
            skew, kurt = None, None

        is_inc = s.is_sorted()
        is_dec = s[::-1].is_sorted()
        monotonicity = (
            "Monotonic increasing" if is_inc else
            ("Monotonic decreasing" if is_dec else "Not monotonic")
        )

        desc = {
            "Standard deviation": std,
            "Coefficient of variation (CV)": (std / mean) if mean not in (0, None) else None,
            "Kurtosis": kurt,
            "Mean": mean,
            "Median Absolute Deviation (MAD)": mad,
            "Skewness": skew,
            "Sum": sum_val,
            "Variance": variance,
            "Monotonicity": monotonicity,
        }

        # ✅ Polars strict=False veya tümünü stringe çevir
        df = pl.DataFrame({
            "": list(desc.keys()),
            "value": [str(v) if v is not None else "—" for v in desc.values()]
        })

        return df

    @staticmethod
    def correlation_matrix(df: pl.DataFrame) -> pl.DataFrame:
        """
        Sayısal değişkenler için korelasyon matrisini döndürür (Polars vektörel).
        """
        # Sadece sayısal sütunları seç
        numeric_cols = [c for c, dtype in zip(df.columns, df.dtypes)
                        if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]

        if not numeric_cols:
            return pl.DataFrame({"column": [], "message": ["Sayısal değişken bulunamadı."]})

        # Polars 0.20+ sürümü için correlation_matrix
        corr = df.select(numeric_cols).to_pandas().corr(method="pearson")
        corr_df = pl.DataFrame(corr.reset_index(names="column"))
        return corr_df

    @staticmethod
    def missing_value_summary(df: pl.DataFrame) -> pl.DataFrame:
        """
        Her kolon için eksik değer sayısı ve oranını hesaplar (Polars vektörel).
        """
        n_rows = df.height

        summary = (
            df.select([
                pl.col(c).is_null().sum().alias(c)
                for c in df.columns
            ])
            .transpose(include_header=True, header_name="column", column_names=["missing_count"])
            .with_columns([
                (pl.col("missing_count") / n_rows * 100).alias("missing_pct")
            ])
            .sort("missing_pct", descending=True)
        )
        return summary

    # Eksik Değerlerin Doldurulması
    @staticmethod
    def get_missing_columns(df: pl.DataFrame) -> list[str]:
        """
        Eksik değer (null) içeren kolonları döndürür.
        """
        null_counts = df.null_count().to_dicts()[0]
        return [c for c, v in null_counts.items() if v > 0]

    # === 🧩 2. Tüm Doldurma Yöntemleri (Tek Nokta Tanımı) ===
    @staticmethod
    def get_fill_methods() -> dict[str, str]:
        """
        Mevcut tüm doldurma yöntemlerini (anahtar + açıklama) döndürür.
        UI ve dahili işlem mantığı bu sözlükten beslenir.
        """
        return {
            "specific": "🪄 Belirli bir değerle doldur",
            "forward": "➡️ İleri yönlü doldur (ffill)",
            "backward": "⬅️ Geri yönlü doldur (bfill)",
            "mean": "📊 Ortalama ile doldur",
            "median": "📈 Medyan ile doldur",
            "mode": "🔁 Mod (en sık görülen) ile doldur",
            "zero": "0️⃣ Sıfır (0) ile doldur",
            "min": "🔽 Minimum değerle doldur",
            "max": "🔼 Maksimum değerle doldur",
            "custom": "✏️ Sabit (manuel) değerle doldur",
        }

    # === 🧩 3. Tip Bazlı Uygun Yöntem Önerisi ===
    @staticmethod
    def suggest_fill_methods(dtype: pl.DataType) -> list[str]:
        """
        Veri tipine göre uygulanabilir doldurma yöntemlerini döndürür.
        """
        if dtype in (pl.Int64, pl.Float64):
            return ["mean", "median", "mode", "min", "max", "zero", "custom"]
        elif dtype in (pl.Utf8, pl.Boolean):
            return ["mode", "custom"]
        elif dtype in (pl.Date, pl.Datetime):
            return ["forward", "backward", "mode", "custom"]
        else:
            return ["custom"]

    # === 🧩 4. Doldurma Değeri Hesaplama (Metoda Göre) ===
    @staticmethod
    def compute_fill_value(df: pl.DataFrame, column: str, method: str, custom_value=None):
        """
        Kolon ve seçilen metoda göre doldurma değerini hesaplar.
        None dönerse doldurma yapılmaz (örneğin tüm değerler null ise).
        """
        s = df[column]

        if s.null_count() == len(s):
            # Kolon tamamen boşsa hiçbir şey yapılmaz
            return None

        if method == "mean":
            val = s.mean()
        elif method == "median":
            val = s.median()
        elif method == "mode":
            modes = s.drop_nulls().mode().to_list()
            val = modes[0] if modes else None
        elif method == "min":
            val = s.min()
        elif method == "max":
            val = s.max()
        elif method == "zero":
            val = 0
        elif method in ("specific", "custom"):
            val = custom_value
        else:
            raise ValueError(f"Desteklenmeyen doldurma yöntemi: {method}")

        # Eğer sonuç hala None ise, None döndür (fill_missing uyarı verecek)
        return val

    @staticmethod
    def fill_missing(df: pl.DataFrame, column: str, method: str, custom_value=None) -> pl.DataFrame:
        """
        Seçilen kolonun eksik değerlerini belirtilen metoda göre doldurur.
        Polars 1.x uyumludur (fill_null(strategy) yerine forward_fill/backward_fill).
        """
        col_expr = pl.col(column)

        try:
            # 1️⃣ İleri / geri doldurma
            if method == "forward":
                expr = col_expr.forward_fill()
            elif method == "backward":
                expr = col_expr.backward_fill()
            else:
                fill_val = DataUtils.compute_fill_value(df, column, method, custom_value)

                if fill_val is None:
                    # Eğer hesaplanabilir bir değer yoksa işlem yapma
                    print(f"[UYARI] '{column}' için {method} yöntemiyle doldurma değeri hesaplanamadı. "
                          f"Kolon tamamen boş olabilir.")
                    return df  # no-op

                expr = col_expr.fill_null(fill_val)

            return df.with_columns(expr)

        except Exception as e:
            raise ValueError(f"{column} sütununda doldurma hatası: {e}")




