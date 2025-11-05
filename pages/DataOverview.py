# pages/DataOverview.py
from __future__ import annotations
from dataclasses import asdict
from typing import Optional, List
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from utils.DataUtils import DataUtils
from utils.VizUtils import VizUtils

try:
    _IS_DARK = (st.get_option("theme.base") == "dark")
except Exception:
    _IS_DARK = False

@st.cache_data(show_spinner=False)
def cache_profile(_df: pd.DataFrame, dataset_name: str):
    return DataUtils.profile(_df)

class DataOverview:
    SESSION_KEY_DF = "__do_df"
    SESSION_KEY_NAME = "__do_name"
    SESSION_KEY_DATASETS = "__do_datasets"              # çoklu dosyalar için dict: {name: df}
    SESSION_KEY_DATASETS_META = "__do_datasets_meta"

    def _load_file(self, up) -> Optional[pd.DataFrame]:
        if not up:
            return None
        df = DataUtils.read_any(up.name, up.getvalue())
        df = DataUtils.sanitize_df(df)

        # --- KORUMA: id benzeri kolonları otomatik dönüştürme! ---
        protected = [c for c in df.columns if str(c).strip().lower() in {
            "id", "key", "user_id", "customer_id", "kod", "code"
        }]
        df = DataUtils.infer_dtypes(
            df,
            datetime_guess=True,
            protected_cols=protected,        # kritik: id'ler string kalsın
            protect_id_like_names=True
        )

        ok, msg = DataUtils.validate(df)
        if not ok:
            st.error(msg)
            return None
        return df

    def _reset_state(self):
        st.session_state.get(self.SESSION_KEY_DATASETS, {}).clear()
        st.session_state.get(self.SESSION_KEY_DATASETS_META, {}).clear()
        for k in [self.SESSION_KEY_DF, self.SESSION_KEY_NAME]:
            if k in st.session_state:
                del st.session_state[k]

    def render(self):
        st.header("📊 Veri Seti · Genel Bakış")

        with st.container(border=True):
            c1, c2 = st.columns([3, 2], vertical_alignment="center")
            ups = c1.file_uploader(
                "Dosya yükle (CSV / XLSX / Parquet)",
                type=["csv", "xlsx", "xls", "parquet"],
                accept_multiple_files=True
            )
            sample_n = c2.number_input("Head", min_value=5, max_value=500, value=50, step=5)
            clear = c2.button("Temizle", use_container_width=True)

        if self.SESSION_KEY_DATASETS not in st.session_state:
            st.session_state[self.SESSION_KEY_DATASETS] = {}  # {filename: df}
        if self.SESSION_KEY_DATASETS_META not in st.session_state:
            st.session_state[self.SESSION_KEY_DATASETS_META] = {}

        datasets = st.session_state[self.SESSION_KEY_DATASETS]
        meta = st.session_state[self.SESSION_KEY_DATASETS_META]

        if clear:
            datasets.clear()
            st.session_state.pop(self.SESSION_KEY_NAME, None)
            st.cache_data.clear()
            st.rerun()

        # --- Çoklu yükleme: boyut ve DF ekleme akışını düzelt ---
        if ups:
            for up in ups:
                if up is None:
                    continue
                # Aynı isimle gelirse üzerine yazmak istersen bu 'if' bloğunu kaldırabilirsin.
                # Şimdilik aynı isim gelirse atlıyoruz:
                if up.name in datasets:
                    continue
                # Boyutu güvenilir şekilde al
                size_bytes = getattr(up, "size", None)
                if size_bytes is None:
                    try:
                        size_bytes = up.getbuffer().nbytes
                    except Exception:
                        size_bytes = len(up.getvalue())
                df_new = self._load_file(up)
                if df_new is not None:
                    datasets[up.name] = df_new
                    meta[up.name] = {"size_bytes": int(size_bytes)}

        if not datasets:
            st.info("Bir veya daha fazla veri dosyası yüklediğinizde burada listelenecek.")
            return

        # === 🧩 Veri Birleştirme ===
        with st.expander("🧩 Veri Birleştirme (Join/Merge)", expanded=False):
            datasets = st.session_state[DataOverview.SESSION_KEY_DATASETS]
            ds_names = list(datasets.keys())
            if len(ds_names) < 2:
                st.info("Birleştirme için en az iki veri seti yükleyin.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    left_ds = st.selectbox("Left (sol) veri seti", options=ds_names, key="__merge_left")
                with c2:
                    right_ds = st.selectbox(
                        "Right (sağ) veri seti",
                        options=[n for n in ds_names if n != st.session_state.get("__merge_left")],
                        key="__merge_right"
                    )

                if left_ds and right_ds:
                    L, R = datasets[left_ds], datasets[right_ds]

                    st.markdown("### 1) Anahtar Kolonları Seç")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        left_cols = st.multiselect(
                            f"{left_ds} anahtar(lar)ı (sıra eşleşir)",
                            options=list(L.columns),
                            key="__left_keys"
                        )
                    with cc2:
                        right_cols = st.multiselect(
                            f"{right_ds} anahtar(lar)ı (sıra eşleşir)",
                            options=list(R.columns),
                            key="__right_keys"
                        )

                    # Eşleşme yardımcı notu
                    if left_cols and right_cols:
                        if len(left_cols) != len(right_cols):
                            st.warning("Sol ve sağ anahtar sayıları eşit olmalı. Sıra, eşleşmeyi belirler.")
                        else:
                            pairs = list(zip(left_cols, right_cols))
                            st.caption("Eşleşmeler: " + ", ".join([f"{l} ⇄ {r}" for l, r in pairs]))

                    st.markdown("### 2) Anahtar Tipi (Çift Bazlı)")
                    key_cast_seq = []
                    if left_cols and right_cols and len(left_cols) == len(right_cols):
                        for i, (lcol, rcol) in enumerate(zip(left_cols, right_cols), start=1):
                            sel = st.selectbox(
                                f"#{i} {lcol} ⇄ {rcol} tipi",
                                options=["string", "numeric", "datetime", "auto"],
                                index=0,  # güvenli varsayılan: string
                                key=f"__pair_cast_{i}"
                            )
                            key_cast_seq.append(sel)

                    st.markdown("### 3) Join Türü ve Sonekler")
                    c3, c4, c5 = st.columns([1, 1, 2])
                    with c3:
                        how = st.selectbox("Join türü", options=["inner", "left", "right", "outer"], index=1)
                    with c4:
                        sfx1 = st.text_input("Sol sonek", value="_x")
                    with c5:
                        sfx2 = st.text_input("Sağ sonek", value="_y")

                    target_name = st.text_input("Yeni veri seti adı", value="merged_join")

                    do_merge = st.button("Birleştir ve Kaydet", type="primary", use_container_width=True)
                    if do_merge:
                        if not left_cols or not right_cols:
                            st.error("Her iki taraftan da anahtar kolon(lar)ı seçin.")
                        elif len(left_cols) != len(right_cols):
                            st.error("Sol ve sağ anahtar sayıları eşit olmalı.")
                        else:
                            try:
                                out = DataUtils.merge_safe_lr(
                                    L, R,
                                    left_on=left_cols,
                                    right_on=right_cols,
                                    how=how,
                                    suffixes=(sfx1, sfx2),
                                    key_cast_seq=key_cast_seq if key_cast_seq else None
                                )
                            except Exception as e:
                                st.error(f"Birleştirme hatası: {e}")
                            else:
                                datasets[target_name] = out
                                st.session_state[DataOverview.SESSION_KEY_DATASETS_META][target_name] = {
                                    "size_bytes": 0}
                                st.cache_data.clear()
                                st.session_state[DataOverview.SESSION_KEY_NAME] = target_name
                                st.success(
                                    f"✔ {target_name} oluşturuldu · {out.shape[0]:,} satır × {out.shape[1]:,} sütun")
                                st.rerun()


        # Varsayılan olarak ilk anahtar seçili olsun
        default_name = st.session_state.get(self.SESSION_KEY_NAME) or list(datasets.keys())[0]
        st.session_state[self.SESSION_KEY_NAME] = default_name

        # callback: seçimi state'e yazsın (rerun otomatik)
        def _on_active_change():
            st.session_state[self.SESSION_KEY_NAME] = st.session_state["__active_name"]

        active_name = st.selectbox(
            "Aktif veri seti",
            options=list(datasets.keys()),
            index=list(datasets.keys()).index(default_name),
            key="__active_name",
            on_change=_on_active_change
        )

        # Seçimi state’e yaz
        st.session_state[self.SESSION_KEY_NAME] = active_name

        # Aktif DF
        df = datasets[active_name]
        name = active_name
        size_bytes = meta.get(name, {}).get("size_bytes", 0)

        # ---- Profil kartları
        prof = cache_profile(df, name)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Satır", f"{prof.n_rows:,}")
        m2.metric("Sütun", f"{prof.n_cols:,}")
        m3.metric("Dosya Boyut", "—" if not size_bytes else f"{DataUtils._bytes_to_mb(size_bytes)}")
        m4.metric("RAM (MB)", f"{prof.mem_usage_mb}")
        m5.metric("Eksik Oranı", f"{prof.missing_ratio*100:.2f}%")

        with st.expander("Sütun Türleri", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.write("**Numerik**")
            c1.write(", ".join(prof.numeric_cols) or "—")
            c2.write("**Kategorik**")
            c2.write(", ".join(prof.categorical_cols) or "—")
            c3.write("**Tarih/Zaman**")
            c3.write(", ".join(prof.datetime_cols) or "—")

        st.subheader("Örnek Kayıtlar")
        st.dataframe(prof.sample.head(int(sample_n)), use_container_width=True, height=400)

        # ---------------------------------------------------------------------------
        # === Variables (tek değişken odaklı kart) ===================================
        st.markdown("## Variables")

        col_sel, col_blank = st.columns([2, 1])
        with col_sel:
            var_col = st.selectbox("Select Columns", options=list(df.columns), key="__vars_sel")

        with st.container(border=True):
            if var_col:
                vp = DataUtils.variable_profile(df, var_col, bins=40)

                # Başlık
                st.markdown(f"### <span style='color:#2B6CB0'>{var_col}</span>", unsafe_allow_html=True)

                # Tür etiketi
                s = df[var_col]
                if pd.api.types.is_integer_dtype(s):
                    dtype_label = "Integer (I)"
                elif pd.api.types.is_float_dtype(s):
                    dtype_label = "Real number (R)"
                elif pd.api.types.is_datetime64_any_dtype(s):
                    dtype_label = "Datetime (D)"
                else:
                    dtype_label = "Categorical (C)"

                # Özet bilgileri
                n = vp["n"]
                non_null = int(n - vp["missing"])
                st.caption(
                    f"{dtype_label}"
                )

                # Üst metrik kartı: iki tablo + sağda mini histogram
                g1, g2, g3 = st.columns([1.2, 1.2, 1.4])

                # Sol tablo (Distinct/Missing/Infinite/Mean)
                with g1:
                    left_rows = {
                        "Distinct": f"{vp['distinct']:,}",
                        "Distinct (%)": f"{vp['distinct_pct']:.1f}%",
                        "Missing": f"{vp['missing']:,}",
                        "Missing (%)": f"{vp['missing_pct']:.1f}%",
                        "Infinite": "0",
                        "Infinite (%)": "0.0%",
                        "Mean": ("—" if vp["mean"] is None else f"{vp['mean']:.6g}"),
                    }
                    # infinite varsa sayısal seriden hesapla
                    s_tmp = pd.to_numeric(df[var_col], errors="coerce")
                    if pd.api.types.is_numeric_dtype(df[var_col]):
                        inf_cnt = int(np.isinf(s_tmp).sum())
                        left_rows["Infinite"] = f"{inf_cnt}"
                        left_rows["Infinite (%)"] = f"{(inf_cnt / max(1, vp['n'])) * 100:.1f}%"

                    left_df = pd.DataFrame({
                        "metric": list(left_rows.keys()),
                        "value": list(left_rows.values())
                    })
                    st.dataframe(
                        left_df,
                        use_container_width=True,
                        height=280,
                        hide_index=True
                    )

                # Orta tablo (Min/Max/Zeros/Negative/Memory size)
                with g2:
                    right_rows = {
                        "Minimum": ("—" if vp["min"] is None else f"{vp['min']}"),
                        "Maximum": ("—" if vp["max"] is None else f"{vp['max']}"),
                        "Zeros": ("—" if vp["zeros"] is None else f"{vp['zeros']:,}"),
                        "Zeros (%)": ("—" if vp["zeros_pct"] is None else f"{vp['zeros_pct']:.1f}%"),
                        "Negative": ("—" if vp["neg"] is None else f"{vp['neg']:,}"),
                        "Negative (%)": ("—" if vp["neg_pct"] is None else f"{vp['neg_pct']:.1f}%"),
                        "Memory size": f"{vp['mem_mb']:.2f} MiB",
                    }
                    right_df = pd.DataFrame({
                        "metric": list(right_rows.keys()),
                        "value": list(right_rows.values())
                    })
                    st.dataframe(
                        right_df,
                        use_container_width=True,
                        height=280,
                        hide_index=True
                    )

                # Sağ: mini histogram - kategorik dağılım
                with g3:
                    s_this = df[var_col]

                    if pd.api.types.is_numeric_dtype(s_this) and vp.get("hist") is not None and vp.get(
                            "hist_edges") is not None:
                        # Numerik → küçük histogram (Plotly)
                        fig = VizUtils.pretty_histogram(df[var_col], bins=60, title="Histogram", height=260,
                                                        dark=_IS_DARK)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    elif (pd.api.types.is_object_dtype(s_this) or pd.api.types.is_categorical_dtype(s_this) or s_this.dtype == "string"):
                        fig = VizUtils.top_categories_bar(df, var_col, top=6, height=260, title="Top categories",
                                                          dark=_IS_DARK)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    elif pd.api.types.is_datetime64_any_dtype(s_this):
                        fig = VizUtils.time_count_bar(df[var_col], freq="D", height=260, title="Daily counts",
                                                      dark=_IS_DARK)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Bu değişken için grafik uygun değil.")

                # Alt sekmeler: Statistics / Histogram / Common
                tab1, tab2, tab3 = st.tabs(["Statistics", "Histogram", "Common values"])

                with tab1:
                    s = df[var_col]
                    if pd.api.types.is_numeric_dtype(s):
                        qdf = DataUtils.variable_quantile_table(s)  # DataUtils tarafı hazır
                        ddf = DataUtils.variable_descriptive_table(s)

                        t1, t2 = st.columns(2)
                        with t1:
                            st.subheader("Quantile statistics")
                            st.dataframe(qdf, use_container_width=True, height=360, hide_index=True)
                        with t2:
                            st.subheader("Descriptive statistics")
                            st.dataframe(ddf, use_container_width=True, height=360, hide_index=True)
                    else:
                        st.info("Sayısal olmayan sütun için bu sekme sınırlıdır.")

                with tab2:
                    bins = st.slider("Histogram bins", 5, 120, 40, step=5, key="__vars_bins")
                    if pd.api.types.is_numeric_dtype(df[var_col]):
                        fig = VizUtils.pretty_histogram(
                            df[var_col], bins=int(bins), title=f"{var_col} · Histogram",
                            height=320, dark=_IS_DARK
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    elif pd.api.types.is_datetime64_any_dtype(df[var_col]):
                        fig = VizUtils.time_count_bar(
                            df[var_col], freq="D", height=320,
                            title=f"{var_col} · Daily counts", dark=_IS_DARK
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        fig = VizUtils.top_categories_bar(
                            df, var_col, top=30, height=320, title="Top values", dark=_IS_DARK
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                with tab3:
                    top = st.slider("Top-N", 5, 50, 20, step=5, key="__vars_top")
                    cv = DataUtils.variable_common_values(df, var_col, top=top)

                    st.dataframe(
                        cv,
                        use_container_width=True,
                        height=min(500, 30 * (len(cv) + 2)),
                        hide_index=True,
                        column_config={
                            "value": "Value",
                            "count": "Count",
                            "freq_pct": st.column_config.ProgressColumn(
                                "Frequency (%)",
                                help="Value frequency as percentage of total",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                    )

        # ---------------------------------------------------------------------------

        # === 🧪 Değişken Odaklı Analiz (Pro) ===
        st.markdown("## 🧪 Değişken Odaklı Analiz")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            feature = st.selectbox("Değişken seç", options=list(df.columns), key="__feature_focus")
        with col_right:
            target_opt = st.selectbox("Hedef (opsiyonel)", options=["(yok)"] + list(df.columns), index=0,
                                      key="__target_opt")

        # Küçük ayarlar
        with st.expander("Grafik Ayarları", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                bins_u = st.slider("Histogram bins", 5, 120, 40, step=5)
            with c2:
                top_n = st.slider("Kategorik Top-N", 3, 50, 20)
            with c3:
                corr_k = st.slider("Korelasyonda kolon üst sınır", 5, 50, 15)

        if feature:
            s = df[feature]
            is_num = pd.api.types.is_numeric_dtype(s)
            is_cat = pd.api.types.is_categorical_dtype(s) or s.dtype == "object"
            is_dt = pd.api.types.is_datetime64_any_dtype(s)

            st.subheader(f"📌 {feature} — Genel Özet")
            miss = int(s.isna().sum())
            miss_ratio = float(miss) / max(1, len(s)) * 100.0
            nunique = int(s.nunique(dropna=True))
            st.caption(f"Tür: **{str(s.dtype)}** · Eksik: **{miss} ({miss_ratio:.2f}%)** · Benzersiz: **{nunique}**")

            # === 1) Tek değişken görselleştirme ===
            st.markdown("### Tek Değişken Dağılımı")
            if is_num:
                fig = VizUtils.pretty_histogram(s, bins=int(bins_u), title=f"{feature} · Histogram",
                                                height=320, dark=_IS_DARK)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.dataframe(pd.DataFrame(pd.to_numeric(s, errors="coerce").describe()).T,
                             use_container_width=True, height=120, hide_index=True)
            elif is_cat:
                vc = s.astype("string").fillna("NA").value_counts(dropna=False)
                if not vc.empty:
                    tmp = pd.DataFrame({feature: s})
                    fig = VizUtils.top_categories_bar(tmp, feature, top=int(top_n), height=320,
                                                      title=f"{feature} · Top-{min(top_n, vc.shape[0])}")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            elif is_dt:
                fig = VizUtils.time_count_bar(s, freq="D", height=320, title=f"{feature} · Daily counts")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # === 2) İlişkisel analiz (opsiyonel hedef seçilirse) ===
            if target_opt and target_opt != "(yok)" and target_opt != feature:
                st.markdown("### Hedefe Göre İlişki")
                t = df[target_opt]
                t_is_num = pd.api.types.is_numeric_dtype(t)
                t_is_cat = pd.api.types.is_categorical_dtype(t) or t.dtype == "object"

                if is_num and t_is_num:
                    xy = df[[feature, target_opt]].dropna()
                    if not xy.empty:
                        fig, ax = plt.subplots(figsize=(6.0, 3.5))
                        ax.scatter(xy[feature], xy[target_opt], s=8, alpha=0.5)
                        ax.set_xlabel(feature); ax.set_ylabel(target_opt)
                        ax.set_title(f"{feature} vs {target_opt}")
                        ax.grid(alpha=0.2); st.pyplot(fig, clear_figure=True)

                elif is_num and t_is_cat:
                    g = df[[target_opt, feature]].dropna()
                    if not g.empty:
                        agg = g.groupby(target_opt)[feature].mean().sort_values(ascending=False).head(top_n)
                        fig, ax = plt.subplots(figsize=(6.0, 3.5))
                        agg.plot(kind="bar", ax=ax)
                        ax.set_title(f"{target_opt} gruplarına göre {feature} ortalaması")
                        ax.set_xlabel(target_opt); ax.set_ylabel(f"{feature} ort.")
                        ax.tick_params(axis="x", labelrotation=45); ax.grid(axis="y", alpha=0.2)
                        st.pyplot(fig, clear_figure=True)

                elif is_cat and t_is_num:
                    g = df[[feature, target_opt]].dropna()
                    if not g.empty:
                        agg = g.groupby(feature)[target_opt].mean().sort_values(ascending=False).head(top_n)
                        fig, ax = plt.subplots(figsize=(6.0, 3.5))
                        agg.plot(kind="bar", ax=ax)
                        ax.set_title(f"{feature} kategorilerine göre {target_opt} ortalaması")
                        ax.set_xlabel(feature); ax.set_ylabel(f"{target_opt} ort.")
                        ax.tick_params(axis="x", labelrotation=45); ax.grid(axis="y", alpha=0.2)
                        st.pyplot(fig, clear_figure=True)

                elif is_cat and t_is_cat:
                    tbl = pd.crosstab(df[feature].astype("string").fillna("NA"),
                                      df[target_opt].astype("string").fillna("NA"))
                    if not tbl.empty:
                        top_rows = tbl.sum(axis=1).sort_values(ascending=False).head(min(top_n, tbl.shape[0])).index
                        m = tbl.loc[top_rows]
                        row_pct = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
                        fig, ax = plt.subplots(
                            figsize=(min(10.0, 0.6 * row_pct.shape[1] + 2),
                                     min(6.0, 0.45 * row_pct.shape[0] + 2))
                        )
                        im = ax.imshow(row_pct.values, interpolation="nearest", aspect="auto")
                        ax.set_xticks(range(row_pct.shape[1])); ax.set_xticklabels(list(row_pct.columns), rotation=90, fontsize=8)
                        ax.set_yticks(range(row_pct.shape[0])); ax.set_yticklabels(list(row_pct.index), fontsize=8)
                        fig.colorbar(im, ax=ax, shrink=0.85)
                        ax.set_title(f"{feature} × {target_opt} (satır-normalize)")
                        fig.tight_layout(); st.pyplot(fig, clear_figure=True)

            # === 3) Hızlı korelasyon (yalnızca numerik değişken seçildiyse) ===
            if is_num:
                st.markdown("### Korelasyon (Hızlı)")
                num_df = df.select_dtypes(include=[np.number]).drop(columns=[feature], errors="ignore")
                if not num_df.empty:
                    corr = num_df.corr(numeric_only=True).get(feature, pd.Series(dtype=float))
                    if corr is None or corr.empty:
                        corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)[feature].drop(labels=[feature])
                    corr = corr.sort_values(key=lambda x: x.abs(), ascending=False).head(corr_k)
                    st.dataframe(corr.to_frame("corr"), use_container_width=True,
                                 height=min(360, 26 * max(1, len(corr)) + 20))

        # ---- Kategorik özet
        st.subheader("Kategorik Dağılım (Top 20)")
        cat = st.selectbox("Sütun seç", options=prof.categorical_cols or ["(kategorik yok)"])
        if prof.categorical_cols:
            vc = DataUtils.value_counts_frame(df, cat, top=20)
            st.dataframe(vc, use_container_width=True, height=360)

        # === 🎯 Hedef Değişken Analizi ===
        st.markdown("## 🎯 Hedef Değişken")

        target_col = st.selectbox("Hedef sütunu seç", options=list(df.columns))
        if target_col:
            tgt_series = df[target_col]
            tgt_type = DataUtils.detect_target_type(tgt_series)
            st.caption(f"Hedef tipi: **{tgt_type}**")

            missing_rate = float(tgt_series.isna().mean() * 100.0)
            st.caption(f"Eksik oranı: **{missing_rate:.2f}%**")

            with st.expander("Hedef Grafik Ayarları", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    bins_t = st.slider("Target bins", 5, 200, 40, step=5)
                with c2:
                    top_cats = st.slider("Kategorik top-N", 3, 50, 20)
                with c3:
                    max_feats = st.slider("Özellik sınırı (tablolar)", 10, 100, 30)
                with c4:
                    heat_top = st.slider("Isı haritası top-N kategori", 5, 30, 12)

            if tgt_type == "numeric":
                # Sayısal hedef vs KATEGORİK özellikler
                st.subheader("Hedef (sayısal) vs Kategorik Özellikler")
                cat_scores = DataUtils.group_diff_scores_for_numeric_target(df, target_col, max_features=max_feats)
                if not cat_scores.empty:
                    st.dataframe(cat_scores.head(max_feats), use_container_width=True, height=min(360, 24 * max_feats))
                    best_cat = st.selectbox(
                        "Kategorik özellik seç (hedef ortalaması grafiği için)",
                        options=cat_scores["feature"].tolist()[:max_feats],
                        key="__best_cat_numeric_target"
                    )
                    if best_cat:
                        g = df[[best_cat, target_col]].dropna()
                        if not g.empty:
                            agg = g.groupby(best_cat)[target_col].mean().sort_values(ascending=False).head(top_cats)
                            fig, ax = plt.subplots(figsize=(6.0, 3.5))
                            agg.plot(kind="bar", ax=ax)
                            ax.set_title(f"{best_cat} gruplarına göre {target_col} ortalaması (top-{min(top_cats, agg.shape[0])})")
                            ax.set_xlabel(best_cat); ax.set_ylabel(f"{target_col} ort.")
                            ax.tick_params(axis="x", labelrotation=45); ax.grid(axis="y", alpha=0.2)
                            st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Kategorik özellikler için anlamlı skor hesaplanamadı veya veri uygun değil.")

                st.subheader("Hedef Dağılımı")
                s = pd.to_numeric(tgt_series, errors="coerce").dropna()
                if not s.empty:
                    fig = VizUtils.pretty_histogram(s, bins=int(bins_t), title=f"{target_col} · Histogram",
                                                    height=320, dark=_IS_DARK)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("Hedef sayısal değerler üretmedi.")

                prof_t = DataUtils.target_profile_numeric(tgt_series)
                st.write(pd.DataFrame([prof_t]))

                st.subheader("Özellik Korelasyonları (Numerik → Hedef)")
                corr_tbl = DataUtils.corr_with_target_numeric(df, target_col, method="pearson")
                if not corr_tbl.empty:
                    st.dataframe(corr_tbl.head(max_feats), use_container_width=True, height=min(360, 24 * max_feats))
                    top3 = corr_tbl["feature"].head(3).tolist()
                    if top3:
                        st.caption("En güçlü 3 özellik için küçük scatter grafikleri")
                        for feat in top3:
                            xy = df[[feat, target_col]].dropna()
                            if xy.empty:
                                continue
                            fig, ax = plt.subplots(figsize=(6.0, 3.5))
                            ax.scatter(xy[feat], xy[target_col], s=6, alpha=0.5)
                            ax.set_xlabel(feat); ax.set_ylabel(target_col)
                            ax.set_title(f"{feat} vs {target_col}")
                            ax.grid(alpha=0.2); st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Numerik özellik korelasyonu hesaplanamadı.")

            else:
                # Kategorik hedef vs KATEGORİK özellikler (Cramér's V)
                st.subheader("Hedef (kategorik) vs Kategorik Özellikler")
                assoc = DataUtils.cate_cate_assoc(df, target_col, max_features=max_feats)
                if not assoc.empty:
                    st.dataframe(assoc.head(max_feats), use_container_width=True, height=min(360, 24 * max_feats))
                    top_feat = st.selectbox(
                        "Kategorik özellik seç (ısı haritası için)",
                        options=assoc["feature"].tolist()[:max_feats],
                        key="__top_feat_categorical_target"
                    )
                    if top_feat:
                        tbl = pd.crosstab(
                            df[top_feat].astype("string").fillna("NA"),
                            df[target_col].astype("string").fillna("NA")
                        )
                        top_rows = tbl.sum(axis=1).sort_values(ascending=False).head(heat_top).index
                        tbl_small = tbl.loc[top_rows]
                        if not tbl_small.empty:
                            row_pct = tbl_small.div(tbl_small.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
                            fig, ax = plt.subplots(
                                figsize=(min(10.0, 0.6 * row_pct.shape[1] + 2),
                                         min(6.0, 0.45 * row_pct.shape[0] + 2))
                            )
                            im = ax.imshow(row_pct.values, interpolation="nearest", aspect="auto")
                            ax.set_xticks(range(row_pct.shape[1])); ax.set_xticklabels(list(row_pct.columns), rotation=90, fontsize=8)
                            ax.set_yticks(range(row_pct.shape[0])); ax.set_yticklabels(list(row_pct.index), fontsize=8)
                            fig.colorbar(im, ax=ax, shrink=0.85)
                            ax.set_title(f"{top_feat} × {target_col} (satır-normalize, top-{row_pct.shape[0]})")
                            fig.tight_layout(); st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Kategorik-kategorik ilişki analizi için uygun veri bulunamadı.")

                st.subheader("Sınıf Dağılımı")
                if not tgt_series.empty:
                    tmp_df = pd.DataFrame({target_col: tgt_series})
                    fig = VizUtils.top_categories_bar(
                        tmp_df, target_col, top=int(top_cats), height=320,
                        title=f"{target_col} · Top-{top_cats} classes", dark=_IS_DARK
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                prof_c = DataUtils.target_profile_categorical(tgt_series)
                st.dataframe(prof_c["value_counts"], use_container_width=True,
                             height=min(360, 24 * min(top_cats, vc.shape[0]) + 80))
                st.caption(
                    f"Sınıf sayısı: **{prof_c['k_classes']}**, Majority oranı: **{prof_c['majority_ratio']:.2f}**, Entropy: **{prof_c['entropy_bits']} bit**")

                st.subheader("Sayısal Özelliklerin Ayrıştırma Gücü (ANOVA-benzeri)")
                anova_tbl = DataUtils.anova_like_scores(df, target_col, max_features=max_feats)
                if not anova_tbl.empty:
                    st.dataframe(anova_tbl.head(max_feats), use_container_width=True, height=min(360, 24 * max_feats))
                    topF = anova_tbl["feature"].head(min(12, max_feats)).tolist()
                    if topF:
                        g = df[target_col].astype("string").fillna("NA")
                        mat = df[topF].join(g.rename("__g")).groupby("__g").mean(numeric_only=True)
                        if not mat.empty:
                            st.caption("Sınıf bazında ortalama ısı haritası (top-N sayısal özellik)")
                            fig, ax = plt.subplots(
                                figsize=(min(10.0, 0.6 * len(topF) + 2), min(6.0, 0.4 * mat.shape[0] + 2)))
                            im = ax.imshow(mat.values, interpolation="nearest", aspect="auto")
                            ax.set_xticks(range(len(topF))); ax.set_xticklabels(topF, rotation=90, fontsize=8)
                            ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(list(mat.index), fontsize=8)
                            fig.colorbar(im, ax=ax, shrink=0.85)
                            fig.tight_layout(); st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Sayısal özellikler için ANOVA-benzeri skor hesaplanamadı.")
