# pages/DataOverview.py
from __future__ import annotations
from typing import Optional
import streamlit as st
import numpy as np
import pandas as pd
import polars as pl

from utils.DataUtils import DataUtils
from utils.VizUtils import VizUtils

try:
    _IS_DARK = (st.get_option("theme.base") == "dark")
except Exception:
    _IS_DARK = False

@st.cache_data(show_spinner=False)
def cache_profile(_df: pl.DataFrame, dataset_name: str):
    """
    Veri setinin profilini önbelleğe alır.
    DataUtils.profile fonksiyonunu Polars DataFrame ile çağırır.
    """
    return DataUtils.profile(_df)


class DataOverview:
    SESSION_KEY_DF = "__do_df"
    SESSION_KEY_NAME = "__do_name"
    SESSION_KEY_DATASETS = "__do_datasets"              # çoklu dosyalar için dict: {name: df}
    SESSION_KEY_DATASETS_META = "__do_datasets_meta"

    def _load_file(self, up) -> Optional[pl.DataFrame]:
        """
        Dosyayı okur, temizler ve tip çıkarımı uygular.
        Artık Polars DataFrame döner.
        """
        if not up:
            return None

        # Veri okuma (DataUtils artık Polars döndürüyor)
        df = DataUtils.read_any(up.name, up.getvalue())
        df = DataUtils.sanitize_df(df)

        # id benzeri kolonları koruma listesine al
        protected = [c for c in df.columns if str(c).strip().lower() in {
            "id", "key", "user_id", "customer_id", "kod", "code"
        }]

        # Tip çıkarımı (Polars destekli)
        df = DataUtils.infer_dtypes(
            df,
            datetime_guess=True,
            protected_cols=protected,
            protect_id_like_names=True
        )

        # Boyut kontrolü
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
            st.session_state[self.SESSION_KEY_DATASETS] = {}  # {filename: pl.DataFrame}
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

        # === 🔧 Veri Tipi Dönüştürme ve Tarih İşleme (Gelişmiş Kart) ===
        with st.expander("🔧 Veri Tipi Dönüştürme ve Tarih İşleme", expanded=False):

            with st.container():
                col1, col2, col3 = st.columns([2, 2, 2])

                with col1:
                    # 1. Etiketi manuel olarak ekle
                    st.markdown("🧩 **Dönüştürülecek Kolon**")
                    selected_col = st.selectbox(
                        "Dönüştürülecek Kolon",  # Bu, ekran okuyucular için gereklidir
                        df.columns,
                        key="conv_col",
                        label_visibility="collapsed"  # Dahili etiketi gizle
                    )

                with col2:
                    # 2. Bu kod zaten doğru yapıda (Etiket + İçerik)
                    st.markdown("🔍 **Seçilen Kolonun Mevcut Tipi**")
                    current_dtype = df.schema[selected_col]
                    st.markdown(
                        f"<div style='padding:8px;border-radius:6px;background-color:#0E1117;border:1px solid #444;color:#8ab4f8;'>"
                        f"{current_dtype}</div>",
                        unsafe_allow_html=True,
                    )

                with col3:
                    # 3. Etiketi manuel olarak ekle
                    st.markdown("🎯 **Tür Dönüştür**")
                    dtype_options = ["string", "int", "float", "boolean", "date", "datetime"]
                    selected_type = st.selectbox(
                        f"Tür Dönüştür",  # Bu, ekran okuyucular için gereklidir
                        dtype_options,
                        key="conv_type",
                        label_visibility="collapsed"  # Dahili etiketi gizle
                    )

            # --- Tarih ayarları (sadece tarih tipleri için) ---
            extract_parts = False  # Varsayılan değer

            if selected_type in ("date", "datetime"):
                st.markdown("📅 Tarih Ayarları (İsteğe bağlı)")
                extract_parts = st.checkbox("Yıl / Ay / Gün Kolonları Oluştur", key="extract_date_parts")

            # Bu sayede her zaman görünür olacak.
            if st.button("🚀 Dönüştürmeyi Uygula", key="apply_type"):
                try:
                    # === Gerekli değişkenleri ve tipleri hazırla ===
                    dtype_map = {
                        "string": pl.Utf8, "int": pl.Int64, "float": pl.Float64,
                        "boolean": pl.Boolean, "date": pl.Date, "datetime": pl.Datetime
                    }
                    target_dtype_obj = dtype_map.get(selected_type)  # Hedef Polars tipi
                    current_dtype_obj = df.schema[selected_col]  # Mevcut Polars tipi

                    # Bayraklar: Hangi işlemlerin yapıldığını takip et
                    did_convert = False
                    did_extract = False

                    # === 1. TÜR DÖNÜŞÜMÜ ===
                    # Hedef tip, mevcut tipten farklıysa
                    conversion_is_needed = target_dtype_obj and current_dtype_obj != target_dtype_obj

                    if conversion_is_needed:
                        df = DataUtils.convert_column_type(df, selected_col, selected_type)
                        st.session_state[DataOverview.SESSION_KEY_DATASETS][name] = df
                        st.success(
                            f"✅ {selected_col} sütunu '{current_dtype_obj}' ➜ '{df.schema[selected_col]}' tipine dönüştürüldü.")
                        did_convert = True

                    # === 2. TARİH PARÇALAMA GEREKLİ Mİ? ===
                    # Bu blok 'extract_parts' bayrağına bağlı olduğu için
                    # zaten sadece tarih tiplerinde ve checkbox seçiliyse çalışır
                    if selected_type in ("date", "datetime") and extract_parts:
                        # (Dönüşüm yeni yapılmış olabilir, df'in son halini kontrol et)
                        current_dtype_after_conv = df.schema[selected_col]

                        if current_dtype_after_conv not in (pl.Datetime, pl.Date):
                            # Hata değil uyarı: Önce dönüştürmesi gerekir
                            st.warning(
                                f"'{selected_col}' sütunu {current_dtype_after_conv} tipinde. "
                                f"Tarih parçalama için önce 'date' veya 'datetime' tipine dönüştürülmeli."
                            )
                        else:
                            # Sadece bu işlem istendiyse (dönüşüm yapılmadıysa)
                            if not did_convert:
                                st.info(
                                    f"'{selected_col}' zaten {current_dtype_after_conv} tipinde. Sadece tarih parçalama yapılıyor...")

                            df = DataUtils.extract_date_parts(df, selected_col)
                            st.session_state[DataOverview.SESSION_KEY_DATASETS][name] = df
                            st.success("✅ Tarih parçaları oluşturuldu (year, month, day).")
                            did_extract = True

                            # Parçalanan kısımları göster
                            st.dataframe(
                                df.select([
                                    selected_col,
                                    f"{selected_col}_year",
                                    f"{selected_col}_month",
                                    f"{selected_col}_day"
                                ]).head(5),
                                use_container_width=True,
                            )

                    # === 3. İŞLEM YAPILMADIYSA BİLGİ VER ===
                    if not did_convert and not did_extract:
                        st.info("Seçilen kolon zaten istenen tipte ve/veya bir işlem (parçalama) seçilmedi.")

                    # === 4. PROFİLİ YENİLE (Değişiklik varsa) ===
                    if did_convert or did_extract:
                        st.session_state["__profile_dirty__"] = True

                        # Eğer sadece dönüşüm yapıldıysa (tarih tablosu yukarıda gösterilmediyse)
                        # ana kolonun son halini göster
                        if did_convert and not did_extract:
                            st.dataframe(df[[selected_col]].head(5), use_container_width=True)

                except Exception as e:
                    st.error(f"Dönüştürme hatası: {e}")

        # ---- Profil kartları
        if st.session_state.get("__profile_dirty__"):
            st.cache_data.clear()
            st.session_state.pop("__profile_dirty__")

        prof = cache_profile(df, name)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Satır", f"{prof.n_rows:,}")
        m2.metric("Sütun", f"{prof.n_cols:,}")
        m3.metric("Dosya Boyut", "—" if not size_bytes else f"{DataUtils._bytes_to_mb(size_bytes)}")
        m4.metric("RAM (MB)", f"{prof.mem_usage_mb}")
        m5.metric("Eksik Oranı", f"{prof.missing_ratio * 100:.2f}%")

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

        # ====================== Variables (tek değişken odaklı kart) ================
        st.markdown("## Değişkenler ")

        col_sel, col_blank = st.columns([2, 1])
        with col_sel:
            var_col = st.selectbox("Kolon Seçiniz", options=list(df.columns), key="__vars_sel")

        with st.container(border=True):
            if var_col:
                vp = DataUtils.variable_profile(df, var_col, bins=40)

                st.markdown(f"### <span style='color:#2B6CB0'>{var_col}</span>", unsafe_allow_html=True)

                s = df[var_col]
                dtype = s.dtype

                # --- Polars tipi etiketi ---
                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                    dtype_label = "Integer (I)"
                elif dtype in (pl.Float32, pl.Float64):
                    dtype_label = "Real number (R)"
                elif dtype == pl.Datetime:
                    dtype_label = "Datetime (D)"
                else:
                    dtype_label = "Categorical (C)"

                st.caption(f"{dtype_label}")

                # Üst metrikler
                n = vp["n"]
                non_null = int(n - vp["missing"])

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
                        "Metric": list(left_rows.keys()),
                        "Value": list(left_rows.values())
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

                    # Sayısal
                    if s_this.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                        if vp.get("hist") is not None and vp.get("hist_edges") is not None:
                            ch = VizUtils.histogram(df[var_col], bins=60, title="Histogram", height=260, dark=_IS_DARK)
                            st.altair_chart(ch, use_container_width=True)

                    # Kategorik
                    elif s_this.dtype == pl.Utf8:
                        ch = VizUtils.top_categories(df, var_col, top=6, title="Top Categories", height=260,
                                                     dark=_IS_DARK)
                        st.altair_chart(ch, use_container_width=True)

                    # Tarih
                    elif s_this.dtype == pl.Datetime:
                        ch = VizUtils.time_count(df[var_col], freq="D", title="Daily counts", height=260, dark=_IS_DARK)
                        st.altair_chart(ch, use_container_width=True)

                    else:
                        st.caption("Bu değişken için grafik uygun değil.")

                # Alt sekmeler: Statistics / Histogram / Common
                tab1, tab2, tab3 = st.tabs(["Statistics", "Histogram", "Common values"])

                # --- Tab 1: Quantile & Descriptive ---
                with tab1:
                    s = df[var_col]
                    if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                        qdf = DataUtils.variable_quantile_table(s)
                        ddf = DataUtils.variable_descriptive_table(s)

                        t1, t2 = st.columns(2)
                        with t1:
                            st.subheader("Quantile statistics")
                            st.dataframe(qdf.to_pandas(), use_container_width=True, height=360, hide_index=True)
                        with t2:
                            st.subheader("Descriptive statistics")
                            st.dataframe(ddf.to_pandas(), use_container_width=True, height=360, hide_index=True)
                    else:
                        st.info("Sayısal olmayan sütun için bu sekme sınırlıdır.")

                # --- Tab 2: Histogram ---
                with tab2:
                    bins = st.slider("Histogram bins", 5, 120, 40, step=5, key="__vars_bins")
                    s = df[var_col]

                    if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                        ch = VizUtils.histogram(s, bins=int(bins), title=f"{var_col} · Histogram", height=320,
                                                dark=_IS_DARK)
                        st.altair_chart(ch, use_container_width=True)
                    elif s.dtype == pl.Datetime:
                        ch = VizUtils.time_count(s, freq="D", title=f"{var_col} · Daily counts", height=320,
                                                 dark=_IS_DARK)
                        st.altair_chart(ch, use_container_width=True)
                    else:
                        ch = VizUtils.top_categories(df, var_col, top=30, title="Top values", height=320, dark=_IS_DARK)
                        st.altair_chart(ch, use_container_width=True)

                # --- Tab 3: Common Values ---
                with tab3:
                    top = st.slider("Top-N", 5, 50, 20, step=5, key="__vars_top")
                    cv = DataUtils.variable_common_values(df, var_col, top=top)
                    st.dataframe(
                        cv.to_pandas(),
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

        # ----------------------------------------------------
        # 🔗 Korelasyon Analiz Kartı
        # ----------------------------------------------------
        with st.container(border=True):
            st.markdown("## 🔗 Korelasyon Analizi")

            # Sekmeler (2 farklı görsel)
            tab1, tab2= st.tabs([
                "Correlation Matrix","Correlation Strength"
            ])

            corr_df = DataUtils.correlation_matrix(df)
            with tab1:
                st.altair_chart(VizUtils.correlation_heatmap(corr_df, dark=_IS_DARK), use_container_width=True)
            with tab2:
                valid_targets = [c for c in corr_df.columns if c != "column"]
                target_col = st.selectbox("🎯 Hedef Değişken Seçin", valid_targets)
                st.altair_chart(
                    VizUtils.correlation_strength_bar(corr_df, target_col, dark=_IS_DARK),
                    use_container_width=True
                )

        # ----------------------------------------------------
        # 🧩 Eksik Değer Analizi Kartı
        # ----------------------------------------------------
        with st.container(border=True):
            st.markdown("## 🧩 Eksik Değer Analizi")

            # 1️⃣ Eksik özet hesapla
            missing_df = DataUtils.missing_value_summary(df)
            total_missing_cols = (missing_df["missing_count"] > 0).sum()
            total_missing_vals = int(missing_df["missing_count"].sum())
            avg_missing_pct = float(missing_df["missing_pct"].mean())

            # 2️⃣ Özet kartlar
            m1, m2, m3 = st.columns(3)
            m1.metric("Eksik Değerli Kolon", f"{total_missing_cols:,}")
            m2.metric("Toplam Eksik Hücre", f"{total_missing_vals:,}")
            m3.metric("Ortalama Eksik (%)", f"{avg_missing_pct:.2f}%")

            # 3️⃣ Kolon bazında özet tablo
            st.markdown("#### 📋 Kolon Bazında Eksik Değer Özeti")
            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True,
                height=min(400, 30 * (missing_df.height + 1)),
                column_config={
                    "column": "Kolon Adı",
                    "missing_count": st.column_config.NumberColumn("Eksik Sayısı", format="%.0f"),
                    "missing_pct": st.column_config.ProgressColumn("Eksik (%)", format="%.2f%%", min_value=0,
                                                                   max_value=100),
                },
            )

            # 4️⃣ Sekmeler (5 farklı görsel)
            tab1, tab2, tab3, tab4, tab5= st.tabs([
                "Bar Plot", "Matrix", "Heatmap", "Dendrogram", "Correlation Plot",
            ])

            with tab1:
                st.altair_chart(VizUtils.missing_bar(missing_df, dark=_IS_DARK), use_container_width=True)
            with tab2:
                st.pyplot(VizUtils.missing_matrix(df), use_container_width=True)
            with tab3:
                st.pyplot(VizUtils.missing_heatmap(df), use_container_width=True)
            with tab4:
                st.pyplot(VizUtils.missing_dendrogram(df), use_container_width=True)
            with tab5:
                st.pyplot(VizUtils.missing_corr_plot(df), use_container_width=True)

        # ----------------------------------------------------
        # 🧩 Eksik Değer Doldurma Kartı
        # ----------------------------------------------------
        with st.container(border=True):
            st.markdown("## 🧩 Eksik Değerleri Doldurma")

            missing_cols = DataUtils.get_missing_columns(df)

            if not missing_cols:
                st.success("✅ Veri setinde eksik değer bulunmuyor.")
                st.stop()

            # 1️⃣ Kolon Seçimi + Bilgiler
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                st.markdown("🎯 **Doldurulacak Kolon**")
                fill_col = st.selectbox(
                    "Doldurulacak Kolon",
                    missing_cols,
                    key="fill_col",
                    label_visibility="collapsed"
                )

            current_dtype = df.schema[fill_col]
            missing_count = df[fill_col].null_count()

            with c2:
                st.markdown("🔍 **Veri Tipi**")
                st.markdown(
                    f"<div style='padding:8px;border-radius:6px;background-color:#0E1117;border:1px solid #444;"
                    f"color:#8ab4f8;text-align:left;'>{current_dtype}</div>",
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown("📉 **Eksik Değer Sayısı**")
                st.markdown(
                    f"<div style='padding:8px;border-radius:6px;background-color:#0E1117;border:1px solid #444;"
                    f"color:#f88a8a;text-align:left;'>{missing_count:,}</div>",
                    unsafe_allow_html=True,
                )

            # 2️⃣ Geçerli Yöntemleri Al
            all_methods = DataUtils.get_fill_methods()
            valid_methods = DataUtils.suggest_fill_methods(current_dtype)

            c4, c5 = st.columns([2, 2])
            with c4:
                st.markdown("🛠️ **Doldurma Yöntemi**")
                selected_method = st.selectbox(
                    "Doldurma Yöntemi",
                    options=[m for m in all_methods.keys() if m in valid_methods],
                    format_func=lambda k: all_methods[k],
                    key="fill_method",
                    label_visibility="collapsed"
                )



            # 3️⃣ Koşullu Değer Girişi
            fill_value = None
            preview_methods = ["mean", "median", "mode", "min", "max", "zero"]
            with c5:
                if selected_method in ("specific", "custom"):
                    st.markdown("📝 **Doldurulacak Değer**")
                    if current_dtype in pl.NUMERIC_DTYPES:
                        fill_value = st.number_input("Değer", value=0, key="fill_val_num", label_visibility="collapsed")
                    elif current_dtype == pl.Boolean:
                        fill_value = st.selectbox("Değer", [True, False], key="fill_val_bool",
                                                  label_visibility="collapsed")
                    elif current_dtype == pl.Date:
                        fill_value = st.date_input("Değer", key="fill_val_date", label_visibility="collapsed")
                    elif current_dtype == pl.Datetime:
                        fill_value = st.datetime_input("Değer", key="fill_val_datetime", label_visibility="collapsed")
                    else:
                        fill_value = st.text_input("Değer", value="NA", key="fill_val_str",
                                                   label_visibility="collapsed")
                elif selected_method in preview_methods:
                    try:
                        preview_value = DataUtils.compute_fill_value(
                            df, fill_col, selected_method
                        )

                        if preview_value is not None:
                            # Değeri formatla
                            if isinstance(preview_value, float):
                                preview_val_str = f"{preview_value:,.4f}"
                            elif isinstance(preview_value, int):
                                preview_val_str = f"{preview_value:,}"
                            else:
                                preview_val_str = str(preview_value)

                            st.markdown("**Hesaplanan Değer**")
                            st.markdown(
                                f"<div style='padding:8px; margin-top: 1px; border-radius:6px; background-color:#0E1117;"
                                f"border:1px solid #444; color:#8ab4f8; text-align:left; font-size: 0.9em;'>"
                                f"<strong>{preview_val_str}</strong></div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.caption("Değer hesaplanamadı (örn: kolon boş).")
                    except Exception:
                        st.caption("Değer hesaplanamadı.")

            # 4️⃣ Doldurma Uygulama
            if st.button("🚀 Doldurmayı Uygula", key="apply_fill", disabled=(missing_count == 0)):
                try:
                    before = df[fill_col].null_count()

                    df = DataUtils.fill_missing(df, fill_col, selected_method, fill_value)
                    after = df[fill_col].null_count()

                    st.session_state[DataOverview.SESSION_KEY_DATASETS][name] = df
                    st.session_state["__profile_dirty__"] = True

                    st.success(
                        f"✅ '{fill_col}' kolonundaki {before:,} eksik değer "
                        f"'{all_methods[selected_method]}' yöntemiyle dolduruldu. "
                        f"Kalan eksik: {after:,}"
                    )

                    st.dataframe(df[[fill_col]].head(10), use_container_width=True)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Doldurma hatası: {e}")

                # ----------------------------------------------------
                # 🎯 Hedef Odaklı Analiz Kartı (Target-Aware)
                # ----------------------------------------------------
                with st.container(border=True):
                    st.markdown("## 🎯 Hedef Odaklı Analiz (Bivariate)")
                    st.caption(
                        "Bir hedef değişken seçin; sistem otomatik olarak görev türünü "
                        "(binary/multiclass/regression) algılasın ve "
                        "diğer tüm özelliklerle istatistiksel ilişkisini hesaplasın."
                    )

                    # df = aktif dataframe (yukarıda zaten tanımlı olmalı)
                    if df is None or df.is_empty():
                        st.info("Analiz için lütfen önce bir veri seti yükleyin.")
                        st.stop()

                    # Profil verisinden uygun kolonları al
                    prof = cache_profile(df, name)
                    potential_targets = prof.categorical_cols + prof.numeric_cols + prof.datetime_cols

                    if not potential_targets:
                        st.warning("Veri setinde analiz edilecek uygun (sayısal, kategorik) kolon bulunamadı.")
                        st.stop()

                    # 1. Hedef Seçimi
                    sel_target = st.selectbox(
                        "🎯 Hedef (Target) Değişkeni Seçin",
                        options=potential_targets,
                        index=0,
                        key="__target_aware_select"
                    )

                    # 2. Analizi Çalıştır Butonu
                    if st.button("🚀 Hedef Odaklı Analizi Çalıştır", type="primary", use_container_width=True):

                        # 3. Analizi çalıştır (veya önbellekten al)
                        results_df, task_type = cache_target_analysis(df, sel_target)

                        if results_df is None:
                            st.error("Analiz çalıştırılamadı. Hedef değişken geçerli değil.")
                        else:
                            st.metric("Tespit Edilen Görev Türü (Task Type)", f"**{task_type.upper()}**")

                            # 4. Sonuçları Göster
                            st.markdown("#### İstatistiksel Analiz Raporu")
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "feature": "Özellik",
                                    "feature_type": "Tip",
                                    "test": "Test",
                                    "effect": st.column_config.NumberColumn("Etki Büyüklüğü", format="%.3f"),
                                    "effect_abs": st.column_config.NumberColumn("Etki (Mutlak)", format="%.3f",
                                                                                help="Sıralama için kullanılır (en güçlü ilişki)"),
                                    "pvalue": st.column_config.NumberColumn("p-value", format="%.4f"),
                                    "missing_pct": st.column_config.ProgressColumn("Eksik (%)", format="%.2f%%",
                                                                                   min_value=0, max_value=100),
                                    "note": "Not",
                                    "viz_hint": "Görsel Tipi"
                                }
                            )

                            # 5. Görev Türüne Özel Görselleştirme
                            st.markdown("---")
                            st.markdown("#### Öne Çıkan Görselleştirmeler")

                            try:
                                # ----- Binary Görev Görseli (Sizin demo koddaki gibi) -----
                                if task_type == "binary":
                                    st.markdown("##### Kategorik Değişkenler vs. Hedef Oranı")
                                    # Analiz raporundan kategorik kolonları al
                                    cat_cols = results_df[
                                        (results_df["feature_type"] == "categorical") &
                                        (results_df["effect_abs"].notna())
                                        ]["feature"].tolist()

                                    if cat_cols:
                                        sel_cat_feat = st.selectbox("Görselleştirmek için bir özellik seçin", cat_cols)

                                        # Veriyi hazırla (Polars -> Pandas)
                                        x = df[sel_cat_feat].cast(pl.Utf8).fill_null("NA")
                                        y = df[sel_target]
                                        y01 = _to_binary01(y.to_pandas())
                                        pdf = pd.DataFrame({sel_cat_feat: x.to_pandas(), sel_target: y01})

                                        grp = (pdf.groupby(sel_cat_feat)[sel_target]
                                               .agg(["count", "mean"])
                                               .rename(columns={"count": "n", "mean": "target_rate"})
                                               .reset_index())

                                        # Altair Grafiği
                                        ch = (alt.Chart(grp).mark_bar()
                                              .encode(
                                            x=alt.X("target_rate:Q", title=f"'{sel_target}=1' Oranı",
                                                    scale=alt.Scale(domain=[0, 1])),
                                            y=alt.Y(f"{sel_cat_feat}:N", sort="-x", title=sel_cat_feat),
                                            tooltip=[sel_cat_feat, "n:Q", alt.Tooltip("target_rate:Q", format=".2%")]
                                        ).properties(height=max(200, min(500, grp.shape[0] * 25))))  # Dinamik yükseklik

                                        st.altair_chart(ch, use_container_width=True)

                                    else:
                                        st.info("Bu görev için uygun kategorik özellik bulunamadı.")

                                # ----- Regression Görev Görseli -----
                                elif task_type == "regression":
                                    st.markdown("##### Sayısal Değişkenler vs. Hedef")
                                    # Rapordaki en güçlü ilişkili sayısal kolonu al
                                    num_cols = results_df[
                                        (results_df["feature_type"] == "numeric") &
                                        (results_df["effect_abs"].notna())
                                        ]["feature"].tolist()

                                    if num_cols:
                                        sel_num_feat = st.selectbox("Görselleştirmek için bir özellik seçin", num_cols)

                                        # Polars'tan Pandas'a
                                        pdf_sample = df.select([sel_num_feat, sel_target]).sample(
                                            n=min(5000, df.height)).to_pandas()

                                        # Altair Scatter Plot
                                        ch = (alt.Chart(pdf_sample).mark_circle(opacity=0.5)
                                              .encode(
                                            x=alt.X(sel_num_feat, title=sel_num_feat),
                                            y=alt.Y(sel_target, title=sel_target),
                                            tooltip=[sel_num_feat, sel_target]
                                        ).properties(title=f"{sel_target} vs {sel_num_feat} (5k örneklem)")
                                              .interactive())

                                        st.altair_chart(
                                            ch + ch.transform_regression(sel_num_feat, sel_target).mark_line(
                                                color="red"), use_container_width=True)
                                    else:
                                        st.info("Bu görev için uygun sayısal özellik bulunamadı.")

                                else:
                                    st.info(
                                        f"'{task_type}' görev türü için otomatik görselleştirme henüz tanımlanmadı.")

                            except Exception as e:
                                st.error(f"Görselleştirme hatası: {e}")