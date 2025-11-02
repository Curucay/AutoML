# pages/DataOverview.py
from __future__ import annotations
from dataclasses import asdict
from typing import Optional, List
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.DataUtils import DataUtils

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
        df = DataUtils.infer_dtypes(df)
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

        # === Veri Birleştirme (beta) ===
        with st.expander("🧩 Veri Birleştirme", expanded=False):
            op = st.radio("Birleştirme türü", ["Satır birleştirme (concat)", "Anahtarla birleştirme (merge)"],
                          horizontal=False)

            # Kısa referanslar
            datasets = st.session_state[DataOverview.SESSION_KEY_DATASETS]
            meta = st.session_state[DataOverview.SESSION_KEY_DATASETS_META]
            ds_names = list(datasets.keys())

            # ---------------- Concat ----------------
            if op.startswith("Satır"):
                pick = st.multiselect("Birleştirilecek veri setleri (en az 2)", options=ds_names)
                col_mode = st.radio("Sütun hizalama", ["union (tüm sütunlar)", "intersection (ortak sütunlar)"],
                                    horizontal=True, index=0)
                add_src = st.checkbox("Kaynak adı etiketi ekle (__source__)", value=False)
                target_name = st.text_input("Yeni veri seti adı", value="merged_concat")

                do_concat = st.button("Birleştir ve Kaydet", type="primary", use_container_width=True)
                if do_concat:
                    if not pick or len(pick) < 2:
                        st.error("En az iki veri seti seçin.")
                    else:
                        dfs = [datasets[nm] for nm in pick]
                        mode = "union" if col_mode.startswith("union") else "intersection"
                        out = DataUtils.concat_safe(dfs, column_mode=mode, add_source_label=add_src, source_names=pick)
                        # kayıt
                        datasets[target_name] = out
                        meta[target_name] = {"size_bytes": 0}  # dosya boyutu bilinmiyor → 0 (UI'da '—' göstereceğiz)
                        st.cache_data.clear()
                        st.session_state[DataOverview.SESSION_KEY_NAME] = target_name
                        st.success(f"✔ {target_name} oluşturuldu · {out.shape[0]:,} satır × {out.shape[1]:,} sütun")
                        st.rerun()

            # ---------------- Merge ----------------
            else:
                c1, c2 = st.columns(2)
                with c1:
                    left_ds = st.selectbox("Left (sol) veri seti", options=ds_names, key="__merge_left")
                with c2:
                    right_ds = st.selectbox("Right (sağ) veri seti",
                                            options=[n for n in ds_names if n != st.session_state.get("__merge_left")],
                                            key="__merge_right")
                if left_ds and right_ds:
                    L, R = datasets[left_ds], datasets[right_ds]
                    # anahtar önerileri
                    suggest = DataUtils.suggest_join_keys(L, R, max_candidates=5)
                    st.caption(f"Anahtar önerileri: {', '.join(suggest) if suggest else '—'}")
                    # kullanıcı seçimi
                    commons = [c for c in L.columns if c in R.columns]
                    on_cols = st.multiselect("Join anahtar(lar)ı", options=commons, default=suggest)
                    how = st.selectbox("Join türü", options=["inner", "left", "right", "outer"],
                                       index=1)  # default left
                    sfx1 = st.text_input("Sol sonek", value="_x")
                    sfx2 = st.text_input("Sağ sonek", value="_y")
                    target_name = st.text_input("Yeni veri seti adı", value="merged_join")

                    do_merge = st.button("Birleştir ve Kaydet", type="primary", use_container_width=True)
                    if do_merge:
                        try:
                            out = DataUtils.merge_safe(L, R, on=on_cols, how=how, suffixes=(sfx1, sfx2))
                        except Exception as e:
                            st.error(f"Birleştirme hatası: {e}")
                        else:
                            datasets[target_name] = out
                            meta[target_name] = {"size_bytes": 0}  # dosya boyutu bilinmiyor
                            st.cache_data.clear()
                            st.session_state[DataOverview.SESSION_KEY_NAME] = target_name
                            st.success(f"✔ {target_name} oluşturuldu · {out.shape[0]:,} satır × {out.shape[1]:,} sütun")
                            st.rerun()

        if clear:
            datasets.clear()
            st.session_state.pop(self.SESSION_KEY_NAME, None)
            st.cache_data.clear()
            st.rerun()

        if ups:
            for up in ups:
                if up is None:
                    continue
                # Aynı isimle daha önce eklenmişse atla (istersen timestamp/numara ekleyebilirsin)
                if up.name in datasets:
                    continue
                    # Boyutu verimli şekilde al
                size_bytes = getattr(up, "size", None)
                if size_bytes is None:
                    size_bytes = up.getbuffer().nbytes
                df_new = self._load_file(up)
                if df_new is not None:
                    datasets[up.name] = df_new
                    meta[up.name] = {"size_bytes": int(size_bytes)}

        if not datasets:
            st.info("Bir veya daha fazla veri dosyası yüklediğinizde burada listelenecek.")
            return

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
        m1, m2, m3, m4, m5= st.columns(5)
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

        # ---- Hızlı istatistik
        st.subheader("Hızlı İstatistik")
        sel_cols = st.multiselect(
            "İstatistik alınacak sütunlar (boş → numerikler)",
            options=list(df.columns),
            default=prof.numeric_cols[: min(8, len(prof.numeric_cols))]
        )
        stats_df = DataUtils.quick_stats(df, sel_cols if sel_cols else None)
        st.dataframe(stats_df, use_container_width=True, height=380)

        # === Görselleştirmeler ===
        st.markdown("## Görselleştirmeler")

        # Kullanıcı kontrolleri (grafik boyutu ve içerik sınırları)
        with st.expander("Grafik Ayarları", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                bins = st.slider("Bins", 5, 200, 40, step=5)
            with c2:
                density = st.checkbox("Histogram normalize", value=False)
            with c3:
                max_plot_cols = st.slider("Max grafikte gösterilecek sütun (num/kat)", 3, 24, 10)
            with c4:
                corr_max_cols = st.slider("Max kolon (korelasyon ısı haritası)", 5, 50, 20)

        # Histogramlar (küçük ebat)
        DataUtils._hist_numeric(
            df,
            bins=bins,
            density=density,
            max_cols=max_plot_cols,
            fig_size=(6.0, 3.5)  # küçük ve okunaklı
        )

        # Kategorik dağılımlar (Top-N bar grafikleri)
        DataUtils._bar_categorical(
            df,
            top=20,  # her sütunda top-20 değer
            max_cols=max_plot_cols,
            fig_size=(6.0, 3.5)  # küçük ebat
        )

        # Korelasyon ısı haritası
        method = st.radio("Korelasyon yöntemi", ["pearson", "spearman", "kendall"], horizontal=True, index=0)
        DataUtils._corr_heatmap(
            df,
            method=method,
            max_cols=corr_max_cols  # çok kolonlu veri setlerinde matrisi küçük tutar
        )

        # ---- Kategorik özet
        st.subheader("Kategorik Dağılım (Top 20)")
        cat = st.selectbox("Sütun seç", options=prof.categorical_cols or ["(kategorik yok)"])
        if prof.categorical_cols:
            vc = DataUtils.value_counts_frame(df, cat, top=20)
            st.dataframe(vc, use_container_width=True, height=360)

        # === 🎯 Hedef Değişken Analizi ===
        st.markdown("## 🎯 Hedef Değişken")

        # 1) Hedef seçimi
        target_col = st.selectbox("Hedef sütunu seç", options=list(df.columns))

        if target_col:
            tgt_series = df[target_col]
            tgt_type = DataUtils.detect_target_type(tgt_series)
            st.caption(f"Hedef tipi: **{tgt_type}**")

            # Genel boşluk oranı
            missing_rate = float(tgt_series.isna().mean() * 100.0)
            st.caption(f"Eksik oranı: **{missing_rate:.2f}%**")

            # Kontrol paneli (küçük grafik ayarları)
            with st.expander("Hedef Grafik Ayarları", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    bins_t = st.slider("Target bins", 5, 200, 40, step=5)
                with c2:
                    top_cats = st.slider("Kategorik top-N", 3, 50, 20)
                with c3:
                    max_feats = st.slider("Özellik sınırı (tablolar)", 10, 100, 30)
                with c4:
                    heat_top = st.slider("Isı haritası top-N kategori", 5, 30, 12)  # <<< yeni

            # 2) Hedef TÜRÜNE GÖRE analiz
            if tgt_type == "numeric":
                # === EKLEME 1: Sayısal hedef vs KATEGORİK özellikler ===
                st.subheader("Hedef (sayısal) vs Kategorik Özellikler")

                # 1) Kategorik özelliklerin ayrıştırma gücü (between/within variance oranı)
                cat_scores = DataUtils.group_diff_scores_for_numeric_target(
                    df, target_col, max_features=max_feats
                )
                if not cat_scores.empty:
                    st.dataframe(
                        cat_scores.head(max_feats),
                        use_container_width=True,
                        height=min(360, 24 * max_feats)
                    )

                    # 2) En güçlü tek bir kategorik özelliği seçip hedef ortalaması bar grafiği
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
                            ax.set_title(
                                f"{best_cat} gruplarına göre {target_col} ortalaması (top-{min(top_cats, agg.shape[0])})")
                            ax.set_xlabel(best_cat)
                            ax.set_ylabel(f"{target_col} ort.")
                            ax.tick_params(axis="x", labelrotation=45)
                            ax.grid(axis="y", alpha=0.2)
                            st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Kategorik özellikler için anlamlı skor hesaplanamadı veya veri uygun değil.")

                # --- Dağılım: Histogram (küçük)
                st.subheader("Hedef Dağılımı")
                s = pd.to_numeric(tgt_series, errors="coerce").dropna()
                if not s.empty:
                    fig, ax = plt.subplots(figsize=(6.0, 3.5))
                    ax.hist(s, bins=bins_t, density=False)
                    ax.set_title(f"Histogram · {target_col}")
                    ax.set_xlabel(target_col)
                    ax.set_ylabel("Frekans")
                    ax.grid(alpha=0.2)
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Hedef sayısal değerler üretmedi.")

                # --- Temel istatistikler
                prof_t = DataUtils.target_profile_numeric(tgt_series)
                st.write(pd.DataFrame([prof_t]))

                # --- Özellik korelasyonları (numerik)
                st.subheader("Özellik Korelasyonları (Numerik → Hedef)")
                corr_tbl = DataUtils.corr_with_target_numeric(df, target_col, method="pearson")
                if not corr_tbl.empty:
                    st.dataframe(corr_tbl.head(max_feats), use_container_width=True, height=min(360, 24 * max_feats))
                    # küçük scatter'lar (en güçlü ilk 3)
                    top3 = corr_tbl["feature"].head(3).tolist()
                    if top3:
                        st.caption("En güçlü 3 özellik için küçük scatter grafikleri")
                        for feat in top3:
                            xy = df[[feat, target_col]].dropna()
                            if xy.empty:
                                continue
                            fig, ax = plt.subplots(figsize=(6.0, 3.5))
                            ax.scatter(xy[feat], xy[target_col], s=6, alpha=0.5)
                            ax.set_xlabel(feat);
                            ax.set_ylabel(target_col)
                            ax.set_title(f"{feat} vs {target_col}")
                            ax.grid(alpha=0.2)
                            st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Numerik özellik korelasyonu hesaplanamadı.")

            else:
                # === EKLEME 2: Kategorik hedef vs KATEGORİK özellikler (Cramér's V) ===
                st.subheader("Hedef (kategorik) vs Kategorik Özellikler")

                assoc = DataUtils.cate_cate_assoc(df, target_col, max_features=max_feats)
                if not assoc.empty:
                    st.dataframe(
                        assoc.head(max_feats),
                        use_container_width=True,
                        height=min(360, 24 * max_feats)
                    )

                    # En güçlü kategorik özelliği seçip kontenjans ısı haritası
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
                        # Satırlarda en sık görülen top-N kategori
                        top_rows = tbl.sum(axis=1).sort_values(ascending=False).head(heat_top).index
                        tbl_small = tbl.loc[top_rows]

                        if not tbl_small.empty:
                            row_pct = tbl_small.div(tbl_small.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
                            fig, ax = plt.subplots(
                                figsize=(min(10.0, 0.6 * row_pct.shape[1] + 2),
                                         min(6.0, 0.45 * row_pct.shape[0] + 2))
                            )
                            im = ax.imshow(row_pct.values, interpolation="nearest", aspect="auto")
                            ax.set_xticks(range(row_pct.shape[1]))
                            ax.set_xticklabels(list(row_pct.columns), rotation=90, fontsize=8)
                            ax.set_yticks(range(row_pct.shape[0]))
                            ax.set_yticklabels(list(row_pct.index), fontsize=8)
                            fig.colorbar(im, ax=ax, shrink=0.85)
                            ax.set_title(f"{top_feat} × {target_col} (satır-normalize, top-{row_pct.shape[0]})")
                            fig.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Kategorik-kategorik ilişki analizi için uygun veri bulunamadı.")

                # === CATEGORICAL TARGET ===
                st.subheader("Sınıf Dağılımı")
                vc = tgt_series.astype("string").fillna("NA").value_counts(dropna=False)
                if not vc.empty:
                    fig, ax = plt.subplots(figsize=(6.0, 3.5))
                    vc.head(top_cats).plot(kind="bar", ax=ax)
                    ax.set_title(f"{target_col} · en sık {min(top_cats, vc.shape[0])} sınıf")
                    ax.set_xlabel(target_col);
                    ax.set_ylabel("Frekans")
                    ax.tick_params(axis="x", labelrotation=45)
                    ax.grid(axis="y", alpha=0.2)
                    st.pyplot(fig, clear_figure=True)
                # Profil tablosu
                prof_c = DataUtils.target_profile_categorical(tgt_series)
                st.dataframe(prof_c["value_counts"], use_container_width=True,
                             height=min(360, 24 * min(top_cats, vc.shape[0]) + 80))
                st.caption(
                    f"Sınıf sayısı: **{prof_c['k_classes']}**, Majority oranı: **{prof_c['majority_ratio']:.2f}**, Entropy: **{prof_c['entropy_bits']} bit**")

                # --- Sayısal özelliklerde sınıflar arası fark (ANOVA-benzeri)
                st.subheader("Sayısal Özelliklerin Ayrıştırma Gücü (ANOVA-benzeri)")
                anova_tbl = DataUtils.anova_like_scores(df, target_col, max_features=max_feats)
                if not anova_tbl.empty:
                    st.dataframe(anova_tbl.head(max_feats), use_container_width=True, height=min(360, 24 * max_feats))
                    # küçük ısı haritası: sınıf bazında ortalamalar (top-N özellik)
                    topF = anova_tbl["feature"].head(min(12, max_feats)).tolist()
                    if topF:
                        g = df[target_col].astype("string").fillna("NA")
                        mat = df[topF].join(g.rename("__g")).groupby("__g").mean(numeric_only=True)
                        if not mat.empty:
                            st.caption("Sınıf bazında ortalama ısı haritası (top-N sayısal özellik)")
                            fig, ax = plt.subplots(
                                figsize=(min(10.0, 0.6 * len(topF) + 2), min(6.0, 0.4 * mat.shape[0] + 2)))
                            im = ax.imshow(mat.values, interpolation="nearest", aspect="auto")
                            ax.set_xticks(range(len(topF)))
                            ax.set_xticklabels(topF, rotation=90, fontsize=8)
                            ax.set_yticks(range(mat.shape[0]))
                            ax.set_yticklabels(list(mat.index), fontsize=8)
                            fig.colorbar(im, ax=ax, shrink=0.85)
                            fig.tight_layout()
                            st.pyplot(fig, clear_figure=True)

                else:
                    st.info("Sayısal özellikler için ANOVA-benzeri skor hesaplanamadı.")


