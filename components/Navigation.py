# Components/Navigation.py
from pathlib import Path
from config.Settings import Settings
import base64
import streamlit as st

def img_to_b64(path: Path) -> str | None:
    p = path if path.is_absolute() else (Settings.ROOT / path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class Navigation:
    MENU_STRUCTURE = {
        "📊 Veri Seti": {
            "Genel Bakış": "data_overview",
            # diğer sayfalar...
        }
    }

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            b64 = img_to_b64(Settings.ROOT / "assets" / "CodeCosmosIcon.png")
            if b64:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:-8px; margin-bottom:8px;">
                      <img src="data:image/png;base64,{b64}" alt="Code Cosmos" style="width:120px;height:auto;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("Logo bulunamadı.")

            if "active_page" not in st.session_state:
                st.session_state["active_page"] = "data_overview"

            for category_name, items in Navigation.MENU_STRUCTURE.items():
                with st.expander(category_name, expanded=True if "Raporlar" in category_name else False):
                    for item_name, item_key in items.items():
                        if st.button(
                                f"  {item_name}",
                                key=f"nav_{item_key}_{category_name}",
                                use_container_width=True
                        ):
                            # 1) Hedef sayfayı ata
                            st.session_state["active_page"] = item_key
                            # 2) Sayfa değişiminde filtreleri sıfırla (prefixsiz kullanıyoruz)
                            st.session_state["__pending_clear"] = True
                            # 3) Top-level rerun
                            st.rerun()

        return st.session_state["active_page"]
