import streamlit as st
import pandas as pd
from config.Settings import Settings
from components.Navigation import Navigation
from styles.theme import get_custom_css
from pages.DataOverview import DataOverview

def main():
    # Sayfa konfigürasyonu
    st.set_page_config(
        page_title=Settings.PAGE_TITLE,
        page_icon=Settings.PAGE_ICON,
        layout=Settings.LAYOUT,
        initial_sidebar_state="expanded"
    )

    # Pandas ayarları
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)

    # Özel CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Navigasyon
    selected_page = Navigation.render_sidebar()

    # Sayfa routing
    if selected_page == "data_overview":
        DataOverview().render()


if __name__ == "__main__":
    main()

