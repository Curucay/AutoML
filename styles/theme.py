def get_custom_css():
    """Özel CSS stilleri"""
    return """
    <style>
    /* Konteyner paddinglerini minimuma indir */
    .block-container {
        padding: 0.75rem 1rem 1.25rem 1rem;
    }

    /* ====== NAVBAR ====== */
    
    .top-nav {
        position: sticky;
        top: 0;
        z-index: 1000;
        height: 56px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
    }

    /* Koyu/Açık tema için arka plan */
    @media (prefers-color-scheme: dark) {
        .top-nav {
            background: rgba(20,20,24,0.95);
            border-color: rgba(255,255,255,0.08);
        }
        .brand .title {
            color: #e5e7eb;
        }
        .pill {
            background: rgba(255,255,255,0.1);
            color: #e5e7eb;
        }
    }

    @media (prefers-color-scheme: light) {
        .top-nav {
            background: rgba(255,255,255,0.95);
            border-color: rgba(0,0,0,0.06);
        }
        .brand .title {
            color: #111827;
        }
        .pill {
            background: rgba(0,0,0,0.05);
            color: #374151;
        }
    }

    .top-nav-inner {
        width: 100%;
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 1rem;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }

    .brand {
        justify-self: flex-start !important;
        margin-right: auto !important;
    }

    .brand .logo {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    }

    .brand .title {
        font-weight: 700;
        font-size: 1.05rem;
    }

    .nav-actions {
        display: flex-end !important;
        margin-left: auto !important;
    }

    .pill {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* ====== SIDEBAR STYLING ====== */
    .stSelectbox > div > div {
        border-radius: 10px;
    }

    /* Gizli elementlerin yerini de kaldır */
    .stSidebar .st-emotion-cache-3gfel5,
    .stSidebar .ej6j6k44,
    .stSidebar ul li:first-child,
    .stSidebar ul li:nth-child(2) {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Sidebar'ın en üstündeki boşluğu kaldır */
    .stSidebar > div:first-child {
        padding-top: 0 !important;
    }

    .st-emotion-cache-kuzxwl.ej6j6k411,
    [data-testid="stSidebarNavSeparator"] {
        display: none !important;
    }

    /* ====== HEADER ====== */
    /* Header'ı başlangıçta gizle - dark tema */
    header[data-testid="stHeader"] {
      height: 0.5rem !important;
      min-height: 0.5rem !important;
      background: transparent !important;
      border-bottom: none !important;
      overflow: hidden !important;
      transition: all 0.3s ease !important;
    }

    /* Header içeriğini başlangıçta gizle */
    header[data-testid="stHeader"] > div {
      opacity: 0 !important;
      transition: opacity 0.3s ease !important;
    }

    /* Mouse header'a geldiğinde açıl - dark tema */
    header[data-testid="stHeader"]:hover {
      height: 3.5rem !important;
      min-height: 3.5rem !important;
      background: dark !important;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* Mouse geldiğinde tüm içeriği göster */
    header[data-testid="stHeader"]:hover > div {
      opacity: 1 !important;
      display: flex !important;
      visibility: visible !important;
    }

    /* Header içindeki tüm elementleri görünür yap */
    header[data-testid="stHeader"]:hover * {
      display: inherit !important;
      visibility: visible !important;
      opacity: 1 !important;
      color: #ecf0f1 !important;
    }

    /* Sidebar kontrolünü her zaman görünür tut */
    [data-testid="collapsedControl"] {
      display: flex !important;
      visibility: visible !important;
      opacity: 1 !important;
      z-index: 9999 !important;
      color: #ecf0f1 !important;
    }

    /* Ana içeriği yukarı çek */
    .block-container {
      padding-top: 1rem !important;
      margin-top: 0 !important;
    }
    """