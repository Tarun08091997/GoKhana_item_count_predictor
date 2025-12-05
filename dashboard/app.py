"""
Streamlit Dashboard for GoKhana Pipeline Results
Main entry point for the dashboard application.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path to import utilities
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from dashboard.pages.fri_result import show_fri_result

# Page configuration
st.set_page_config(
    page_title="GoKhana Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* Fix tabs at top and prevent horizontal scrolling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        overflow-x: visible !important;
        overflow-y: visible !important;
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: white;
        border-bottom: 2px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        overflow: visible !important;
    }
    /* Prevent horizontal scroll on main container */
    .main .block-container {
        max-width: 100%;
        padding-top: 1rem;
        overflow-x: hidden;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">📊 GoKhana Pipeline Dashboard</div>', unsafe_allow_html=True)
    
    # Create top tabs (like browser tabs)
    tab1 = st.tabs(["FRI Result"])[0]
    
    with tab1:
        show_fri_result()

if __name__ == "__main__":
    main()
else:
    # When running via streamlit run, call main directly
    main()
