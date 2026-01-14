# -*- coding: utf-8 -*-
"""

@author: pmartins 
 
"""
# Neededed for google analytics
from bs4 import BeautifulSoup
import shutil
import pathlib
import logging
# Streamlit
import streamlit as st


#Theming
CURRENT_THEME = "Dark"
IS_DARK_THEME = True

# Add pages
from page_analyse import page_analyse
from page_about import page_about


# Set the default elements on the sidebar
st.set_page_config(page_title='ACCU-RATES')

st.sidebar.markdown("<h2 style='text-align: center;'>ACCU-RATES</h2>", 
            unsafe_allow_html=True)
st.sidebar.success('- Accurate determination of enzyme kinetics and initial reaction rates')

def main_nag():
    """
    Register pages to Explore and Fit:
        page_introduction - contains page with images and brief explanations
        page_analyse - contains various functions that allows user to upload
                    data as a .ods file and fit parameters.
       page_about - about               
    """
    pages = {
        "ANALYSE": page_analyse,
        "GUIDELINES": page_about,
    }

    st.sidebar.title("Main options")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Select:", tuple(pages.keys()))
                                
    # Display the selected page with the session state
    pages[page]()

  
if __name__ == "__main__":
    main_nag()