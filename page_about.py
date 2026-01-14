# -*- coding: utf-8 -*-
"""
@author: pmartins
"""

# Package imports
import streamlit as st

def page_about():
        
    
    # st.markdown("<h2 style='text-align: center;'>About</h2>", 
    #             unsafe_allow_html=True)
    # st.markdown('''
    # **ACCU-RATES** is a free-to-use web tool for accurate determiantion of ...\n
    # ''')
    # st.markdown('Please <a href="mailto:nagpkin@gmail.com">let us know</a> if you have any questions or need more in-depth analysis.', unsafe_allow_html=True)
    
    # # Lower next markdowns
    # st.write("")

   
    def make_line():
        """ Line divider between images. """
            
        line = st.markdown('<hr style="border:1px solid gray"> </hr>',
                unsafe_allow_html=True)

        return line    

    st.markdown("<h1 style='text-align: center; color: grey;'>🧐 ACCU-RATES </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> Guidelines </h2>", 
                unsafe_allow_html=True)
   
    st.markdown("""
    ACCU-RATES calculates enzyme kinetic parameters ($K_m$ and $V$) with high precision, 
    eliminating the need for subjective 'linear phase' selection.
    """)

    st.divider()

    # --- SECTION 1: DATA PREPARATION ---
    st.header("1. Prepare Your File")
    
    # col1, col2 = st.columns([2, 1])
    
    # with col1:
    st.markdown("""
    Download and fill out the **.ods (OpenDocument Spreadsheet)** template. 
    It is compatible with **Excel** and **LibreOffice**.
    
    * **Template 1:** Unique time columns for every curve.
    * **Template 2:** One shared time column for all curves.
    
    **Key Requirements:**
    * **Substrate ($S_0$):** You must provide the initial concentration for every curve.
    * **Units:** Use concentration (M, mM, µM) OR raw data (Absorbance/Fluorescence).
    * **Flexibility:** You can leave gaps or remove outliers; the tool handles uneven columns automatically.
    """)
    
    # with col2:
        # st.info("**Minimum Data:** Only 2 time points per curve are required.")
        # Replace with your actual hosted image link or local path
        # st.image("path/to/template_screenshot.png", caption="Template Overview")
        

    st.divider()

    # --- SECTION 2: UPLOAD & MODES ---
    st.header("2. Upload & Settings")
    
    st.markdown("""
    Upload your file via the sidebar or main dashboard. Choose your calculation mode:
    
    * **Default Mode:** Use this if you have a stable background or a standard calibration.
    * **Variable Background:** Select this if you are using raw signal units without a calibration curve. 
        *Note: In this mode, $V$ will be in arbitrary units, but $K_m$ will remain in concentration units.*
    """)

    # --- SECTION 3: RESULTS ---
    st.header("3. Review Results")
    
    # st.image("path/to/results_preview.png")
    

    st.success("""
    **The tool generates two views:**
    1. **Fitted Curves:** Visual validation of the numerical fit against your raw data.
    2. **MM Plot & Constants:** Final $K_m$ and $V$ values with **Standard Error** for your reports.
    """)
   
    st.subheader("⚠️ Data Quality & Limitations")
    st.markdown("""
    While ACCU-RATES is robust, it cannot fix a poorly designed assay. Note the following:
    
    * **Statistical Power:** If your data has high noise or very few time points, you may see **Standard Errors >100%**. To fix this, increase your sampling frequency to better capture the early reaction stages.
    * **Heuristic Constants ($A$ & $B$):** These parameters help to interpret the results. If you notice the ratio **$A/B$** changes significantly when plotted against substrate concentration ($S_0$), it may indicate time-dependent enzyme inactivation or other non-ideal kinetics.
    """)
    
    make_line()
    
    st.markdown("<h2 style='text-align: center;'>Disclaimer</h2>", 
                unsafe_allow_html=True)
    st.markdown('''
    This webtool is free to use for all non-commercial purposes. This webtool is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this webtool, even if advised of the possibility of such damage.
    ''')
    
    make_line()
    
    # Lower next markdowns
    st.write("")
    
    st.markdown("<h2 style='text-align: center;'>Help and Support</h2>", 
                unsafe_allow_html=True)
    st.markdown('''
    If you need technical assistance or a stand-alone version of ACCU-RATES please contact
    pmartins2106@gmail.com. 
    ''')
    
    make_line()
    
    # Lower next markdowns
    st.write("")
    
    st.markdown("<h2 style='text-align: center;'>Funding</h2>", 
                unsafe_allow_html=True)
    # st.image("images//logos.png", use_container_width=True)
    
    st.markdown('''
                This work is funded by the Portuguese Foundation for Science and Technology (FCT) in the framework of project PTDC/QUICOL/2444/2021, FCT 2023.11892.PEX, and COMPETE2030-FEDER-00690800.
    ''')
    
# page_about()
