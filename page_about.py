# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:04 2023

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
   
    # make_line()
    
    # Lower next markdowns
    # st.write("")

    
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
    pmartins@i3s.up.pt or frocha@fe.up.pt. 
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
