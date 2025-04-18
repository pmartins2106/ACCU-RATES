# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:39 2023

@author: pmartins
"""
# Needed for google analytics
import streamlit as st

def page_introduction():

    # Lower next markdowns
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    
    # st.markdown("<h2 style='text-align: center;'> NAGkpin Guidelines </h2>", 
    #             unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Introduction</h2>", 
                unsafe_allow_html=True)
    
    st.markdown('''
    **NAGPKin** characterizes the mechanisms of protein phase separation as described [in this paper](https://doi.org/10.1091/mbc.E23-07-0289). Protein phase separation is relevant in both health and disease in phenomena such as the liquid-liquid phase separation of functional droplets, or the self-assembly of proteins into solid aggregates called amyloids.
    ''')
    
    with st.expander("What do I *need*?"):
        st.write('''
            You just need to upload the raw data describing the formation of a new protein phase into section **ANALYSE**. You can choose between mass-based or size-based progress curves.
        ''')
    with st.expander("What are *mass-based progress curves*?"):
        st.write('''
            In mass-based progress curves the mass of the new phase is given as a function of time. Indirect measurements such as thioflavin-T fluorescence are also possible.
        ''')
    with st.expander("What are *size-based progress curves*?"):
        st.write('''
            Size-based progress curves give information about the evolution of the mean size with time. This size can be measured using, for example, image analysis tools or dynamic light scattering.
        ''')
    with st.expander("What information is provided?"):
        st.write('''
            **NAGPKin** quantifies the relative importance of the kinetic steps of primary nucleation, secondary nucleation and growth. Information about the possible occurrence of [Off-Pathway Aggregation](https://doi.org/10.3390/biom8040108), [Surface Tension Effects](https://doi.org/10.1101/2022.11.23.517626) and other parallel processes is also provided.
        ''')
    with st.expander("How good are NAGPKin predictions?"):
        st.write('''
            For better results, progress curves measured at different protein concentrations *[P]* should be used as input to **NAGPKin**. Since only two or three kinetic parameters are fitted to mass- or size-based progress curves, respectively, the use of different *[P]* values decreases the degrees of freedom (and uncertainty) associated with **NAGpkin** predictions. In addition, r-squared values are provided to quantify the goodness of fit.
        ''')
    with st.expander("How can NAGPKin predictions be tested?"):
        st.write('''
            The **NAGPKin** parameters fitted to mass-based progress curves can be used to predict size-based progress curves and *vice versa*. The same parameters can be used to predict the mean size and variance of the new phase’s size distribution. See more details [here](https://doi.org/10.1101/2022.11.23.517626 ). This means that **NAGPKin**’s predictions are directly testable by performing complementary measurements of progress curves and size distributions.
        ''')
    with st.expander("Where can I find more information?"):
        st.write('''
            - A report summarizing the main conclusions of the analysis is provided in section **ANALYSE** together with the links to bibliographic references.
            - [Here](https://doi.org/10.1074/jbc.M112.375345) you can find the Crystallization-Like Model (CLM) describing mass-based progress curves with only two kinetic parameters. These parameters characterize primary nucleation and the combined influence of the autocatalytic processes of secondary nucleation and growth.
        	- [Here](https://doi.org/10.1074/jbc.M115.699348) you can learn how to identify off-pathway aggregation from mass-based progress curves.
        	- [Here](https://doi.org/10.1002/anie.201707345) you can find the CLM extended to consider size-based progress curves.
        	- [Here](https://doi.org/10.1002/advs.202301501) you can find the CLM extended to liquid-liquid phase separation processes, and how to predict particle size distribution from the fitted kinetic parameters.
        	- More application examples of the CLM can be found [here](https://doi.org/10.1021/acs.jpcb.7b01120) (on the effect of molecular crowding), [here]( https://doi.org/10.1002/asia.201801703) (on applications in drug discovery), and [here]( https://doi.org/10.3390/biom8040108) (on the use of kinetic scaling laws).
        ''')
        
        
    # st.info("""
    #         Write Something Here
    #         """)
    # st.info("""
    #         - Write Something Here
    #         - ...
    #         """)

    def make_line():
        """ Line divider between images. """
            
        line = st.markdown('<hr style="border:1px solid gray"> </hr>',
                unsafe_allow_html=True)

        return line    
   

    # st.error('Write Something Here')
    # feature1, feature2 = st.columns([0.5,0.4])
    # # with feature1:
    #     # st.image(image1, use_column_width=True)
    # with feature2:
    #     st.warning('Write Something Here')
    #     st.info("""
    #             - Write Something Here
            
    #             """)
    
    make_line()
  
# page_introduction()
     
    return