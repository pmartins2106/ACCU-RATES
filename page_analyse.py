# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:04 2023

@author: pmartins
"""

# Package imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.stats import pearsonr



def page_analyse():
    
    # function to make expander-selection
    def make_expanders(expander_name, sidebar=True):
        """ Set up Figure Mode expander. """
        if sidebar:         
            try:
                return st.sidebar.expander(expander_name)
            except:
                return st.sidebar.beta_expander(expander_name)
    
    # Select type of progress curves
    with make_expanders("**Type of progress curves:**"):
        # st.markdown("**type of progress curves:**")
        analysis_mode = st.radio("Options", ('in concentration units'
                                             , 'raw data - calibration curve needed:'))
        if analysis_mode == 'in concentration units':
            slope = 1
            intercept = 0
        else:
            slope = st.number_input("Slope [concentration/signal]", format="%2.2f", value = 1.)
            intercept = st.number_input("Intercept [concentration]", format="%2.2f", value = 0.)

    # analysis_mode = st.sidebar.radio('**What type of progress curves do you have?**', ('Mass-based', 'Size-based'))   
    # st.sidebar.markdown("#")
  
    st.markdown("<h2 style='text-align: center;'> Enzyme Kinetic Analysis </h2>", 
                unsafe_allow_html=True)
        
    st.info("""
                - Upload progress curves measured at different (>4) substrate concentrations
                - Find Michaelis-Menten parameters
                - Estimate  initial reaction rates
                """)
    # st.markdown("#")
    
    
    # Figure display properties expander
    with make_expanders("Select Figure Mode:"):
        # st.markdown("Select Figure Mode:")
        plot_mode = st.radio("Options", ('Dark Mode', 'Light Mode'))
    
    def load_odf():
        """ Get the loaded .odf into Pandas dataframe. """
        try:
            df_load = pd.read_excel(input, skiprows=0,  engine="odf", header = None)
            template = df_load.iloc[0,1]
            S0_load = df_load.iloc[4,:]
            S0_load.dropna(how='all', axis=0, inplace=True)
            S0 = list(S0_load)
            df_load = df_load.iloc[8:]
            # Remove empty columns
            df_load.dropna(how='all', axis=1, inplace=True)
            # number of curves
            Ncurves = S0_load.shape[0]
            # Curve1, Curve2...
            colist = ['Curve '+ str(i) for i in range(1, Ncurves+2)]
            flag_return = 0
        except:
            df_load = 0
            Ncurves  = 0
            colist = 0
            flag_return = 1
            st.stop()
            
        return df_load, S0, Ncurves, colist, flag_return, template
    
    
    input = st.file_uploader('')
 
    # The run_example variable is session state and is set to False by default
    # Therefore, loading an example is possible anytime after running an example.  
    st.write('Upload data') 
    st.write('---------- OR ----------')
    st.session_state.run_example = False
    st.session_state.run_example = st.checkbox('Run a prefilled example') 
        
    # Ask for upload (if not yet) or run example.       
    if input is None:
        # Get the template  
        download_sample = st.checkbox("Download template")
      
    try:
        if download_sample:
            template_mode = st.radio("Choose Template", ('Multiple Time Columns', 
                                                         'Single Time Column'))
            if template_mode == 'Single Time Column':
                template_address = "datasets//ACCU-RATES_template2.ods"
                slcted_template = "ACCU-RATES_template2.ods"
            else:
                template_address = "datasets//ACCU-RATES_template1.ods"
                slcted_template = "ACCU-RATES_template1.ods"
                
            with open(template_address, "rb") as fp:
                st.download_button(
                label="Download",
                data=fp,
                file_name=slcted_template,
                mime="application/vnd.ms-excel"
                )
            st.markdown("""**fill the template; save the file (keeping the .ods format); 
                        upload the file into ACCU-RATES*""")
           
            # if run option is selected
        if st.session_state.run_example:
            input = "datasets/ACCU-RATES_template2.ods"
            df, S0, Ncurves, colist, flag_return, template = load_odf()
    except:
        # If the user imports file - parse it
       if input:
           with st.spinner('Loading data...'):
                df, S0, Ncurves, colist, flag_return, template = load_odf()
    
    
    def extract_curves(dfi):
        # remove empty cells
        dfi = dfi.dropna()
        ydata = dfi.iloc[:, 1]
        xdata = dfi.iloc[:, 0]
        return xdata, ydata
    
    #plot experimental and fitted data
    def plot_fit(p_x, p_xfit, p_y, p_yfit, i):
        # if plot_mode == 'Dark Mode':
        #     plt.style.use("dark_background")
        # else:
        #     plt.style.use("default")
        if i < 10:
            plt.scatter(p_x, p_y, color=colors[i], label=S0[i]) #
        elif i == 11:
            plt.scatter(p_x, p_y, color=colors[i], label = '...')
        else:
            plt.scatter(p_x, p_y, color=colors[i])
                
        plt.plot(p_xfit, p_yfit, color='g', linewidth=1)
               
    
    # DATA ANALYSIS AND REPORTING
    
    if input and flag_return == 0:
        # define color in plots
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colormap = plt.cm.coolwarm 
        colors = [colormap(i) for i in np.linspace(0, 1, Ncurves)]
        #background color
        if plot_mode == 'Dark Mode':
            backc = 'black'
        else:
            backc = 'white'   
    
        # If S vs t curves are used, conversion to P vs t is needed        
        mint = []
        maxt = []
        minP = []
        maxP = []
        SvstFlag = 0
        for n in range(Ncurves):
            if template == 1:
                dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
            else:
                dfi = df.iloc[:, [0,n+1]].astype(float) 
            t, P = extract_curves(dfi)
            P = P*slope + intercept
            mint.extend([min(t)])
            maxt.extend([max(t)])
            minP.extend([min(P)])
            maxP.extend([max(P)])
            if list(P)[-1] < list(P)[0]:
                SvstFlag = SvstFlag + 1
        
        if SvstFlag == Ncurves:
            st.warning('Substrate depletion curves were detected. Conversion to' + 
                                ' Product production curves was performed using the approximation $P = S_0-S$.')
       
        # Defining number of final sampled points
        nosample = 7
    
        pointnumber = []
        for n in range(Ncurves):
            if template == 1:
                dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
            else:
                dfi = df.iloc[:, [0,n+1]].astype(float) 
            xdata, ydata = extract_curves(dfi)
            pointnumber.extend([len(xdata)]);
        
        # Fit full progress curve
        def dS_dt(S, t, V, K):
            return - V * S / (K + S)
    
        def substrate_ode(t, V, K):
            P = Si - odeint(dS_dt, Si, t, args=(V, K))
            return P.flatten()
        
        fig1 = plt.figure(facecolor=backc)
        # fig2 = plt.figure(facecolor=backc)
        Km_int = []
        Vmax_int = []
        S0_int = []
        for n in range(Ncurves):
            try:
                if template == 1:
                    dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
                else:
                    dfi = df.iloc[:, [0,n+1]].astype(float) 
                t, P = extract_curves(dfi)
                P = P*slope + intercept
                if SvstFlag == Ncurves:
                    P = S0[n] - P
                dt = t-list(t)[0]
                dP = P -list(P)[0]
                ig = np.asarray([max(S0)/10000, max(np.diff(dP)/np.diff(dt))])
                bds =  ([0, ig[1]], [np.inf, np.inf])
                Si = S0[n] - list(P)[0]
                parameters, covariance = curve_fit(substrate_ode, dt, dP, 
                                  p0=ig, bounds=bds, maxfev=10000) #xtol=1e-20*tmax, ftol=1e-20*ymax, 
                
                
                Km_int.extend([parameters[1]])
                Vmax_int.extend([parameters[0]])
                S0_int.extend([Si])
                p_xfit = np.linspace(0, max(dt))
                p_yfit = substrate_ode(p_xfit, parameters[0], parameters[1])
                if plot_mode == 'Dark Mode':
                    plt.style.use("dark_background")
                else:
                    plt.style.use("default")
                plt.subplot(1, 2, 1)
                plot_fit(dt, p_xfit, dP, p_yfit, n)                         
                plt.subplot(1, 2, 2)
                plot_fit(dt, p_xfit, dP/Si, p_yfit/Si, n)                         
            except:
                st.error("Error: ACCU-RATES method not applied.")
                st.stop()
       
        plt.subplot(1, 2, 1)
        plt.legend(title="S0")
        plt.title("Progress Curves")
        plt.xlabel('Time')
        plt.ylabel('Product Concentration')
        plt.subplot(1, 2, 2)
        plt.legend(title="S0")
        plt.title("Normalized Units")
        plt.xlabel('Time')
        
        st.pyplot(fig1)  
        
        # Flag is 1 is at least one vector is found to have less than 3 points.    
        pointnumber_flag = all(i < 3 for i in pointnumber)
    
        if not(pointnumber_flag):
            
            percentile95 = []
            index95 = []
            intfilt_all = []
            for n in range(Ncurves):
                if template == 1:
                    dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
                else:
                    dfi = df.iloc[:, [0,n+1]].astype(float) 
                t, P = extract_curves(dfi)
                P = P*slope + intercept
                if SvstFlag == Ncurves:
                    P = S0[n] - P
                t = t-list(t)[0]
                P = P -list(P)[0]
                percentile95.extend([np.percentile(P, 95)])
                index95.extend([(np.abs(P - percentile95[n])).argmin()])
                intfilt_all.extend([round(index95[n]/nosample)]);
            
            # Definition of largest interval for strong filter application
            intfilthalf = round(max(intfilt_all)/2) if round(max(intfilt_all)/2) >0 else 1 ;
            
            Pfiltered = []
            tfiltered = []
            for n in range(Ncurves):
                if template == 1:
                    dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
                else:
                    dfi = df.iloc[:, [0,n+1]].astype(float) 
                t, P = extract_curves(dfi)
                P = P*slope + intercept
                if SvstFlag == Ncurves:
                    P = S0[n] - P
                t = t-list(t)[0]
                P = P -list(P)[0]
                filtereddata = []
                filteredtime = []
                for q in range(intfilthalf, (pointnumber[n]-intfilthalf+2)):
                    filtereddata.extend([np.mean( P[(q - intfilthalf):(q + intfilthalf)] )])  
                    filteredtime.extend([np.mean( t[(q - intfilthalf):(q + intfilthalf)] )])
                Pfiltered.append(np.transpose(filtereddata))
                tfiltered.append(np.transpose(filteredtime))
            
            # Sampling x Points, Product-wise
            Psampled = []
            tsampled = []
            UniqueFlag = 0
            for n in range(Ncurves):
                tempP = Pfiltered[n]
                tempt = tfiltered[n]
                productsamples = np.linspace(min(tempP) , max(tempP), nosample )
                sampleP = []
                samplet = []
                for q in range(nosample):
                    closestindex = (np.abs(tempP - productsamples[q])).argmin()
                    sampleP.extend([tempP[closestindex]])
                    samplet.extend([tempt[closestindex]])
                # check if arrays are unique
                if len(sampleP) > len(set(sampleP)):
                    UniqueFlag = 1
                Psampled.append(np.transpose(sampleP))
                tsampled.append(np.transpose(samplet))
                
        # Less than 3 timepoints/curve or non-unique  
        else:
            Psampled = []
            tsampled = []
            for n in range(Ncurves):
                tempP = Pfiltered[n]
                tempt = tfiltered[n]
                Psampled.append(np.transpose(tempP))
                tsampled.append(np.transpose(tempt))
            st.warning('\n Interferenzy analysis not performed. Number of timepoints < 3.\n')

        # Non-unique arrays
        if UniqueFlag == 1:
            Psampled = []
            tsampled = []
            for n in range(Ncurves):
                tempP = Pfiltered[n]
                tempt = tfiltered[n]
                Psampled.append(np.transpose(tempP))
                tsampled.append(np.transpose(tempt))

        # Fit MM curve
        def MM(x, param0, param1):
            K = param0
            V = param1
            y = x*V/(K+x)
            return y
        
        st.success("""
                    ACCU-RATES plots and reports
                    """)
        
        v0 = []
        for n in range(Ncurves):
            v0.extend([S0[n]*Vmax_int[n] / (S0[n] + Km_int[n])])
        v0_df = pd.DataFrame({'S0': S0,
                   'v0': v0})
            
   
        ig = np.asarray([max(S0)/5, max(v0)])
        bds =  ([0, 0], [np.inf, np.inf])
        parameters, covariance = curve_fit(MM, S0, v0, p0=ig,
                                       bounds=bds, maxfev=10000)

        Km = parameters[0]
        Vmax = parameters[1]
        sigma_ab = np.sqrt(np.diagonal(covariance))

        fig2 = plt.figure(facecolor=backc)
        if plot_mode == 'Dark Mode':
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        plt.scatter(S0, v0, facecolors='none', edgecolors='r', s = 70, label='ACCU-RATES')
        p_xfit = np.linspace(0, max(S0))
        plt.plot(p_xfit, Vmax*p_xfit / (Km + p_xfit), 'b', label='fitted line')
        plt.xlabel('S0')
        plt.ylabel('v0')
        plt.legend(loc="upper left")
        st.pyplot(fig2)
        
        # Interferenzy Analysis------------------------------------------------
        FinalP = S0_int
        X = []
        Y = []
        for n in range(Ncurves):
            arg_log = 1 - (Psampled[n][1] - Psampled[n][0])/(FinalP[n] - Psampled[n][0])
            if arg_log > 0:
                linearX = -np.log(arg_log) / (tsampled[n][1] - tsampled[n][0])
                linearY = (Psampled[n][1] - Psampled[n][0]) / (tsampled[n][1] - tsampled[n][0])
                X.extend([linearX])
                Y.append(np.transpose(linearY))
            else:
                st.warning('\n\nError during Interferenzy linearization in at least 1 curve.')

        Y2 = []
        for n in range(Ncurves):
            Y2.append(Vmax_int[n] - Km_int[n]*X[n])
   
    
        # Reporting
        st.info('Simplified Interferenzy Analysis')
        try:
            # Pearson correlation
            corr, p_value = pearsonr(Y, Y2)
            st.write(f"\n\nPearson correlation: {corr:.3f}, p-value: {p_value:.3e}")
        except:
            st.write('\n\nInterferenzy validation not possible.') 
            
        def count_sig_figs(value):
            """
            Count significant figures in a number.
            
            Args:
                value (float): Number to analyze
            
            Returns:
                int: Number of significant figures
            """
            if value == 0 or np.isinf(value):
                return 1
            str_value = f"{abs(value):.10e}".split("e")[0].replace(".", "").rstrip("0")
            return len(str_value)

        def round_to_sig_figs(value, sig_figs):
            """
            Round a number to specified significant figures.
            
            Args:
                value (float): Number to round
                sig_figs (int): Number of significant figures
            
            Returns:
                float: Rounded number
            """
            if value == 0 or np.isinf(value):
                return 0
            magnitude = np.floor(np.log10(abs(value)))
            factor = 10 ** (sig_figs - 1 - magnitude)
            return round(value * factor) / factor

        def format_mean_sd(mean, sd):
            """
            Format mean with precision set by SD's effective significant figures.
            Rounds SD to practical magnitude (e.g., 10.00001 -> 10).
            
            Args:
                mean (float): Mean value
                sd (float): Standard deviation
            
            Returns:
                str: Formatted string as 'mean ± sd' with consistent precision
            """
            # Handle infinite SD
            if np.isinf(sd):
               # Use default precision (e.g., 2 sig figs) for mean
               default_sig_figs = 2
               mean_rounded = round_to_sig_figs(mean, default_sig_figs)
               if abs(mean_rounded) < 1e-4 or abs(mean_rounded) > 1e4:
                   return f"{mean_rounded:.1e} ± inf"
               mean_str = f"{abs(mean_rounded):.2f}".rstrip("0").rstrip(".")
               decimal_places = len(mean_str.split(".")[1]) if "." in mean_str else 0
               return f"{mean_rounded:.{decimal_places}f} ± inf"
            
            # Round SD to practical magnitude (e.g., 10.00001 -> 10)
            sd_magnitude = round(abs(sd))
            if sd_magnitude == 0:
                sd_rounded = sd
                sig_figs =  min(count_sig_figs(sd) + 1, 2)
            else:
                # Use 2 sig figs for SDs like 10.00001, ignoring minor decimals
                sig_figs =  min(count_sig_figs(sd) + 1, 2)
                sd_rounded = round_to_sig_figs(sd, sig_figs)
            
            # Round mean to match SD's precision
            if sd_magnitude >= 1:
                # For SDs like 10, round mean to same decimal places as rounded SD
                decimal_places = max(1, -int(np.log10(abs(sd_rounded))))
                mean_rounded = round(mean, decimal_places)
            else:
                # For small SDs, use sig figs
                mean_rounded = round_to_sig_figs(mean, sig_figs)
            
            # Choose format based on magnitude
            if abs(mean_rounded) < 1e-4 or abs(mean_rounded) > 1e4 or abs(sd_rounded) < 1e-4 or abs(sd_rounded) > 1e4:
                format_string = f"{{:.{sig_figs-1}e}} ± {{:.{sig_figs-1}e}}"
            else:
                mean_str = f"{abs(mean_rounded):.{sig_figs}f}".rstrip("0").rstrip(".")
                decimal_places = len(mean_str.split(".")[1]) if "." in mean_str else 0
                format_string = f"{{:.{decimal_places}f}} ± {{:.{decimal_places}f}}"
            
            return format_string.format(mean_rounded, sd_rounded)    
        
        st.info('Michaelis-Menten Parameters:')
        st.write(f"\nKm ± SD: {format_mean_sd(Km, sigma_ab[0])}")
        st.write(f"\nV ± SD: {format_mean_sd(Vmax, sigma_ab[1])}")
        st.info('\n\nInitial Rates:\n')
        st.dataframe(v0_df, hide_index=True)
        
        
