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
# import io
# from scipy.optimize import curve_fit
from scipy import stats
# from scipy.stats import t as tt
# from scipy.special import lambertw


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
            df_load = pd.read_excel(input, skiprows=3, engine="odf", 
                                    header = None) #nrows= 1000, usecols = range(0,18), 
            Ncurves0 = int(df_load.shape[0]/2)
            S0_load = df_load.iloc[0][0:min(Ncurves0,10)]
            S0_load.dropna(how='all', axis=0, inplace=True)
            S0 = list(S0_load)
            df_load = df_load.iloc[4:]
            # Remove empty columns
            df_load.dropna(how='all', axis=1, inplace=True)
            # number of curves
            Ncurves = S0_load.shape[0]
            # Curve1, Curve2...
            colist = ['Curve '+ str(i) for i in range(1, Ncurves+2)]
            # df_load.columns =  np.concatenate((['Time'], colist), axis=0)
            flag_return = 0
        except:
            st.error('Error - check if the uploaded file is in the right format')
            df_load = 0
            Ncurves  = 0
            colist = 0
            flag_return = 1
            st.stop()
            
        return df_load, S0, Ncurves, colist, flag_return
    
    
    input = st.file_uploader('')
 
    # The run_example variable is session state and is set to False by default
    # Therefore, loading an example is possible anytime after running an example.  
    st.write('Upload your data') 
    st.write('---------- OR ----------')
    st.session_state.run_example = False
    st.session_state.run_example = st.checkbox('Run a prefilled example') 
        
    # Ask for upload (if not yet) or run example.       
    if input is None:
        # Get the template  
        download_sample = st.checkbox("Download template")
      
    try:
        if download_sample:
            with open("datasets//ACCU-RATES_template.ods", "rb") as fp:
                st.download_button(
                label="Download",
                data=fp,
                file_name="ACCU-RATES_template.ods",
                mime="application/vnd.ms-excel"
                )
            st.markdown("""**fill the template; save the file (keeping the .ods format); 
                        upload the file into ACCU-RATES*""")
           
            # if run option is selected
        if st.session_state.run_example:
            input = "datasets/ACCU-RATES_example.ods"
            df, S0, Ncurves, colist, flag_return = load_odf()
    except:
        # If the user imports file - parse it
       if input:
           with st.spinner('Loading data...'):
                df, S0, Ncurves, colist, flag_return = load_odf()
    
    
    # Error Handling
    def analysisterminated(err):
        if err == 0:
            st.error("ACCU-RATES method not applied. Check if the input file contains at" + 
                     " least 5 different progress curves.")
        elif err == 1:
            st.error("ACCU-RATES method not applied. Check if the input file and calibration curves are OK")
        st.stop()
    
    def extract_curves(dfi):
        # remove empty cells
        dfi = dfi.dropna()
        ydata = dfi.iloc[:, 1]
        xdata = dfi.iloc[:, 0]
        return xdata, ydata
    
    #plot experimental and fitted data
    def plot_fit(p_x, p_xfit, p_y, p_yfit, i):
        if plot_mode == 'Dark Mode':
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        if i < 10:
            plt.scatter(p_x, p_y, color=colors[i], label=S0[i]) #
        elif i == 11:
            plt.scatter(p_x, p_y, color=colors[i], label = '...')
        else:
            plt.scatter(p_x, p_y, color=colors[i])
                
        plt.plot(p_xfit, p_yfit, color='g', linewidth=1)
        
        
    # def integratedMM(x, param0, param1, param2):
    #     K = param0
    #     V = param1
    #     S = param2
    #     y = S - K*(lambertw(S/K*np.exp((S-V*x))/K))
    #     return y
    
                
    
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
            
        # Check if at least 5 different S0 values are present 
        if len(S0) < 5:
            err = 0
            analysisterminated(err)
        
        # If S vs t curves are used, conversion to P vs t is needed
        SvstFlag = 0
        for n in range(Ncurves):
            dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
            t, P = extract_curves(dfi)
            if list(P)[-1] < list(P)[0]:
                SvstFlag = SvstFlag + 1      
        
        if SvstFlag == Ncurves:
            st.warning('Substrate depletion curves were detected. Conversion to' + 
                                ' Product production curves was performed using the approximation $P = S_0-S$.')
        
        # Defining number of final sampled points
        nosample = 7
        
        pointnumber = []
        for n in range(Ncurves):
            dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
            xdata, ydata = extract_curves(dfi)
            pointnumber.extend([len(xdata)]);
            
        # # Fit full progress curve
        # fig_fit = plt.figure(facecolor=backc)
        
        # Km_int = []
        # Vmax_int = []
        # S0_int = []
        
        # for n in range(Ncurves):
        #     try:
        #         dfi = df.iloc[:, [2*n, 2*n+1]].astype(float)
        #         t, P = extract_curves(dfi)
        #         if SvstFlag == Ncurves:
        #             P = S0[n] - P
        #         ig = np.asarray([max(S0)/5, max(S0)/max(t), S0[n]])
        #         bds =  ([0, 0, 0], [np.inf, np.inf, np.inf])
        #         parameters, covariance = curve_fit(integratedMM, t, P, p0=ig,
        #                                         bounds=bds, maxfev=10000) #xtol=1e-20*tmax, ftol=1e-20*ymax, 
                
        #         Km_int.extend([parameters[0]])
        #         Vmax_int.extend([parameters[1]])
        #         S0_int.extend([parameters[2]])
        #         p_xfit = np.linspace(0, max(t))
        #         p_yfit = integratedMM(p_xfit, parameters[0], parameters[1], parameters[2])
        #         plot_fit(t, p_xfit, P, p_yfit, n) 
        #     except:
        #         S0_int.extend([S0[n]])
        #         Km_int.extend([0])
        #         Vmax_int.extend([0])
        
        # Flag is 1 is at least one vector is found to have less than 3 points.    
        pointnumber_flag = all(i < 3 for i in pointnumber)
        if not(pointnumber_flag):
            
            percentile95 = []
            index95 = []
            intfilt_all = []
            for n in range(Ncurves):
                dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
                t, P1 = extract_curves(dfi)
                P = P1*slope + intercept
                if SvstFlag == Ncurves:
                    P = S0[n] - P
                percentile95.extend([np.percentile(P, 95)])
                index95.extend([(np.abs(P - percentile95[n])).argmin()])
                intfilt_all.extend([round(index95[n]/nosample)]);
            # Definition of largest interval for strong filter application
            intfilthalf = round(max(intfilt_all)/2) if round(max(intfilt_all)/2) >0 else 1 ;
            
            Pfiltered = []
            tfiltered = []
            for n in range(Ncurves):
                dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
                t, P1 = extract_curves(dfi)
                P = P1*slope + intercept
                if SvstFlag == Ncurves:
                    P = S0[n] - P
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
            st.error('ACCU-RATES method not applied. Number of timepoints < 3.')
            st.stop()

        # Non-unique arrays
        if UniqueFlag == 1:
            Psampled = []
            tsampled = []
            for n in range(Ncurves):
                tempP = Pfiltered[n]
                tempt = tfiltered[n]
                Psampled.append(np.transpose(tempP))
                tsampled.append(np.transpose(tempt))
        
        # (2) Individual Curve Analysis------------------------------------------------
        
        
        #     # Unlike Interferenzy, Pinf always correspond to S0 (for simplicity)
                # Choosing whether to use Pinf or not.
        # eval = []
        FinalP = []
        for n in range(Ncurves):
        #     eval.extend([(Psampled[n][-1] > 0.9*S0[n]) & (Psampled[n][-1] < 1.1*S0[n])])
        # for n in range(Ncurves):
        #     if all(eval): # & SvstFlag != Ncurves
        #         FinalP.extend([S0_int[n]]) #max(Psampled[n])
        #     else: 
        #         FinalP.extend([S0[n]])
            FinalP.extend([S0[n]])
        
        # Determination of Y and X for ti = first points. 
        # Unlike Interferenzy, a single initial timpoint (the second) is assumed 

            # linearlization
        X = []
        Y = []
        for n in range(Ncurves):
            linearX = -np.log(1 - (Psampled[n][2] - Psampled[n][1]) / 
                              (FinalP[n] - Psampled[n][1])) / (tsampled[n][2] - tsampled[n][1])
            linearY = (Psampled[n][2] - Psampled[n][1]) / (tsampled[n][2] - tsampled[n][1])
            X.extend([linearX])
            Y.append(np.transpose(linearY))

        
        
        res = stats.linregress(np.multiply(X,-1),Y)
        # Two-sided inverse Students t-distribution
        # p - probability, dfr - degrees of freedom
        # tinv = lambda p, dfr: abs(tt.ppf(p/2, dfr))
        # ts = tinv(0.05, len(X)-2)
        Km = res.slope
        Vmax = res.intercept
        if np.isnan(Km):
            analysisterminated(1)


        v0 = []
        for n in range(Ncurves):
            v0.extend([S0[n]*Vmax / (S0[n] + Km)])
        v0_df = pd.DataFrame({'S0': S0,
                   'v0': v0})
       
        st.success("""
                    ACCU-RATES plots and reports
                    """)
        
        # Plot linearization fit
        fig1 = plt.figure(facecolor=backc)
        p_xfit = np.linspace(0, max(X), 200)
        p_yfit = res.intercept - res.slope*p_xfit
        plot_fit(X, p_xfit, Y, p_yfit, 0)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("Linearization Plot")
        st.pyplot(fig1)
        
        
        # plot initial rates
        fig2 = plt.figure(facecolor=backc)
        Pmax = max([max(Psampled[i]) for i in range(Ncurves)])
        p_tmax = Pmax/max(v0)
        p_xfit = np.linspace(0, p_tmax, 200)
        for n in range(Ncurves):
            dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
            t, P1 = extract_curves(dfi)
            P = P1*slope + intercept
            if SvstFlag == Ncurves:
                P = S0[n] - P
            p_yfit = v0[n]*(p_xfit - tsampled[n][1]) + Psampled[n][1]
            plot_fit(t, p_xfit, P, p_yfit, n)
        plt.legend(title="S0")
        plt.title("Initial Rates")
        plt.xlabel('Time')
        plt.ylabel('Product Concentration')
        st.pyplot(fig2)
        
        fig3 = plt.figure(facecolor=backc)
        p_xfit = np.linspace(0, max(S0))
        p_yfit = Vmax*p_xfit / (Km + p_xfit)
        plot_fit(S0, p_xfit, v0, p_yfit, 5)
        plt.title("Michaelis-Menten Plot")
        plt.xlabel('Substrate Concentration (S0)')
        plt.ylabel('Initial Rates (v0)')
        st.pyplot(fig3)      
        
        # Reporting
        st.info('Kinetic Parameters:')
        st.write('\nKm (95%): ' + repr(round(res.slope,6)) + 
                          ' +/- ' + repr(round(res.stderr,6)))
        st.write('\nVmax (95%): ' + repr(round(res.intercept,6)) + 
                          ' +/- ' + repr(round(res.intercept_stderr,6)))
        st.info('\n\nInitial Rates:\n')
        st.write(v0_df)
      
# page_analyse()
