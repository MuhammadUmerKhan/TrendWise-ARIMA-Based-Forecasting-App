import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import itertools
from sklearn.metrics import mean_squared_error

# ----------------------------------Custom CSS for styling-----------------------------------------
st.markdown("""
    <style>
        /* Stock Price Forecasting Theme */
        .stApp {
            background: linear-gradient(rgba(31, 41, 55, 0.9), rgba(31, 41, 55, 0.9)), url('https://towardsdatascience.com/wp-content/uploads/2024/07/16UZCeIFV6-lphBOPnyfKaQ.jpeg');
            background-size: cover;
            background-attachment: fixed;
            color: #f3f4f6;
            font-family: 'Inter', sans-serif;
        }
        .main-container {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.85), rgba(16, 185, 129, 0.85));
            border-radius: 15px;
            padding: 30px;
            margin: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
            border: 2px solid #f97316;
            backdrop-filter: blur(10px);
        }
        .main-title {
            font-size: 3.2em;
            font-weight: 700;
            color: #f97316;
            text-align: center;
            margin-bottom: 35px;
            text-shadow: 0 0 12px rgba(249, 115, 22, 0.8);
            animation: pulseGlow 2s ease-in-out infinite;
        }
        .section-title {
            font-size: 2.2em;
            font-weight: 600;
            color: #1d4ed8;
            margin: 40px 0 20px;
            text-shadow: 0 0 10px rgba(29, 78, 216, 0.8);
            border-left: 6px solid #1d4ed8;
            padding-left: 18px;
            animation: slideInLeft 0.6s ease-in-out;
        }
        .content {
            font-size: 1.15em;
            color: #f3f4f6;
            line-height: 1.9;
            text-align: justify;
        }
        .highlight {
            color: #f97316;
            font-weight: bold;
        }
        .separator {
            height: 2px;
            background: linear-gradient(to right, #1d4ed8, #047857);
            margin: 20px 0;
        }
        .stButton>button {
            background: linear-gradient(45deg, #1d4ed8, #047857);
            color: #f97316;
            border-radius: 12px;
            padding: 14px 30px;
            font-weight: 600;
            font-size: 1.1em;
            border: none;
            box-shadow: 0 0 15px rgba(249, 115, 22, 0.8);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #1e40af, #065f46);
            box-shadow: 0 0 25px rgba(249, 115, 22, 1);
            transform: scale(1.1);
            color: #f3f4f6;
        }
        .stButton>button::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300%;
            height: 300%;
            background: rgba(249, 115, 22, 0.2);
            transition: all 0.6s ease;
            transform: translate(-50%, -50%) scale(0);
            border-radius: 50%;
        }
        .stButton>button:hover::after {
            transform: translate(-50%, -50%) scale(1);
        }
        .stFileUploader, .stSelectbox, .stSlider, .stCheckbox {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.9), rgba(16, 185, 129, 0.9));
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #f97316;
            color: #f3f4f6;
            transition: all 0.3s ease;
        }
        .stFileUploader:hover, .stSelectbox:hover, .stSlider:hover, .stCheckbox:hover {
            border-color: #fb923c;
            box-shadow: 0 0 8px rgba(249, 115, 22, 0.5);
        }
        .stFileUploader label, .stSelectbox label, .stSlider label, .stCheckbox label {
            color: #f97316;
            font-weight: 500;
        }
        .stPlotlyChart {
            background-color: rgba(31, 41, 55, 0.95);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .footer {
            font-size: 0.95em;
            color: #f3f4f6;
            margin-top: 50px;
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.85), rgba(16, 185, 129, 0.85));
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border: 2px solid #f97316;
            backdrop-filter: blur(10px);
        }
        .footer a {
            color: #fb923c;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        .footer a:hover {
            color: #1d4ed8;
            text-decoration: underline;
        }
        .content ul li::marker {
            color: #f97316;
        }
        /* Animations */
        @keyframes pulseGlow {
            0% { text-shadow: 0 0 10px rgba(249, 115, 22, 0.8); }
            50% { text-shadow: 0 0 20px rgba(249, 115, 22, 1); }
            100% { text-shadow: 0 0 10px rgba(249, 115, 22, 0.8); }
        }
        @keyframes slideInLeft {
            from { transform: translateX(-30px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes scaleIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

def TimeSeries_graph(ts, model='additive', decomposition_plot=True):
    fig_ts = go.Figure()

    fig_ts.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values.flatten(),
        mode='markers+lines',
        marker=dict(color='#1d4ed8', size=6, opacity=0.8),
        line=dict(color='#1d4ed8', width=2),
        name=ts.columns[0]
    ))

    fig_ts.update_layout(
        title="📈 Time Series Plot",
        xaxis_title="Date",
        yaxis_title=ts.columns[0],
        plot_bgcolor="#222",
        font=dict(color="#fff"),
        xaxis=dict(
            tickformat="%b %Y",
            showgrid=True,
            gridcolor="#444"
        ),
        yaxis=dict(showgrid=True, gridcolor="#444"),
        margin=dict(l=40, r=40, b=40, t=40),
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)

    if decomposition_plot:
        decomposition = sm.tsa.seasonal_decompose(ts, model=model)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        resid = decomposition.resid
        observed = decomposition.observed

        # Create separate plots for each decomposition component

        # Observed plot
        fig_observed = go.Figure()
        fig_observed.add_trace(go.Scatter(
            x=observed.index,
            y=observed.values.flatten(),
            mode='lines',
            name="Observed",
            line=dict(color='#1d4ed8', width=3)
        ))
        fig_observed.update_layout(
            title="📉 Observed",
            xaxis_title="Time",
            yaxis_title="Values",
            plot_bgcolor="#222",
            font=dict(color="#fff"),
            xaxis=dict(showgrid=True, gridcolor="#444"),
            yaxis=dict(showgrid=True, gridcolor="#444"),
            margin=dict(l=40, r=40, b=40, t=40),
        )
        st.plotly_chart(fig_observed, use_container_width=True)

        # Trend plot
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend.index,
            y=trend.values.flatten(),
            mode='lines',
            name="Trend",
            line=dict(color='#22c55e', width=3)
        ))
        fig_trend.update_layout(
            title="📈 Trend",
            xaxis_title="Time",
            yaxis_title="Values",
            plot_bgcolor="#222",
            font=dict(color="#fff"),
            xaxis=dict(showgrid=True, gridcolor="#444"),
            yaxis=dict(showgrid=True, gridcolor="#444"),
            margin=dict(l=40, r=40, b=40, t=40),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Seasonal plot
        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Scatter(
            x=seasonal.index,
            y=seasonal.values.flatten(),
            mode='lines',
            name="Seasonal",
            line=dict(color='#f97316', width=3)
        ))
        fig_seasonal.update_layout(
            title="🌦 Seasonal",
            xaxis_title="Time",
            yaxis_title="Values",
            plot_bgcolor="#222",
            font=dict(color="#fff"),
            xaxis=dict(showgrid=True, gridcolor="#444"),
            yaxis=dict(showgrid=True, gridcolor="#444"),
            margin=dict(l=40, r=40, b=40, t=40),
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

        # Residual plot
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=resid.index,
            y=resid.values.flatten(),
            mode='lines',
            name="Residual",
            line=dict(color='#ef4444', width=3)
        ))
        fig_resid.update_layout(
            title="🔄 Residual",
            xaxis_title="Time",
            yaxis_title="Values",
            plot_bgcolor="#222",
            font=dict(color="#fff"),
            xaxis=dict(showgrid=True, gridcolor="#444"),
            yaxis=dict(showgrid=True, gridcolor="#444"),
            margin=dict(l=40, r=40, b=40, t=40),
        )
        st.plotly_chart(fig_resid, use_container_width=True)

def adf_test(ts):
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    
    st.markdown('<div class="section-title">Augmented Dickey-Fuller Test Results</div>', unsafe_allow_html=True)
    st.dataframe(dfoutput, width=1500)

    stationary_data = None
    diff_count = 0
    while dftest[1] > 0.05:
        ts = ts.diff().dropna()
        dftest = adfuller(ts, autolag='AIC')
        diff_count += 1
        if diff_count > 5:
            st.warning("⚠️ The series could not be made stationary within 5 differencing operations.")
            return dfoutput, None
    
    if diff_count > 0:
        st.success(f"✅ The series was made stationary after {diff_count} differencing operations.")
    else:
        st.success("✅ The series is already stationary.")
    
    return dfoutput, ts

def ARIMA_model(ts, search_for_params=False, stationary_series=None):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_pdq = None
    best_model = None

    if search_for_params:
        for param in pdq:
            try:
                model = ARIMA(ts, order=param)
                results_ARIMA = model.fit()
                if results_ARIMA.aic < best_aic:
                    best_aic = results_ARIMA.aic
                    best_pdq = param
                    best_model = results_ARIMA
            except:
                continue
    
    else:
        best_pdq = (1, 1, 1)  # Default parameters if not searching
        best_model = ARIMA(ts, order=best_pdq).fit()

    st.markdown('<div class="section-title">ARIMA Model Summary</div>', unsafe_allow_html=True)
    st.write(best_model.summary())

    # Forecast plot
    forecast_steps = 10
    forecast = best_model.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq='M')[1:]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values.flatten(),
        mode='lines',
        name="Original Time Series",
        line=dict(color='#1d4ed8', width=3)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast,
        mode='lines',
        name="Forecast",
        line=dict(color='#f97316', width=3, dash='dash')
    ))
    fig_forecast.update_layout(
        title="📈 Forecast",
        xaxis_title="Time",
        yaxis_title="Values",
        plot_bgcolor="#222",
        font=dict(color="#fff"),
        xaxis=dict(showgrid=True, gridcolor="#444"),
        yaxis=dict(showgrid=True, gridcolor="#444"),
        margin=dict(l=40, r=40, b=40, t=40),
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Residual diagnostics
    residuals = best_model.resid
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=residuals.index,
        y=residuals.values.flatten(),
        mode='lines',
        name="Residuals",
        line=dict(color='#ef4444', width=3)
    ))
    fig_resid.update_layout(
        title="🔄 Residuals",
        xaxis_title="Time",
        yaxis_title="Residuals",
        plot_bgcolor="#222",
        font=dict(color="#fff"),
        xaxis=dict(showgrid=True, gridcolor="#444"),
        yaxis=dict(showgrid=True, gridcolor="#444"),
        margin=dict(l=40, r=40, b=40, t=40),
    )
    st.plotly_chart(fig_resid, use_container_width=True)

# ----------------------------------Main App----------------------------------
st.markdown('<div class="main-title">📈 Time Series Forecasting Tool 📉</div>', unsafe_allow_html=True)
st.markdown('<div class="intro-subtitle">Unlock the future with accurate forecasts. 📊✨</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload CSV File", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.markdown('<div class="section-title">Uploaded Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(data.head(), height=200)

    # Sidebar for user input
    with st.sidebar:
        st.markdown('<div class="section-title">Analysis Settings</div>', unsafe_allow_html=True)
        index_column = st.selectbox("Select Date Column", data.columns)
        if data[index_column].dtype == 'object':
            data[index_column] = pd.to_datetime(data[index_column])
        time_series_col = st.selectbox("Select Time Series Column", data.columns[data.dtypes.isin([np.number])])

    if st.button("Start Analysis 🚀"):
        try:
            if time_series_col:
                if pd.api.types.is_numeric_dtype(data[time_series_col]):
                    data.set_index(index_column, inplace=True)
                    data = data.asfreq('M')  # Assuming monthly frequency
                    ts = data[[time_series_col]]
                    TimeSeries_graph(ts, decomposition_plot=True)

                    st.session_state.first_function_done = False
                    st.session_state.second_function_done = False
                    st.session_state.third_function_done = False

                    # Step 1: Decomposition
                    if not st.session_state.first_function_done:
                        if st.button("Run Decomposition 📊"):
                            TimeSeries_graph(ts, decomposition_plot=True)
                            st.session_state.first_function_done = True
                            st.success("✅ Decomposition Completed! Move to the next step.")

                    # Step 2: ADF Test
                    if st.session_state.first_function_done and not st.session_state.second_function_done:
                        if st.button("Run Augmented Dickey-Fuller Test 🧪"):
                            adf_result, stationary_data = adf_test(data[[time_series_col]])
                            if stationary_data is not None:
                                st.write(adf_result)
                                st.session_state.second_function_done = True
                                st.success("✅ ADF Test Completed! Move to the next step.")
                            else:
                                st.warning("⚠️ ADF test failed to make the series stationary.")

                    # Step 3: ARIMA model
                    if st.session_state.second_function_done and not st.session_state.third_function_done:
                        if st.button("Run ARIMA Model ⚙️"):
                            _, stationary_data = adf_test(data[[time_series_col]])
                            ARIMA_model(data[[time_series_col]], search_for_params=True, stationary_series=stationary_data)
                            st.session_state.third_function_done = True
                            st.success("✅ ARIMA Model Completed!")
                else:
                    st.error("🚫 The selected column is not numeric. Please select a valid numeric column.")
            else:
                st.warning("⚠️ Please select a time series column.")
        except Exception as e:
            st.error(f"🚫 Error processing the selected column: {e}")
    else:
        st.info("💡 Please upload a CSV file to start analysis.")

# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank">Muhammad Umer Khan</a>. Powered by Streamlit and Statsmodel 🌐
    </div>
""", unsafe_allow_html=True)