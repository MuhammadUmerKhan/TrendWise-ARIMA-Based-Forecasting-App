import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import itertools
from sklearn.metrics import mean_squared_error

def TimeSeries_graph(ts, model='additive', decomposition_plot=True):
    fig_ts = go.Figure()

    fig_ts.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values.flatten(),
        mode='markers+lines',
        marker=dict(color='#1f78b4', size=6, opacity=0.8),
        line=dict(color='#1f78b4', width=2),
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
            line=dict(color='#1f78b4', width=3)
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
            line=dict(color='#ff7f0e', width=3)
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

        # Residual plot
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=resid.index,
            y=resid.values.flatten(),
            mode='lines',
            name="Residual",
            line=dict(color='#d62728', width=3)
        ))
        fig_resid.update_layout(
            title="🔍 Residual",
            xaxis_title="Time",
            yaxis_title="Values",
            plot_bgcolor="#222",
            font=dict(color="#fff"),
            xaxis=dict(showgrid=True, gridcolor="#444"),
            yaxis=dict(showgrid=True, gridcolor="#444"),
            margin=dict(l=40, r=40, b=40, t=40),
        )
        st.plotly_chart(fig_resid, use_container_width=True)

def adf_test(series, max_diff=10, alpha=0.05):
    result_dict = {}
    stationary_data = None

    for i in range(max_diff + 1):
        if i > 0:
            series = series.diff().dropna()

        result = adfuller(series)
        p_value = result[1]

        result_dict[f'Differencing Level {i}'] = {
            'ADF Statistic': result[0],
            'p-value': f"{p_value:.4e}",
            'Lags Used': result[2],
            'Number of Observations': result[3],
            'Stationary': "Yes" if p_value <= alpha else "No"
        }

        if p_value <= alpha:
            stationary_data = series  # Store stationary data when test passes
            TimeSeries_graph(series, decomposition_plot=False)
            break
        else:
            stationary_data = series

    result_df = pd.DataFrame(result_dict).T
    result_df['p-value'] = result_df['p-value'].astype(str)
    return result_df, stationary_data

def ARIMA_model(original_series, search_for_params = False, stationary_series = None):
    if search_for_params:
        #  Train-Test Split
        train_len = int(len(stationary_series) * 0.8)
        train = stationary_series[:train_len]
        test = stationary_series[train_len:]

        mse_lst, rmse_lst, order_lst = [], [], []
        pdq_combinations = list(itertools.product(range(0, 8), range(0, 8), range(0, 2)))

        print("Please wait, it may take a few moments...")
        print("Searching for the best ARIMA model parameters...")

        for pdq in pdq_combinations:
            try:
                model = ARIMA(train, order=pdq).fit()
                pred = model.predict(start=len(train), end=len(stationary_series) - 1)

                # Metrics Calculations
                mse = mean_squared_error(test, pred)
                rmse = np.sqrt(mse)

                # Appending to Lists
                order_lst.append(pdq)
                mse_lst.append(mse)
                rmse_lst.append(rmse)

            except Exception as e:
                print(f"Error with ARIMA order {pdq}: {e}")
                continue

        # Create DataFrame for Results
        result_df = pd.DataFrame({
            'Mean Squared Error': mse_lst,
            'Root Mean Squared Error': rmse_lst,
        }, index=order_lst).sort_values(ascending=True, by='Mean Squared Error').head(1)

        best_params = result_df.index[0]
        
        # Final Model Training with Best Parameters
        final_model = ARIMA(original_series, order=best_params).fit()
        prediction = final_model.predict(start=1600, end=len(original_series) + 30)
    else:
        final_model = ARIMA(original_series, order=(6, 0, 0)).fit()
        prediction = final_model.predict(start=1600, end=len(original_series) + 30)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=original_series.index,
        y=original_series.values.flatten(),
        mode='lines',
        name="Original Data",
        line=dict(color='#2ca02c', width=2),
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=pd.date_range(start=original_series.index[-1], periods=len(prediction), freq='D'),
        y=prediction,
        mode='lines+markers',
        name="Prediction",
        line=dict(color='#d62728', width=2, dash='dash'),
        marker=dict(size=6, color='#d62728', opacity=0.8),
    ))

    fig.update_layout(
        title="🔮 ARIMA Predictions",
        xaxis_title="Time",
        yaxis_title="Values",
        plot_bgcolor="#222",
        font=dict(color="#fff"),
        xaxis=dict(showgrid=True, gridcolor="#444", tickformat="%b %Y"),
        yaxis=dict(showgrid=True, gridcolor="#444"),
        margin=dict(l=40, r=40, b=40, t=40),
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)


# Set the page configuration for a more appealing look
st.set_page_config(
    page_title="📉 ARIMA Time Series Forecasting",
    # page_icon="📉",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# Customizing the theme and font
st.markdown(
    """
    <style>
        body {
            background-color: #222222;
            font-family: 'Arial', sans-serif;
            color: #fff;
        }
        h1, h2, h3 {
            color: #ff7f0e;
        }
        .stButton button {
            background-color: #ff7f0e;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #e06000;
        }
        .stSelectbox, .stTextInput, .stSelectSlider {
            background-color: #444;
            border-color: #444;
            color: white;
        }
        .stSelectbox:hover, .stTextInput:hover {
            border-color: #ff7f0e;
        }
        /* Main Title */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Section Titles */
        .section-title {
            font-size: 1.8em;
            color: #3498DB;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        /* Section Content */
        .section-content{
            text-align: center;
        }
        /* Home Page Content */
        .intro-title {
            font-size: 2.5em;
            color: #2C3E50;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #34495E;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            color: #2E86C1;
            font-weight: bold;
        }
        /* Recommendation Titles and Descriptions */
        .recommendation-title {
            font-size: 22px;
            color: #2980B9;
        }
        .recommendation-desc {
            font-size: 16px;
            color: #7F8C8D;
        }
        /* Separator Line */
        .separator {
            margin-top: 10px;
            margin-bottom: 10px;
            border-top: 1px solid #BDC3C7;
        }
        /* Footer */
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="intro-title">📉 ARIMA Time Series Forecasting 📉</div>', unsafe_allow_html=True)
st.markdown('<div class="intro-subtitle">This tool allows you to visualize time series data, check stationarity with the Augmented Dickey-Fuller test 📉</div>', unsafe_allow_html=True)

# Tabs for better organization
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📋 Yahoo Stock Price Forecasting", "📂 Upload your dataset"])
# Main Code

with tab1:
    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I'm Muhammad Umer Khan, an aspiring Data Scientist with a passion for Natural Language Processing (NLP). Currently pursuing my Bachelor's in Computer Science, I have hands-on experience with projects in data science, data scraping, and building intelligent recommendation systems.
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">📢 <strong>Note:</strong></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            In the <strong style="color: #FF5733;">Upload Your Dataset</strong> section, please upload clean data.
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔍 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project focuses on developing an <span class="highlight">Intelligent Stock Price Prediction System</span> using time series techniques. Here's a breakdown of the project:
            <ul>
                <li>📊 <span class="highlight">Data Collection</span>: Gathered <a href="https://github.com/MuhammadUmerKhan/Yahoo-Stock/blob/main/Data/Yahoo%20Stock%20Data%20(correct%20format).csv" target="_blank" style="color: #2980B9;"><span class="highlight">historical stock.</span></a></li>
                <li>🔍 <span class="highlight">Feature Engineering</span>: Extracted key features like Date, High, Low, Open, Close, Volume, etc.</li>
                <li>📈 <span class="highlight">Model Development</span>: Built ARIMA models and used hypothesis testing like ADF.</li>
                <li>🔬 <span class="highlight">Evaluation</span>: Evaluated using metrics like RMSE and MAPE.</li>
                <li>🌐 <span class="highlight">Deployment</span>: Streamlit for an intuitive web interface.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies Used</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            - <span class="highlight">Languages & Libraries</span>: Python, Pandas, NumPy, Statsmodels, Plotly, and Streamlit<br>
            - <span class="highlight">Deployment</span>: Streamlit for creating a user-friendly web interface.
        </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">📋 Forecast Stock Prices</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Use this tab to upload your dataset or select default data to visualize stock prices and forecast future trends.
        </div>
    """, unsafe_allow_html=True)
    data = pd.read_csv("./Data/Yahoo Stock Data (correct format).csv")

    if data is not None:
        columns_with_placeholder = ["Please Select"] + list(data.columns)

        # Index column selection
        index_column = st.selectbox("Select the Index Column 📅", columns_with_placeholder)
        if index_column != "Please Select":
            if np.issubdtype(data[index_column].dtype, np.datetime64) or data[index_column].dtype == object:
                try:
                    # Convert to datetime
                    temp_column = pd.to_datetime(data['Date'], format = '%Y-%m-%d')
                    data[index_column] = temp_column
                    data.set_index(index_column, inplace=True)  
                    
                    # data.index = data.index.strftime('%Y-%m-%d')                 
                    st.success("✅ Index column successfully set as datetime with proper frequency.")

                    # Time series column selection
                    time_series_columns = ["Please Select"] + list(data.columns)
                    time_series_col = st.selectbox("Select the Time Series Column (Values)", time_series_columns)

                    if time_series_col != "Please Select":
                        # Validate numeric column
                        if np.issubdtype(data[time_series_col].dtype, np.number):
                            # Session state management
                            if 'first_function_done' not in st.session_state:
                                st.session_state.first_function_done = False
                            if 'second_function_done' not in st.session_state:
                                st.session_state.second_function_done = False
                            if 'third_function_done' not in st.session_state:
                                st.session_state.third_function_done = False

                            st.table(data[[time_series_col]].sample(5))
                            # Step 1: Time series graph
                            if st.button("Generate Time Series Graph 📈"):
                                if not st.session_state.first_function_done:
                                    TimeSeries_graph(data[[time_series_col]])
                                    st.session_state.first_function_done = True
                                    st.success("✅ Time Series Graph Generated! Move to the next step.")
                                
                            # Step 2: Augmented Dickey-Fuller test
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
                                    ARIMA_model(data[[time_series_col]])
                                    st.session_state.third_function_done = True
                                    st.success("✅ ARIMA Model Completed!")
                        else:
                            st.error("🚫 The selected column is not numeric. Please select a valid numeric column.")
                    else:
                        st.warning("⚠️ Please select a time series column.")
                except Exception as e:
                    st.error("🚫 Error processing the selected column: {e}")
            else:
                st.warning("⚠️ The selected column '{index_column}' appears to be of type {data[index_column].dtype}. It's not recommended to convert it to datetime.")

        else:
            st.warning("⚠️ Please select an index column.")

with tab3:
    
    st.markdown('<div class="section-title"><strong style="color: #FF5733;">📤 Upload your Cleaned CSV file here!</strong>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"])

     
      
    if uploaded_file is not None:
        st.success("✅ ✅ File uploaded successfully! Processing data...")
        data = pd.read_csv(uploaded_file)
        columns_with_placeholder = ["Please Select"] + list(data.columns)

        # Index column selection
        index_column = st.selectbox("Select the Index Column 📅", columns_with_placeholder, key="index_column_key")
        if index_column != "Please Select":
            if np.issubdtype(data[index_column].dtype, np.datetime64) or data[index_column].dtype == object:
                try:
                    # Convert to datetime
                    temp_column = pd.to_datetime(data[index_column])
                    data[index_column] = temp_column
                    data.set_index(index_column, inplace=True)                   
                    st.success("✅ Index column successfully set as datetime with proper frequency.")

                    # Time series column selection
                    time_series_columns = ["Please Select"] + list(data.columns)
                    time_series_col = st.selectbox("Select the Time Series Column (Values)", time_series_columns)

                    if time_series_col != "Please Select":
                        # Validate numeric column
                        if np.issubdtype(data[time_series_col].dtype, np.number):
                            # Session state management
                            if 'first_function_done' not in st.session_state:
                                st.session_state.first_function_done = False
                            if 'second_function_done' not in st.session_state:
                                st.session_state.second_function_done = False
                            if 'third_function_done' not in st.session_state:
                                st.session_state.third_function_done = False

                            # Step 1: Time series graph
                            if st.button("Generate Time Series Graph 📈"):
                                if not st.session_state.first_function_done:
                                    TimeSeries_graph(data[[time_series_col]])
                                    st.session_state.first_function_done = True
                                    st.success("✅ Time Series Graph Generated! Move to the next step.")
                                
                            # Step 2: Augmented Dickey-Fuller test
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
                    st.error("🚫 Error processing the selected column: {e}")
            else:
                st.warning("⚠️ The selected column '{index_column}' appears to be of type {data[index_column].dtype}. It's not recommended to convert it to datetime.")

        else:
            st.info("💡 Please upload a CSV file to start analysis.")

# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by Streamlit and Statsmodel 🌐
    </div>
""", unsafe_allow_html=True)