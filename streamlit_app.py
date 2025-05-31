import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64

# Page configuration
st.set_page_config(
    page_title="📈 Easy Forecasting",
    page_icon="🔮",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin: 0.5rem 0;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
    display: block;
}
.stButton > button {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state with proper defaults
def init_session_state():
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = 0

init_session_state()

# Helper function to safely read files
def safe_read_file(uploaded_file):
    """Safely read uploaded file with proper error handling"""
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Get file info
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }
        
        # Check file size (limit to 50MB)
        if file_details["filesize"] > 50 * 1024 * 1024:
            st.error("❌ File too large. Please upload files smaller than 50MB.")
            return None
        
        # Read based on file type
        if uploaded_file.name.endswith('.csv'):
            # Try different encodings for CSV
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl' if uploaded_file.name.endswith('.xlsx') else None)
        else:
            st.error("❌ Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        # Basic validation
        if df.empty:
            st.error("❌ The file appears to be empty.")
            return None
        
        if len(df.columns) < 2:
            st.error("❌ Need at least 2 columns (date and values).")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

# Generate sample data function
def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-01', freq='MS')
    
    data = []
    base_value = 10000
    
    for i, date in enumerate(dates):
        # Seasonal effects
        seasonal = 1.3 if date.month in [11, 12] else 0.8 if date.month in [1, 2] else 1.0
        # Growth trend
        growth = 1 + (i * 0.015)
        # Random variation
        noise = 1 + np.random.normal(0, 0.08)
        
        value = int(base_value * seasonal * growth * noise)
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Sales': value,
            'Region': np.random.choice(['North', 'South', 'East', 'West'])
        })
    
    return pd.DataFrame(data)

# Auto-detect columns function
def detect_date_value_columns(df):
    """Smart detection of date and value columns"""
    date_col = None
    value_col = None
    
    # Detect date column
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year', 'period']):
            try:
                # Test if we can convert to datetime
                test_series = df[col].dropna().head(10)
                pd.to_datetime(test_series)
                date_col = col
                break
            except:
                continue
    
    # If no obvious date column found, try all columns
    if not date_col:
        for col in df.columns:
            try:
                test_series = df[col].dropna().head(10)
                pd.to_datetime(test_series)
                date_col = col
                break
            except:
                continue
    
    # Detect value column (numeric with good variation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Choose the numeric column with highest coefficient of variation
        best_cv = 0
        for col in numeric_cols:
            if col != date_col:
                try:
                    values = df[col].dropna()
                    if len(values) > 0 and values.std() > 0:
                        cv = values.std() / abs(values.mean())
                        if cv > best_cv:
                            best_cv = cv
                            value_col = col
                except:
                    continue
    
    return date_col, value_col

# Header
st.title("🔮 Easy Forecasting")
st.markdown("**Predict your business future in 3 simple steps**")

# Step 1: Upload Data
st.header("📊 Step 1: Upload Your Data")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose your CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with dates and values you want to predict",
        key=f"file_uploader_{st.session_state.upload_key}"
    )

with col2:
    st.markdown("**Need sample data?**")
    if st.button("📥 Get Sample Data"):
        sample_df = generate_sample_data()
        csv = sample_df.to_csv(index=False)
        
        # Create download link
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sample_sales_data.csv">💾 Download Sample CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# Process uploaded file
if uploaded_file is not None:
    # Only process if it's a new upload or if we don't have data yet
    if st.session_state.uploaded_df is None or st.session_state.get('last_uploaded_file') != uploaded_file.name:
        with st.spinner("Processing your file..."):
            df = safe_read_file(uploaded_file)
            
            if df is not None:
                st.session_state.uploaded_df = df
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.forecast_data = None  # Reset forecast data
                st.rerun()

# Show file info and configuration if data is loaded
if st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df
    
    st.success(f"✅ Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><span class="metric-value">{len(df):,}</span>Rows</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><span class="metric-value">{len(df.columns)}</span>Columns</div>', unsafe_allow_html=True)
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.markdown(f'<div class="metric-card"><span class="metric-value">{memory_mb:.1f}</span>MB</div>', unsafe_allow_html=True)
    
    # Auto-detect columns
    auto_date, auto_value = detect_date_value_columns(df)
    
    # Step 2: Configure
    st.header("⚙️ Step 2: Configure Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Select date column:**")
        date_col_options = list(df.columns)
        default_date_idx = date_col_options.index(auto_date) if auto_date in date_col_options else 0
        date_col = st.selectbox("Date column:", date_col_options, index=default_date_idx, label_visibility="collapsed")
        
        # Validate dates
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    date_range = f"{valid_dates.min().strftime('%b %Y')} to {valid_dates.max().strftime('%b %Y')}"
                    st.success(f"✅ Found {len(valid_dates)} valid dates: {date_range}")
                else:
                    st.error("❌ No valid dates found in this column")
            except Exception as e:
                st.error(f"❌ Error parsing dates: {str(e)}")
    
    with col2:
        st.markdown("**📈 Select value column:**")
        value_col_options = [col for col in df.columns if col != date_col]
        if auto_value and auto_value in value_col_options:
            default_value_idx = value_col_options.index(auto_value)
        else:
            default_value_idx = 0
        
        if value_col_options:
            value_col = st.selectbox("Value column:", value_col_options, index=default_value_idx, label_visibility="collapsed")
            
            # Validate values
            if value_col:
                try:
                    values = pd.to_numeric(df[value_col], errors='coerce')
                    valid_values = values.dropna()
                    if len(valid_values) > 0:
                        avg_val = valid_values.mean()
                        min_val = valid_values.min()
                        max_val = valid_values.max()
                        st.success(f"✅ Found {len(valid_values)} values: {min_val:,.0f} to {max_val:,.0f} (avg: {avg_val:,.0f})")
                    else:
                        st.error("❌ No valid numbers found in this column")
                except Exception as e:
                    st.error(f"❌ Error parsing values: {str(e)}")
        else:
            st.error("❌ No value columns available")
            value_col = None
    
    # Forecast settings
    st.markdown("**🔮 Forecast settings:**")
    col1, col2 = st.columns(2)
    with col1:
        periods = st.selectbox("Periods to forecast:", [3, 6, 12, 18, 24], index=2)
    with col2:
        freq = st.selectbox("Data frequency:", ["Daily", "Weekly", "Monthly", "Quarterly"], index=2)
    
    # Step 3: Generate Forecast
    st.header("🚀 Step 3: Generate Forecast")
    
    # Validate before allowing forecast
    can_forecast = date_col and value_col
    if not can_forecast:
        st.warning("⚠️ Please select valid date and value columns before generating forecast.")
    
    if st.button("🔮 Create Forecast", type="primary", disabled=not can_forecast):
        if can_forecast:
            with st.spinner("Creating your forecast..."):
                try:
                    # Prepare and clean data
                    forecast_df = df[[date_col, value_col]].copy()
                    forecast_df['Date'] = pd.to_datetime(forecast_df[date_col], errors='coerce')
                    forecast_df['Value'] = pd.to_numeric(forecast_df[value_col], errors='coerce')
                    
                    # Remove rows with invalid data
                    forecast_df = forecast_df.dropna().sort_values('Date')
                    
                    if len(forecast_df) < 3:
                        st.error("❌ Need at least 3 valid data points for forecasting.")
                    else:
                        # Simple forecasting algorithm
                        recent_values = forecast_df['Value'].tail(min(12, len(forecast_df)))
                        
                        # Calculate trend
                        if len(recent_values) > 1:
                            trend = (recent_values.iloc[-1] - recent_values.iloc[0]) / len(recent_values)
                        else:
                            trend = 0
                        
                        # Generate future dates
                        last_date = forecast_df['Date'].max()
                        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS"}
                        
                        try:
                            future_dates = pd.date_range(
                                start=last_date + pd.DateOffset(days=1 if freq == "Daily" else 7 if freq == "Weekly" else 30),
                                periods=periods,
                                freq=freq_map[freq]
                            )
                        except:
                            # Fallback to simple date addition
                            days_increment = 1 if freq == "Daily" else 7 if freq == "Weekly" else 30 if freq == "Monthly" else 90
                            future_dates = [last_date + timedelta(days=days_increment * (i + 1)) for i in range(periods)]
                        
                        # Create forecasts
                        base_value = recent_values.iloc[-1]
                        forecast_values = []
                        
                        np.random.seed(42)  # For consistent results
                        for i in range(periods):
                            # Trend component
                            trend_component = trend * (i + 1)
                            
                            # Seasonal component (simple sine wave)
                            seasonal = np.sin(2 * np.pi * i / 12) * base_value * 0.08
                            
                            # Random variation
                            noise = np.random.normal(0, base_value * 0.03)
                            
                            # Combine components
                            forecast = base_value + trend_component + seasonal + noise
                            
                            # Ensure positive values
                            forecast = max(0, forecast)
                            forecast_values.append(forecast)
                        
                        # Store results
                        st.session_state.forecast_data = {
                            'historical': forecast_df,
                            'forecast_dates': future_dates,
                            'forecast_values': forecast_values,
                            'value_col': value_col,
                            'date_col': date_col,
                            'periods': periods,
                            'frequency': freq
                        }
                        
                        st.success("✅ Forecast created successfully!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error creating forecast: {str(e)}")
                    st.error("Please check your data format and try again.")

# Display Results
if st.session_state.forecast_data:
    st.header("📈 Your Forecast Results")
    
    data = st.session_state.forecast_data
    
    # Calculate key metrics
    last_value = data['historical']['Value'].iloc[-1]
    forecast_avg = np.mean(data['forecast_values'])
    change = ((forecast_avg - last_value) / last_value) * 100 if last_value != 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><span class="metric-value">{last_value:,.0f}</span>Latest Value</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><span class="metric-value">{forecast_avg:,.0f}</span>Forecast Average</div>', unsafe_allow_html=True)
    with col3:
        change_color = "#00ff00" if change > 0 else "#ff4444" if change < 0 else "#ffaa00"
        st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{change_color}">{change:+.1f}%</span>Expected Change</div>', unsafe_allow_html=True)
    with col4:
        if change > 5:
            trend = "📈 Growing"
        elif change < -5:
            trend = "📉 Declining"
        else:
            trend = "📊 Stable"
        st.markdown(f'<div class="metric-card"><span class="metric-value">{trend}</span>Trend</div>', unsafe_allow_html=True)
    
    # Create and display chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data['historical']['Date'],
        y=data['historical']['Value'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#4facfe', width=3),
        marker=dict(size=6)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=data['forecast_dates'],
        y=data['forecast_values'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#00f2fe', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title=f"📊 {data['frequency']} Forecast for {data['value_col']}",
        xaxis_title="📅 Date",
        yaxis_title=f"📈 {data['value_col']}",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("💡 Key Insights")
    
    if change > 15:
        st.success("🚀 **Strong growth expected!** Consider scaling operations and preparing for increased demand.")
    elif change > 5:
        st.info("📈 **Moderate growth predicted.** Current strategy appears effective - maintain course.")
    elif change > -5:
        st.warning("📊 **Stable trend expected.** Look for optimization opportunities to drive growth.")
    elif change > -15:
        st.warning("📉 **Slight decline predicted.** Monitor key metrics and consider strategy adjustments.")
    else:
        st.error("⚠️ **Significant decline expected.** Immediate strategic review recommended.")
    
    # Additional insights
    forecast_min = min(data['forecast_values'])
    forecast_max = max(data['forecast_values'])
    volatility = (forecast_max - forecast_min) / forecast_avg * 100 if forecast_avg != 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Forecast Range:**")
        st.write(f"• **Minimum:** {forecast_min:,.0f}")
        st.write(f"• **Maximum:** {forecast_max:,.0f}")
        st.write(f"• **Volatility:** {volatility:.1f}%")
    
    with col2:
        st.markdown("**📅 Timeline:**")
        st.write(f"• **Forecast Period:** {data['periods']} {data['frequency'].lower()} periods")
        st.write(f"• **Start Date:** {data['forecast_dates'][0].strftime('%B %Y')}")
        st.write(f"• **End Date:** {data['forecast_dates'][-1].strftime('%B %Y')}")
    
    # Download section
    st.subheader("💾 Download Results")
    
    # Prepare download data
    forecast_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in data['forecast_dates']],
        data['value_col']: [round(v, 2) for v in data['forecast_values']],
        'Type': 'Forecast'
    })
    
    historical_df = pd.DataFrame({
        'Date': data['historical']['Date'].dt.strftime('%Y-%m-%d'),
        data['value_col']: data['historical']['Value'].round(2),
        'Type': 'Historical'
    })
    
    download_data = pd.concat([historical_df, forecast_df], ignore_index=True)
    csv = download_data.to_csv(index=False)
    
    st.download_button(
        "📥 Download Complete Forecast Data",
        csv,
        f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        help="Download both historical and forecast data as CSV"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🔮 <strong>Easy Forecasting</strong> - Simple business predictions made easy</p>
    <p><small>💡 Works best with regular time series data (daily, weekly, monthly)</small></p>
</div>
""", unsafe_allow_html=True)
