"""
Main User Interface - Streamlit App with Filters and Consolidated Chart
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from data_handler import DataHandler
from forecast_processor import ForecastProcessor

# Page configuration
st.set_page_config(
    page_title="Rolling Collection Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = None
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .filter-section {
        background-color: #e7f3ff;
        border: 2px solid #2196F3;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Rolling Collection Tool</h1>', unsafe_allow_html=True)
st.markdown("### 📊 Statistical Models + Launch Logic for Sales Forecasting")
st.markdown("---")


def configurar_sidebar():
    """Configure sidebar with all controls"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📁 Load File")
        uploaded_file = st.file_uploader(
            "Consolidated Excel File", 
            type=['xlsx'],
            help="Excel file with 3 sheets: Main, LogicsxMonth, and Relations",
            key="file_uploader"
        )
        
        if uploaded_file is None:
            st.markdown(
                '<div class="warning-box">📋 <strong>Required sheets:</strong><br>'
                '• Main (sales data)<br>• LogicsxMonth (logic rules)<br>• Relations (growth factors)</div>', 
                unsafe_allow_html=True
            )
        
        # Model selection
        st.subheader("🔧 Models to Execute")
        modelo_media = st.checkbox("Moving Average", True, 
                                    help="Seasonal moving average with growth factors")
        modelo_suavizacao = st.checkbox("Exponential Smoothing", True,
                                        help="Holt-Winters exponential smoothing")
        modelo_arima = st.checkbox("SARIMA", False,
                                   help="Seasonal ARIMA time series model")
        
        # Parameters
        st.subheader("⚙️ Parameters")
        
        parametros = {}
        
        if modelo_suavizacao:
            parametros['alpha'] = st.slider(
                "Alpha (Smoothing Factor)", 
                0.1, 0.9, 0.3, 0.1,
                help="Smoothing factor (0.1 = smoother, 0.9 = more reactive)"
            )
        
        if modelo_arima:
            st.write("**ARIMA Parameters (p,d,q):**")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.selectbox("p (AR)", [0, 1, 2, 3], 1)
            with col2:
                d = st.selectbox("d (I)", [0, 1, 2], 1)
            with col3:
                q = st.selectbox("q (MA)", [0, 1, 2, 3], 1)
            parametros['arima_params'] = (p, d, q)
        
        st.markdown("---")       
        
        
        return uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros


def mostrar_info_carga(data_handler):
    """Display information about loaded data"""
    
    dataframes = data_handler.obtener_dataframes()
    forecast_start = data_handler.obtener_fecha_inicio()
    
    st.markdown(
        '<div class="success-box">✅ <strong>File loaded successfully!</strong></div>', 
        unsafe_allow_html=True
    )
    
    # Display forecast start date
    if forecast_start:
        st.info(f"📅 **Forecast Start Date:** {forecast_start.strftime('%B %d, %Y')}")
    
    # DataFrame metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        df = dataframes['main']
        st.metric("📊 Main (Products)", f"{df.shape[0]}")
    with col2:
        df = dataframes['logics']
    with col3:
        df = dataframes['relations']
    
    # Data preview
    with st.expander("👀 Data Preview"):
        tab1, tab2, tab3 = st.tabs(["📊 Main", "⚙️ LogicsxMonth", "🔗 Relations"])
        
        with tab1:
            df = dataframes['main']
            st.write(f"**Columns:** {df.shape[1]} | **Products:** {df.shape[0]}")
            display_df = df.iloc[:10, :15] if df.shape[1] > 15 else df.head(10)
            st.dataframe(display_df, use_container_width=True)
        
        with tab2:
            df = dataframes['logics']
            st.write(f"**Logic Rules:** {df.shape[0]}")
            st.dataframe(df.head(15), use_container_width=True)
        
        with tab3:
            df = dataframes['relations']
            st.write(f"**Customers with Growth Factors:** {df.shape[0]}")
            st.dataframe(df.head(15), use_container_width=True)


def extraer_valores_unicos(df_resultado, customer_col, model_col):
    """Extract unique values for filters"""
    
    customers = []
    models = []
    
    if customer_col and customer_col in df_resultado.columns:
        customers = sorted(df_resultado[customer_col].dropna().unique().tolist())
    
    if model_col and model_col in df_resultado.columns:
        models = sorted(df_resultado[model_col].dropna().unique().tolist())
    
    return customers, models


def aplicar_filtros(df: pd.DataFrame, customer_col: str, model_col: str, 
                   selected_customers: list, selected_models: list) -> pd.DataFrame:
    """Apply filters to dataframe"""
    
    df_filtered = df.copy()
    
    # Filter by customers
    if selected_customers and len(selected_customers) > 0 and customer_col and customer_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[customer_col].isin(selected_customers)]
    
    # Filter by models
    if selected_models and len(selected_models) > 0 and model_col and model_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[model_col].isin(selected_models)]
    
    return df_filtered


def crear_grafico_consolidado(resultados: dict, df_filtered_dict: dict):
    """
    Create consolidated interactive chart with all forecast models
    Users can show/hide individual series by clicking on legend
    
    Args:
        resultados: Dictionary with all model results
        df_filtered_dict: Dictionary with filtered dataframes for each model
    """
    
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ Plotly not available for charts. Please install: pip install plotly")
        return
    
    fig = go.Figure()
    
    # Colors for each model
    colors = {
        'media_movil': '#FF6B6B',
        'suavizacao_exponencial': '#4ECDC4',
        'arima': '#95E1D3'
    }
    
    # Names for each model
    names = {
        'media_movil': 'Moving Average',
        'suavizacao_exponencial': 'Exponential Smoothing',
        'arima': 'SARIMA'
    }
    
    has_data = False
    
    # Add trace for each model
    for modelo_key, df_filtered in df_filtered_dict.items():
        
        if df_filtered.empty:
            continue
        
        # Get date columns (forecast columns)
        date_cols = [col for col in df_filtered.columns if isinstance(col, datetime)]
        
        if not date_cols:
            continue
        
        # Calculate total by month
        forecast_summary = df_filtered[date_cols].sum()
        
        if len(forecast_summary) == 0 or forecast_summary.sum() == 0:
            continue
        
        has_data = True
        
        # Add trace with show/hide capability
        fig.add_trace(go.Scatter(
            x=[d.strftime('%Y-%m') for d in forecast_summary.index],
            y=forecast_summary.values,
            mode='lines+markers',
            name=names.get(modelo_key, modelo_key),
            line=dict(color=colors.get(modelo_key, '#999999'), width=3),
            marker=dict(size=8),
            visible=True,  # All visible by default
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Units: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    if not has_data:
        st.warning("⚠️ No data available to display in chart. Try adjusting filters or check if models generated results.")
        return
    
    # Update layout with better styling
    fig.update_layout(
        title={
            'text': "📊 Consolidated Forecast Comparison - All Models",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title="Month",
        yaxis_title="Total Units Forecasted",
        hovermode='x unified',
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Gray",
            borderwidth=2,
            font=dict(size=12)
        ),
        plot_bgcolor='rgba(240, 242, 246, 0.5)',
        xaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='LightGray'
        ),
        font=dict(family="Arial", size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    

def mostrar_resultados(resultados):
    """Display forecasting results with global filters and consolidated chart"""
    
    st.header("📊 Forecast Results")
    
    # General summary
    total_cells = sum([r.get('celulas_actualizadas', 0) for r in resultados.values()])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Cells Updated", f"{total_cells:,}")
    with col2:
        st.metric("🔢 Models Executed", len(resultados))
    with col3:
        if resultados:
            first_result = list(resultados.values())[0]
            forecast_months = first_result.get('metadata', {}).get('forecast_months', 18)
            st.metric("📅 Forecast Horizon", f"{forecast_months} months")
    with col4:
        # Total execution time
        total_time = sum([r.get('metadata', {}).get('execution_time', 0) for r in resultados.values()])
        st.metric("⏱️ Total Time", f"{total_time:.1f}s")
    
    # Get first dataframe to extract column names for filters
    first_df = None
    customer_col = None
    model_col = None
    
    for resultado in resultados.values():
        if 'dataframe' in resultado and not resultado['dataframe'].empty:
            first_df = resultado['dataframe']
            break
    
    if first_df is not None:
        for col in first_df.columns:
            col_str = str(col).lower()
            # if 'customer' in col_str or 'cliente' in col_str:
            if 'id' in col_str and 'customer' in col_str:
                customer_col = col
            # if ('code' in col_str or 'modelo' in col_str) and 'product' in col_str:
            if 'product' in col_str and 'model' in col_str:
                model_col = col
    
    # Extract unique values from first dataframe
    customers_list = []
    models_list = []
    
    if first_df is not None:
        customers_list, models_list = extraer_valores_unicos(first_df, customer_col, model_col)
    
    # ==========================================
    # GLOBAL FILTER SECTION
    # ==========================================
    st.markdown("---")
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.subheader("🔍 Global Filters")
    st.markdown("*Apply filters to all models and update the consolidated chart*")
    
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
    
    with filter_col1:
        if len(customers_list) > 0:
            selected_customers = st.multiselect(
                "📍 Filter by Customer ID",
                options=customers_list,
                default=[],
                key="global_customer_filter",
                help="Select one or more customers (empty = show all customers)"
            )
        else:
            selected_customers = []
            st.info("ℹ️ No customers found in results")
    
    with filter_col2:
        if len(models_list) > 0:
            selected_models = st.multiselect(
                "🏷️ Filter by Product Model",
                options=models_list,
                default=[],
                key="global_model_filter",
                help="Select one or more product models (empty = show all models)"
            )
        else:
            selected_models = []
            st.info("ℹ️ No product models found in results")
    
    with filter_col3:
        st.write("")
        st.write("")
        if st.button("🔄 Clear All Filters", key="clear_global", use_container_width=True):
            # Clear filter values by resetting keys
            st.session_state.global_customer_filter = []
            st.session_state.global_model_filter = []
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters to all dataframes
    df_filtered_dict = {}
    total_products_filtered = 0
    total_products_original = 0
    
    for modelo_key, resultado in resultados.items():
        if 'dataframe' in resultado and not resultado['dataframe'].empty:
            df_resultado = resultado['dataframe']
            df_filtered = aplicar_filtros(df_resultado, customer_col, model_col, 
                                         selected_customers, selected_models)
            df_filtered_dict[modelo_key] = df_filtered
            
            total_products_original = len(df_resultado)
            total_products_filtered = len(df_filtered)
    
    # Show filter summary
    if len(selected_customers) > 0 or len(selected_models) > 0:
        filter_info = []
        if len(selected_customers) > 0:
            filter_info.append(f"**{len(selected_customers)}** customer(s)")
        if len(selected_models) > 0:
            filter_info.append(f"**{len(selected_models)}** product model(s)")
        
        st.success(f"✅ **Active Filters:** {' + '.join(filter_info)} | Showing **{total_products_filtered:,}** of **{total_products_original:,}** products")
    else:
        st.info(f"ℹ️ **No filters applied** - Showing all **{total_products_original:,}** products")
    
    # ==========================================
    # CONSOLIDATED CHART
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Consolidated Forecast Chart - All Models")
    
    if len(df_filtered_dict) > 0:
        crear_grafico_consolidado(resultados, df_filtered_dict)
    else:
        st.warning("⚠️ No data available for chart")
    
    # ==========================================
    # INDIVIDUAL MODEL TABS
    # ==========================================
    st.markdown("---")
    st.subheader("📋 Detailed Results by Model")
    
    tab_names = []
    for modelo in resultados.keys():
        if modelo == 'media_movil':
            tab_names.append("Moving Average")
        elif modelo == 'suavizacao_exponencial':
            tab_names.append("Exponential Smoothing")
        elif modelo == 'arima':
            tab_names.append("SARIMA")
    
    if not tab_names:
        st.warning("No results generated.")
        return
    
    tabs = st.tabs(tab_names)
    
    for i, (modelo_key, resultado) in enumerate(resultados.items()):
        with tabs[i]:
            
            if 'error' in resultado.get('metadata', {}):
                st.error("❌ Error processing this model")
                continue
            
            if 'dataframe' not in resultado or resultado['dataframe'].empty:
                st.warning("No data available for this model")
                continue
            
            df_resultado = resultado['dataframe']
            df_filtered = df_filtered_dict.get(modelo_key, df_resultado)
            
            # Results table
            st.subheader("📊 Results Table")
            
            if not df_filtered.empty:
                # Show filtered results
                if df_filtered.shape[1] > 20:
                    cols_to_show = list(df_filtered.columns[:8]) + list(df_filtered.columns[-10:])
                    df_display = df_filtered[cols_to_show]
                else:
                    df_display = df_filtered
                
                st.dataframe(df_display.head(100), use_container_width=True)
                
                # Summary info
                st.info(f"📊 Showing **{len(df_filtered):,}** products (Total: **{len(df_resultado):,}**)")
            else:
                st.warning("⚠️ No data matches the selected filters")
            
            # Export section
            st.markdown("---")
            st.subheader("💾 Export Data")
            
            df_to_export = df_filtered if not df_filtered.empty else df_resultado
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                # CSV download
                csv = df_to_export.to_csv(index=False)
                suffix = "_filtered" if len(df_filtered) < len(df_resultado) else "_full"
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"forecast_{modelo_key}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_download2:
                # Excel download
                try:
                    from io import BytesIO
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_to_export.to_excel(writer, index=False, sheet_name='Forecast')
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"forecast_{modelo_key}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error creating Excel file: {str(e)}")


def mostrar_pantalla_bienvenida():
    """Display welcome screen when no file is loaded"""
    
    st.info("📤 **Please load the consolidated Excel file to begin.**")
    
    st.markdown("### 📋 Required File Structure:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Sheet: Main**
        - Row 2: Column headers (first day of each month)
        - Cell B2: Forecast start date
        - Columns A-H: Product characteristics
        - Columns I+: Monthly sales history
        - Data starts from row 3
        """)
    
    # with col2:
    #     st.markdown("""
    #     **⚙️ Sheet: LogicsxMonth**
    #     - Row 2: Column headers
    #     - Column C: Class
    #     - Column D: Month (dd/mm/yyyy)
    #     - Column E: Calculation Base
    #     - Column G: P2P Model
    #     - Column H: Launch Month (XF)
    #     - Data starts from row 3
    #     """)
    
    # with col3:
    #     st.markdown("""
    #     **🔗 Sheet: Relations**
    #     - Row 8: Year headers
    #     - Column A: ID Customer (from row 9)
    #     - Columns B+: Growth factors by year
    #     - Factor values (e.g., 1.05 = 5% growth)
    #     """)
    
    st.markdown("---")
    
        
    


def main():
    """Main application function"""
    
    # Configure sidebar and get parameters
    uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros = configurar_sidebar()
    
    # If no file, show welcome screen
    if not uploaded_file:
        mostrar_pantalla_bienvenida()
        # Reset session state when no file
        st.session_state.resultados = None
        st.session_state.data_handler = None
        st.session_state.file_loaded = False
        return
    
    # Load data only if not already loaded or if different file
    if not st.session_state.file_loaded or st.session_state.data_handler is None:
        with st.spinner("🔄 Loading consolidated file..."):
            data_handler = DataHandler(uploaded_file)
            
            if not data_handler.cargar_archivo():
                for error in data_handler.obtener_errores():
                    st.error(error)
                return
            
            # Store in session state
            st.session_state.data_handler = data_handler
            st.session_state.file_loaded = True
    
    # Use data_handler from session state
    data_handler = st.session_state.data_handler
    
    # Display load information
    mostrar_info_carga(data_handler)
    
    # Validate at least one model is selected
    if not any([modelo_media, modelo_suavizacao, modelo_arima]):
        st.warning("⚠️ Please select at least one model to execute.")
        return
    
    # Execute forecast button
    if st.button("Execute Forecast", type="primary", use_container_width=True):
        
        with st.spinner("Processing models... This may take a few minutes."):
            try:
                # Create processor
                processor = ForecastProcessor(
                    data_handler.obtener_dataframes(),
                    data_handler.obtener_fecha_inicio(),
                    parametros
                )
                
                # Execute selected models
                modelos_ejecutar = {
                    'media_movil': modelo_media,
                    'suavizacao_exponencial': modelo_suavizacao,
                    'arima': modelo_arima
                }
                
                resultados = processor.ejecutar_forecast(modelos_ejecutar)
                
                if resultados:
                    # Store results in session state
                    st.session_state.resultados = resultados
                else:
                    st.error("❌ Could not generate results. Please verify input data.")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
    
    # Display results if they exist in session state
    if st.session_state.resultados is not None:
        mostrar_resultados(st.session_state.resultados)


if __name__ == "__main__":
    main()