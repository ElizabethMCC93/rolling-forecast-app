# """
# Main User Interface - Streamlit App (English version with Filters)
# """
# import streamlit as st
# import pandas as pd
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# try:
#     import plotly.graph_objects as go
#     PLOTLY_AVAILABLE = True
# except ImportError:
#     PLOTLY_AVAILABLE = False

# from data_handler import DataHandler
# from forecast_processor import ForecastProcessor

# # Page configuration
# st.set_page_config(
#     page_title="Rolling Forecast Tool",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #FF6B6B;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#     }
#     .success-box {
#         background-color: #d4edda;
#         border: 1px solid #c3e6cb;
#         color: #155724;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 1rem 0;
#     }
#     .warning-box {
#         background-color: #fff3cd;
#         border: 1px solid #ffeaa7;
#         color: #856404;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 1rem 0;
#     }
#     .info-box {
#         background-color: #d1ecf1;
#         border: 1px solid #bee5eb;
#         color: #0c5460;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 1rem 0;
#     }
#     .filter-section {
#         background-color: #e7f3ff;
#         border: 2px solid #2196F3;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main header
# st.markdown('<h1 class="main-header">🎯 Rolling Forecast Tool</h1>', unsafe_allow_html=True)
# st.markdown("### 📊 Statistical Models + Launch Logic for Sales Forecasting")
# st.markdown("---")


# def configurar_sidebar():
#     """Configure sidebar with all controls"""
    
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # File upload
#         st.subheader("📁 Load File")
#         uploaded_file = st.file_uploader(
#             "Consolidated Excel File", 
#             type=['xlsx'],
#             help="Excel file with 3 sheets: Main, LogicsxMonth, and Relations"
#         )
        
#         if uploaded_file is None:
#             st.markdown(
#                 '<div class="warning-box">📋 <strong>Required sheets:</strong><br>'
#                 '• Main (sales data)<br>• LogicsxMonth (logic rules)<br>• Relations (growth factors)</div>', 
#                 unsafe_allow_html=True
#             )
        
#         # Model selection
#         st.subheader("🔧 Models to Execute")
#         modelo_media = st.checkbox("📈 Moving Average", True, 
#                                     help="Seasonal moving average with growth factors")
#         modelo_suavizacao = st.checkbox("📊 Exponential Smoothing", True,
#                                         help="Holt-Winters exponential smoothing")
#         modelo_arima = st.checkbox("🔬 SARIMA", True,
#                                    help="Seasonal ARIMA time series model")
        
#         # Parameters
#         st.subheader("⚙️ Parameters")
        
#         parametros = {}
        
#         if modelo_suavizacao:
#             parametros['alpha'] = st.slider(
#                 "Alpha (Smoothing Factor)", 
#                 0.1, 0.9, 0.3, 0.1,
#                 help="Smoothing factor (0.1 = smoother, 0.9 = more reactive)"
#             )
        
#         if modelo_arima:
#             st.write("**ARIMA Parameters (p,d,q):**")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 p = st.selectbox("p (AR)", [0, 1, 2, 3], 1)
#             with col2:
#                 d = st.selectbox("d (I)", [0, 1, 2], 1)
#             with col3:
#                 q = st.selectbox("q (MA)", [0, 1, 2, 3], 1)
#             parametros['arima_params'] = (p, d, q)
        
#         st.markdown("---")
        
#         # Information box
#         st.markdown(
#             '<div class="info-box">'
#             '<strong>💡 How it works:</strong><br>'
#             '• 18-month forecast from start date (B2)<br>'
#             '• Applies calculation logic per class/month<br>'
#             '• Growth factors from Relations sheet<br>'
#             '• Supports P2P (previous model) logic'
#             '</div>', 
#             unsafe_allow_html=True
#         )
        
#         return uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros


# def mostrar_info_carga(data_handler):
#     """Display information about loaded data"""
    
#     dataframes = data_handler.obtener_dataframes()
#     forecast_start = data_handler.obtener_fecha_inicio()
    
#     st.markdown(
#         '<div class="success-box">✅ <strong>File loaded successfully!</strong></div>', 
#         unsafe_allow_html=True
#     )
    
#     # Display forecast start date
#     if forecast_start:
#         st.info(f"📅 **Forecast Start Date:** {forecast_start.strftime('%B %d, %Y')}")
    
#     # DataFrame metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         df = dataframes['main']
#         st.metric("📊 Main (Products)", f"{df.shape[0]} rows × {df.shape[1]} cols")
#     with col2:
#         df = dataframes['logics']
#         st.metric("⚙️ LogicsxMonth", f"{df.shape[0]} rules")
#     with col3:
#         df = dataframes['relations']
#         st.metric("🔗 Relations", f"{df.shape[0]} customers")
    
#     # Data preview
#     with st.expander("👀 Data Preview"):
#         tab1, tab2, tab3 = st.tabs(["📊 Main", "⚙️ LogicsxMonth", "🔗 Relations"])
        
#         with tab1:
#             df = dataframes['main']
#             st.write(f"**Columns:** {df.shape[1]} | **Products:** {df.shape[0]}")
#             display_df = df.iloc[:10, :15] if df.shape[1] > 15 else df.head(10)
#             st.dataframe(display_df, use_container_width=True)
        
#         with tab2:
#             df = dataframes['logics']
#             st.write(f"**Logic Rules:** {df.shape[0]}")
#             st.dataframe(df.head(15), use_container_width=True)
        
#         with tab3:
#             df = dataframes['relations']
#             st.write(f"**Customers with Growth Factors:** {df.shape[0]}")
#             st.dataframe(df.head(15), use_container_width=True)


# def extraer_valores_unicos(df_resultado, customer_col, model_col):
#     """Extract unique values for filters"""
    
#     customers = []
#     models = []
    
#     if customer_col and customer_col in df_resultado.columns:
#         customers = sorted(df_resultado[customer_col].dropna().unique().tolist())
    
#     if model_col and model_col in df_resultado.columns:
#         models = sorted(df_resultado[model_col].dropna().unique().tolist())
    
#     return customers, models


# def aplicar_filtros(df: pd.DataFrame, customer_col: str, model_col: str, 
#                    selected_customers: list, selected_models: list) -> pd.DataFrame:
#     """Apply filters to dataframe"""
    
#     df_filtered = df.copy()
    
#     # Filter by customers
#     if selected_customers and len(selected_customers) > 0 and customer_col:
#         df_filtered = df_filtered[df_filtered[customer_col].isin(selected_customers)]
    
#     # Filter by models
#     if selected_models and len(selected_models) > 0 and model_col:
#         df_filtered = df_filtered[df_filtered[model_col].isin(selected_models)]
    
#     return df_filtered


# def mostrar_resultados(resultados):
#     """Display forecasting results with filters"""
    
#     st.header("📊 Forecast Results")
    
#     # General summary
#     total_cells = sum([r.get('celulas_actualizadas', 0) for r in resultados.values()])
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("📝 Total Cells Updated", f"{total_cells:,}")
#     with col2:
#         st.metric("🔢 Models Executed", len(resultados))
#     with col3:
#         if resultados:
#             first_result = list(resultados.values())[0]
#             forecast_months = first_result.get('metadata', {}).get('forecast_months', 18)
#             st.metric("📅 Forecast Horizon", f"{forecast_months} months")
    
#     # Tabs for each model
#     tab_names = []
#     for modelo in resultados.keys():
#         if modelo == 'media_movil':
#             tab_names.append("📈 Moving Average")
#         elif modelo == 'suavizacao_exponencial':
#             tab_names.append("📊 Exponential Smoothing")
#         elif modelo == 'arima':
#             tab_names.append("🔬 SARIMA")
    
#     if not tab_names:
#         st.warning("No results generated.")
#         return
    
#     tabs = st.tabs(tab_names)
    
#     for i, (modelo_key, resultado) in enumerate(resultados.items()):
#         with tabs[i]:
            
#             if 'error' in resultado.get('metadata', {}):
#                 st.error("❌ Error processing this model")
#                 continue
            
#             if 'dataframe' not in resultado or resultado['dataframe'].empty:
#                 st.warning("No data available for this model")
#                 continue
            
#             df_resultado = resultado['dataframe']
            
#             # Detect column names for filters
#             customer_col = None
#             model_col = None
            
#             for col in df_resultado.columns:
#                 col_str = str(col).lower()
#                 if 'customer' in col_str or 'cliente' in col_str:
#                     customer_col = col
#                 if 'model' in col_str or 'modelo' in col_str:
#                     model_col = col
            
#             # Extract unique values
#             customers_list, models_list = extraer_valores_unicos(df_resultado, customer_col, model_col)
            
#             # ==========================================
#             # FILTER SECTION
#             # ==========================================
#             st.markdown('<div class="filter-section">', unsafe_allow_html=True)
#             st.subheader("🔍 Filters")
            
#             filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
            
#             with filter_col1:
#                 if len(customers_list) > 0:
#                     selected_customers = st.multiselect(
#                         "Filter by Customer",
#                         options=customers_list,
#                         default=[],
#                         key=f"customer_filter_{modelo_key}",
#                         help="Select one or more customers (empty = all)"
#                     )
#                 else:
#                     selected_customers = []
#                     st.info("No customers found in data")
            
#             with filter_col2:
#                 if len(models_list) > 0:
#                     selected_models = st.multiselect(
#                         "Filter by Product Model",
#                         options=models_list,
#                         default=[],
#                         key=f"model_filter_{modelo_key}",
#                         help="Select one or more models (empty = all)"
#                     )
#                 else:
#                     selected_models = []
#                     st.info("No models found in data")
            
#             with filter_col3:
#                 st.write("")  # Spacing
#                 st.write("")  # Spacing
#                 if st.button("🔄 Clear Filters", key=f"clear_{modelo_key}"):
#                     st.rerun()
            
#             st.markdown('</div>', unsafe_allow_html=True)
            
#             # Apply filters
#             df_filtered = aplicar_filtros(df_resultado, customer_col, model_col, 
#                                          selected_customers, selected_models)
            
#             # Show filter summary
#             if len(selected_customers) > 0 or len(selected_models) > 0:
#                 filter_info = []
#                 if len(selected_customers) > 0:
#                     filter_info.append(f"**{len(selected_customers)}** customer(s)")
#                 if len(selected_models) > 0:
#                     filter_info.append(f"**{len(selected_models)}** model(s)")
                
#                 st.info(f"🔎 Filtering by: {' and '.join(filter_info)} | Showing **{len(df_filtered)}** of **{len(df_resultado)}** products")
            
#             # ==========================================
#             # RESULTS DISPLAY
#             # ==========================================
            
#             col1, col2 = st.columns([3, 1])
            
#             with col1:
#                 st.subheader("📊 Results Table")
                
#                 if not df_filtered.empty:
#                     # Show filtered results
#                     if df_filtered.shape[1] > 20:
#                         # Show first 8 columns + last 10 columns (forecast)
#                         cols_to_show = list(df_filtered.columns[:8]) + list(df_filtered.columns[-10:])
#                         df_display = df_filtered[cols_to_show]
#                     else:
#                         df_display = df_filtered
                    
#                     st.dataframe(df_display.head(50), use_container_width=True)
                    
#                     # Summary statistics
#                     st.subheader("📈 Forecast Summary")
                    
#                     # Calculate total forecast by month
#                     date_cols = [col for col in df_filtered.columns if isinstance(col, datetime)]
#                     if date_cols:
#                         forecast_summary = df_filtered[date_cols].sum()
                        
#                         # Create chart
#                         if PLOTLY_AVAILABLE and len(forecast_summary) > 0:
#                             fig = go.Figure()
#                             fig.add_trace(go.Scatter(
#                                 x=[d.strftime('%Y-%m') for d in forecast_summary.index],
#                                 y=forecast_summary.values,
#                                 mode='lines+markers',
#                                 name='Total Forecast',
#                                 line=dict(color='#FF6B6B', width=2),
#                                 marker=dict(size=6)
#                             ))
#                             fig.update_layout(
#                                 title="Total Forecast by Month (Filtered Data)",
#                                 xaxis_title="Month",
#                                 yaxis_title="Units",
#                                 hovermode='x unified',
#                                 height=350
#                             )
#                             st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.warning("⚠️ No data matches the selected filters")
            
#             with col2:
#                 # Metrics
#                 st.subheader("📊 Metrics")
                
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h4>Cells Updated</h4>
#                     <h2>{resultado.get('celulas_actualizadas', 0):,}</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h4>Products (Total)</h4>
#                     <h2>{len(df_resultado):,}</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h4>Products (Filtered)</h4>
#                     <h2>{len(df_filtered):,}</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 if 'parametros' in resultado:
#                     st.subheader("⚙️ Parameters")
#                     for param, valor in resultado['parametros'].items():
#                         st.write(f"**{param}:** {valor}")
                
#                 # Download buttons
#                 st.subheader("💾 Export")
                
#                 # Determine which dataframe to export
#                 df_to_export = df_filtered if not df_filtered.empty else df_resultado
                
#                 # CSV download
#                 csv = df_to_export.to_csv(index=False)
#                 suffix = "_filtered" if len(df_filtered) < len(df_resultado) else ""
#                 st.download_button(
#                     label="📥 Download CSV",
#                     data=csv,
#                     file_name=f"forecast_{modelo_key}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
#                     mime="text/csv",
#                     use_container_width=True
#                 )
                
#                 # Excel download
#                 try:
#                     from io import BytesIO
#                     buffer = BytesIO()
#                     with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
#                         df_to_export.to_excel(writer, index=False, sheet_name='Forecast')
                    
#                     st.download_button(
#                         label="📥 Download Excel",
#                         data=buffer.getvalue(),
#                         file_name=f"forecast_{modelo_key}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
#                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                         use_container_width=True
#                     )
#                 except:
#                     pass


# def mostrar_pantalla_bienvenida():
#     """Display welcome screen when no file is loaded"""
    
#     st.info("📤 **Please load the consolidated Excel file to begin.**")
    
#     st.markdown("### 📋 Required File Structure:")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         **📊 Sheet: Main**
#         - Row 2: Column headers (first day of each month)
#         - Cell B2: Forecast start date
#         - Columns A-H: Product characteristics
#         - Columns I+: Monthly sales history
#         - Data starts from row 3
#         """)
    
#     with col2:
#         st.markdown("""
#         **⚙️ Sheet: LogicsxMonth**
#         - Row 2: Column headers
#         - Column C: Class
#         - Column D: Month (dd/mm/yyyy)
#         - Column E: Calculation Base
#         - Column G: P2P Model
#         - Column H: Launch Month (XF)
#         - Data starts from row 3
#         """)
    
#     with col3:
#         st.markdown("""
#         **🔗 Sheet: Relations**
#         - Row 8: Year headers
#         - Column A: ID Customer (from row 9)
#         - Columns B+: Growth factors by year
#         - Factor values (e.g., 1.05 = 5% growth)
#         """)
    
#     st.markdown("---")
    
#     # Calculation logic explanation
#     st.markdown("### 🔍 Calculation Logic:")
    
#     st.markdown("""
#     <div class="info-box">
#     <strong>Calculation Base Options:</strong><br><br>
    
#     <strong>1. DE PARA SEGUINTE</strong> (P2P - Previous to Next):<br>
#     Uses historical data from the P2P model (column G) for the same customer.<br><br>
    
#     <strong>2. Não calcula</strong> (No Calculation):<br>
#     Sets forecast to zero or leaves empty.<br><br>
    
#     <strong>3. Depende do mês de Lançamento</strong> (Launch Month Dependent):<br>
#     Like P2P, but only forecasts from the launch month (column H) forward. 
#     Previous months are set to zero.
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     st.markdown("""
#     <div class="warning-box">
#         <strong>⚠️ Important Notes:</strong><br>
#         • Sheet names must be <strong>exactly</strong>: Main, LogicsxMonth, Relations<br>
#         • File format must be <strong>.xlsx</strong><br>
#         • All sheets must contain data<br>
#         • Date format: dd/mm/yyyy (e.g., 01/01/2026)<br>
#         • Growth factors in Relations apply only to Moving Average model
#     </div>
#     """, unsafe_allow_html=True)


# def main():
#     """Main application function"""
    
#     # Configure sidebar and get parameters
#     uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros = configurar_sidebar()
    
#     # If no file, show welcome screen
#     if not uploaded_file:
#         mostrar_pantalla_bienvenida()
#         return
    
#     # Load data
#     with st.spinner("🔄 Loading consolidated file..."):
#         data_handler = DataHandler(uploaded_file)
        
#         if not data_handler.cargar_archivo():
#             # Show errors
#             for error in data_handler.obtener_errores():
#                 st.error(error)
#             return
    
#     # Display load information
#     mostrar_info_carga(data_handler)
    
#     # Validate at least one model is selected
#     if not any([modelo_media, modelo_suavizacao, modelo_arima]):
#         st.warning("⚠️ Please select at least one model to execute.")
#         return
    
#     # Execute forecast button
#     if st.button("🚀 Execute Forecast", type="primary", use_container_width=True):
        
#         with st.spinner("Processing models... This may take a few minutes."):
#             try:
#                 # Create processor
#                 processor = ForecastProcessor(
#                     data_handler.obtener_dataframes(),
#                     data_handler.obtener_fecha_inicio(),
#                     parametros
#                 )
                
#                 # Execute selected models
#                 modelos_ejecutar = {
#                     'media_movil': modelo_media,
#                     'suavizacao_exponencial': modelo_suavizacao,
#                     'arima': modelo_arima
#                 }
                
#                 resultados = processor.ejecutar_forecast(modelos_ejecutar)
                
#                 if resultados:
#                     mostrar_resultados(resultados)
#                 else:
#                     st.error("❌ Could not generate results. Please verify input data.")
                    
#             except Exception as e:
#                 st.error(f"❌ Error during processing: {str(e)}")
#                 st.exception(e)


# if __name__ == "__main__":
#     main()










# """
# Forecasting processing engine - Optimized version
# """
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# from typing import Dict, Any, List, Tuple
# import time

# # Optional imports for advanced models
# try:
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#     STATSMODELS_AVAILABLE = True
# except ImportError:
#     STATSMODELS_AVAILABLE = False

# try:
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     SARIMAX_AVAILABLE = True
# except ImportError:
#     SARIMAX_AVAILABLE = False


# class ForecastProcessor:
#     """Main class for forecasting processing - Optimized"""
    
#     # Calculation base options
#     CALC_BASE_P2P = "DE PARA SEGUINTE"
#     CALC_BASE_NO_CALC = "Não calcula"
#     CALC_BASE_LAUNCH_DEPENDENT = "Depende do mês de Lançamento"
    
#     def __init__(self, dataframes: Dict[str, pd.DataFrame], forecast_start_date: datetime, parametros: Dict[str, Any]):
#         """
#         Initialize forecasting processor
        
#         Args:
#             dataframes: Dictionary with 'main', 'logics', 'relations'
#             forecast_start_date: Forecast start date (from B2)
#             parametros: Model configuration parameters
#         """
#         self.df_main = dataframes['main'].copy()
#         self.df_logics = dataframes['logics'].copy()
#         self.df_relations = dataframes['relations'].copy()
#         self.forecast_start_date = pd.to_datetime(forecast_start_date)
#         self.parametros = parametros
        
#         # Column names will be detected and mapped
#         self.col_names = {}
        
#         # Identify date columns in Main sheet
#         self.date_columns = self._identify_date_columns()
        
#         # Number of months to forecast
#         self.forecast_months = 18
        
#         # Generate forecast dates
#         self.forecast_dates = self._generate_forecast_dates()
        
#         # Cache for optimization
#         self._logic_cache = {}
#         self._growth_factor_cache = {}
#         self._p2p_series_cache = {}
    
#     def _identify_date_columns(self) -> List[datetime]:
#         """Identify columns that are dates in Main DataFrame"""
#         date_cols = []
#         for col in self.df_main.columns:
#             if isinstance(col, datetime):
#                 date_cols.append(col)
#         return sorted(date_cols)
    
#     def _generate_forecast_dates(self) -> List[datetime]:
#         """Generate list of forecast dates (18 months from start date)"""
#         forecast_dates = []
#         current_date = self.forecast_start_date
        
#         for i in range(self.forecast_months):
#             forecast_dates.append(current_date)
#             current_date = current_date + relativedelta(months=1)
        
#         return forecast_dates
    
#     def _detect_column_names(self):
#         """Detect actual column names in DataFrames and create mapping"""
        
#         column_variations = {
#             'customer': ['ID Cliente', 'ID Customer', 'Customer', 'Cliente', 'ID_Customer', 'IDCustomer'],
#             'product_model': ['Modelo do Produto', 'Product Model', 'Model', 'Modelo', 'Product_Model'],
#             'class': ['Classe', 'Class', 'CLASSE', 'CLASS'],
#             'description': ['Descrição do Modelo do Produto', 'Product Description', 'Description', 'Descrição', 'Product_Description'],
#             'calc_base': ['Base de Cálculo', 'Calculation Base', 'Base', 'Calculation_Base'],
#             'month': ['Mês', 'Month', 'MÊS', 'MONTH'],
#             'p2p': ['P2P', 'p2p'],
#             'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Launch_Month']
#         }
        
#         # Detect in Main DataFrame
#         for key, variations in column_variations.items():
#             for var in variations:
#                 if var in self.df_main.columns:
#                     self.col_names[f'main_{key}'] = var
#                     break
        
#         # Detect in Logics DataFrame
#         for key, variations in column_variations.items():
#             for var in variations:
#                 if var in self.df_logics.columns:
#                     self.col_names[f'logics_{key}'] = var
#                     break
        
#         # Detect in Relations DataFrame
#         for key, variations in column_variations.items():
#             for var in variations:
#                 if var in self.df_relations.columns:
#                     self.col_names[f'relations_{key}'] = var
#                     break
        
#         st.info(f"🔍 Detected column mappings: {len(self.col_names)} columns identified")
        
#         with st.expander("🔍 Column Detection Details"):
#             st.write("**Main sheet columns:**")
#             st.write(list(self.df_main.columns[:15]))
#             st.write("**Logics sheet columns:**")
#             st.write(list(self.df_logics.columns))
#             st.write("**Relations sheet columns:**")
#             st.write(list(self.df_relations.columns[:10]))
#             st.write("**Detected mappings:**")
#             st.json(self.col_names)
    
#     def ejecutar_forecast(self, modelos_ejecutar: Dict[str, bool]) -> Dict[str, Any]:
#         """Execute selected forecasting models"""
        
#         resultados = {}
        
#         # Detect column names first
#         self._detect_column_names()
        
#         # Prepare base data
#         datos_preparados = self._preparar_datos()
        
#         # Pre-build caches for optimization
#         self._build_caches(datos_preparados)
        
#         # Execute Moving Average
#         if modelos_ejecutar.get('media_movil', False):
#             with st.spinner("📈 Executing Moving Average (Optimized)..."):
#                 start_time = time.time()
#                 resultados['media_movil'] = self._ejecutar_media_movil(datos_preparados)
#                 resultados['media_movil']['metadata']['execution_time'] = time.time() - start_time
        
#         # Execute Exponential Smoothing
#         if modelos_ejecutar.get('suavizacao_exponencial', False):
#             with st.spinner("📊 Executing Exponential Smoothing..."):
#                 start_time = time.time()
#                 resultados['suavizacao_exponencial'] = self._ejecutar_suavizacao(datos_preparados)
#                 resultados['suavizacao_exponencial']['metadata']['execution_time'] = time.time() - start_time
        
#         # Execute ARIMA
#         if modelos_ejecutar.get('arima', False):
#             with st.spinner("🔬 Executing SARIMA..."):
#                 start_time = time.time()
#                 resultados['arima'] = self._ejecutar_arima(datos_preparados)
#                 resultados['arima']['metadata']['execution_time'] = time.time() - start_time
        
#         return resultados
    
#     def _preparar_datos(self) -> Dict[str, Any]:
#         """Prepare data for forecasting"""
        
#         return {
#             'df_main': self.df_main,
#             'df_logics': self.df_logics,
#             'df_relations': self.df_relations,
#             'date_columns': self.date_columns,
#             'forecast_dates': self.forecast_dates,
#             'forecast_start_date': self.forecast_start_date,
#             'n_products': len(self.df_main),
#             'n_periods': len(self.date_columns),
#             'col_names': self.col_names
#         }
    
#     def _build_caches(self, datos: Dict[str, Any]):
#         """Pre-build caches for faster execution - OPTIMIZATION"""
        
#         df_logics = datos['df_logics']
#         df_relations = datos['df_relations']
#         col_names = datos['col_names']
        
#         # Cache 1: Logic lookup (class + month -> logic)
#         class_col = col_names.get('logics_class')
#         month_col = col_names.get('logics_month')
#         calc_base_col = col_names.get('logics_calc_base')
#         p2p_col = col_names.get('logics_p2p')
#         launch_col = col_names.get('logics_launch_month')
        
#         if class_col and month_col and class_col in df_logics.columns and month_col in df_logics.columns:
#             for idx, row in df_logics.iterrows():
#                 product_class = row.get(class_col)
#                 month_date = pd.to_datetime(row.get(month_col)).replace(day=1)
                
#                 cache_key = (product_class, month_date)
                
#                 self._logic_cache[cache_key] = (
#                     row.get(calc_base_col, self.CALC_BASE_P2P) if calc_base_col else self.CALC_BASE_P2P,
#                     row.get(p2p_col, None) if p2p_col else None,
#                     row.get(launch_col, None) if launch_col else None
#                 )
        
#         # Cache 2: Growth factors (customer + year -> factor)
#         customer_col = col_names.get('relations_customer')
        
#         if customer_col and customer_col in df_relations.columns:
#             for idx, row in df_relations.iterrows():
#                 customer = row.get(customer_col)
                
#                 # Get all year columns
#                 for col in df_relations.columns:
#                     # Check if column is a year (can be string or int)
#                     is_year_column = False
                    
#                     if isinstance(col, str) and col.isdigit():
#                         # String year like '2024', '2025'
#                         is_year_column = True
#                         year = int(col)
#                     elif isinstance(col, int) and 2000 <= col <= 2100:
#                         # Direct integer year
#                         is_year_column = True
#                         year = col
                    
#                     if is_year_column:
#                         factor = row.get(col)
                        
#                         if pd.notna(factor) and isinstance(factor, (int, float)):
#                             cache_key = (customer, year)
#                             self._growth_factor_cache[cache_key] = float(factor)
        
#         # Cache 3: P2P series (customer + p2p_model -> series) - built on demand
#         # This is done lazily to avoid loading all series upfront
    
#     def _get_logic_for_product_cached(self, customer: str, product_class: str, 
#                                      month_date: datetime) -> Tuple[str, str, Any]:
#         """Get logic from cache - OPTIMIZED"""
        
#         month_start = pd.to_datetime(month_date).replace(day=1)
#         cache_key = (product_class, month_start)
        
#         if cache_key in self._logic_cache:
#             return self._logic_cache[cache_key]
        
#         # Default if not found
#         return self.CALC_BASE_P2P, None, None
    
#     def _get_growth_factor_cached(self, customer: str, year: int) -> float:
#         """Get growth factor from cache - OPTIMIZED"""
        
#         cache_key = (customer, year)
        
#         if cache_key in self._growth_factor_cache:
#             return self._growth_factor_cache[cache_key]
        
#         return 1.0
    
#     def _get_p2p_series_cached(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
#         """Get P2P series from cache - OPTIMIZED"""
        
#         cache_key = (customer, p2p_model)
        
#         if cache_key in self._p2p_series_cache:
#             return self._p2p_series_cache[cache_key]
        
#         # Build and cache
#         series = self._get_p2p_series_indexed(customer, p2p_model, datos)
#         self._p2p_series_cache[cache_key] = series
        
#         return series
    
#     def _get_seasonal_window(self, forecast_date: datetime, historical_series: pd.Series) -> List[float]:
#         """Get seasonal window: same month last year + 2 following months"""
        
#         same_month_last_year = forecast_date - relativedelta(years=1)
#         month_after_1 = same_month_last_year + relativedelta(months=1)
#         month_after_2 = same_month_last_year + relativedelta(months=2)
        
#         values = []
        
#         for target_date in [same_month_last_year, month_after_1, month_after_2]:
#             if target_date in historical_series.index:
#                 val = historical_series[target_date]
#                 if pd.notna(val) and val > 0:
#                     values.append(float(val))
        
#         return values
    
#     # ============================================
#     # MOVING AVERAGE MODEL (OPTIMIZED)
#     # ============================================
    
#     def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Seasonal Moving Average - HEAVILY OPTIMIZED"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # OPTIMIZATION: Vectorized operations where possible
#             # Extract all historical series at once
#             historical_data = {}
#             for idx in range(len(df_main)):
#                 series = pd.Series(dtype=float)
#                 row = df_main.iloc[idx]
                
#                 for date_col in datos['date_columns']:
#                     val = pd.to_numeric(row.get(date_col, np.nan), errors='coerce')
#                     series[date_col] = val
                
#                 historical_data[idx] = series
            
#             # Progress bar
#             progress_bar = st.progress(0)
#             total_products = len(df_main)
            
#             # Process each product row
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get(customer_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Get historical series (already computed)
#                 historical_series = historical_data[idx]
                
#                 # Process each forecast month
#                 for forecast_date in datos['forecast_dates']:
                    
#                     # Get logic from cache
#                     calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                         customer, product_class, forecast_date
#                     )
                    
#                     # Handle "Não calcula"
#                     if calc_base == self.CALC_BASE_NO_CALC:
#                         forecast_arrays[forecast_date][idx] = 0
#                         cells_updated += 1
#                         continue
                    
#                     # Handle launch dependent
#                     if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 forecast_arrays[forecast_date][idx] = 0
#                                 cells_updated += 1
#                                 continue
                    
#                     # Handle P2P
#                     if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         # Get historical series (from P2P or same product)
#                         if pd.notna(p2p_model) and p2p_model != '':
#                             series_to_use = self._get_p2p_series_cached(customer, p2p_model, datos)
#                         else:
#                             series_to_use = historical_series
                        
#                         if len(series_to_use) > 0:
#                             # Get seasonal window
#                             seasonal_values = self._get_seasonal_window(forecast_date, series_to_use)
                            
#                             if len(seasonal_values) > 0:
#                                 seasonal_avg = np.mean(seasonal_values)
                                
#                                 # Get growth factor from cache
#                                 year = forecast_date.year
#                                 growth_factor = self._get_growth_factor_cached(customer, year)
                                
#                                 forecasted_value = seasonal_avg * growth_factor
#                                 forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
#                                 cells_updated += 1
#                             else:
#                                 # Fallback to simple average
#                                 simple_avg = series_to_use.mean()
#                                 year = forecast_date.year
#                                 growth_factor = self._get_growth_factor_cached(customer, year)
#                                 forecast_arrays[forecast_date][idx] = max(0, simple_avg * growth_factor)
#                                 cells_updated += 1
#                         else:
#                             forecast_arrays[forecast_date][idx] = 0
#                             cells_updated += 1
                
#                 processed_products += 1
                
#                 # Update progress every 100 products
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             # Clear progress bar
#             progress_bar.empty()
            
#             # Add forecast columns to result
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'seasonal_moving_average',
#                     'window': 'same_month_last_year + 2_months',
#                     'growth_factors_applied': True,
#                     'optimized': True
#                 },
#                 'metadata': {
#                     'n_products_processed': processed_products,
#                     'forecast_months': len(datos['forecast_dates']),
#                     'execution_date': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             st.error(f"Error in Moving Average: {str(e)}")
#             st.exception(e)
#             return self._resultado_vacio()
    
#     # ============================================
#     # EXPONENTIAL SMOOTHING (OPTIMIZED)
#     # ============================================
    
#     def _ejecutar_suavizacao(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Exponential Smoothing - OPTIMIZED"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Progress bar
#             progress_bar = st.progress(0)
#             total_products = len(df_main)
            
#             # Process each product row
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get(customer_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Get logic (use first forecast month)
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Get historical series
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     historical_series = self._get_historical_series_indexed(idx, datos)
                
#                 if len(historical_series.dropna()) >= 12:
                    
#                     try:
#                         if STATSMODELS_AVAILABLE:
#                             clean_series = historical_series.dropna()
#                             clean_series = clean_series[clean_series > 0]
                            
#                             if len(clean_series) >= 12:
#                                 model = ExponentialSmoothing(
#                                     clean_series,
#                                     seasonal_periods=12,
#                                     trend='add',
#                                     seasonal='add'
#                                 )
#                                 fitted = model.fit()
#                                 forecast_values = fitted.forecast(steps=18)
                                
#                                 for i, forecast_date in enumerate(datos['forecast_dates']):
#                                     if i < len(forecast_values):
#                                         forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
#                                         cells_updated += 1
                                
#                                 processed_products += 1
#                         else:
#                             # Simplified
#                             last_value = historical_series.dropna().iloc[-1]
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, last_value)
#                                 cells_updated += 1
#                             processed_products += 1
                    
#                     except Exception as e:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         for forecast_date in datos['forecast_dates']:
#                             forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                             cells_updated += 1
#                         processed_products += 1
                
#                 # Update progress
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
#                     'seasonal_periods': 12,
#                     'optimized': True
#                 },
#                 'metadata': {
#                     'n_products_processed': processed_products,
#                     'forecast_months': len(datos['forecast_dates']),
#                     'execution_date': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             st.error(f"Error in Exponential Smoothing: {str(e)}")
#             st.exception(e)
#             return self._resultado_vacio()
    
#     # ============================================
#     # SARIMA MODEL (OPTIMIZED)
#     # ============================================
    
#     def _ejecutar_arima(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute SARIMA - OPTIMIZED"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             arima_params = self.parametros.get('arima_params', (1, 1, 1))
#             cells_updated = 0
#             processed_products = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Progress bar
#             progress_bar = st.progress(0)
#             total_products = len(df_main)
            
#             # Process each product
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get(customer_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Get logic
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Get historical series
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     historical_series = self._get_historical_series_indexed(idx, datos)
                
#                 if len(historical_series.dropna()) >= 18:
                    
#                     try:
#                         if SARIMAX_AVAILABLE:
#                             clean_series = historical_series.dropna()
#                             clean_series = clean_series[clean_series > 0]
                            
#                             if len(clean_series) >= 18:
#                                 model = SARIMAX(
#                                     clean_series,
#                                     order=arima_params,
#                                     seasonal_order=(1, 1, 1, 12),
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False
#                                 )
                                
#                                 fitted = model.fit(disp=False)
#                                 forecast_values = fitted.forecast(steps=18)
                                
#                                 for i, forecast_date in enumerate(datos['forecast_dates']):
#                                     if i < len(forecast_values):
#                                         forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
#                                         cells_updated += 1
                                
#                                 processed_products += 1
#                         else:
#                             # Simplified trend-based
#                             clean_series = historical_series.dropna()
#                             if len(clean_series) >= 6:
#                                 x = np.arange(len(clean_series))
#                                 y = clean_series.values
#                                 z = np.polyfit(x, y, 1)
#                                 trend = z[0]
#                                 base = y[-1]
                                
#                                 for i, forecast_date in enumerate(datos['forecast_dates']):
#                                     forecast_value = base + trend * (i + 1)
#                                     forecast_arrays[forecast_date][idx] = max(0, forecast_value)
#                                     cells_updated += 1
                                
#                                 processed_products += 1
                    
#                     except Exception as e:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         for forecast_date in datos['forecast_dates']:
#                             forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                             cells_updated += 1
#                         processed_products += 1
                
#                 # Update progress
#                 if processed_products % 50 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
#                     'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
#                     'optimized': True
#                 },
#                 'metadata': {
#                     'n_products_processed': processed_products,
#                     'forecast_months': len(datos['forecast_dates']),
#                     'execution_date': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             st.error(f"Error in ARIMA: {str(e)}")
#             st.exception(e)
#             return self._resultado_vacio()
    
#     # ============================================
#     # AUXILIARY METHODS
#     # ============================================
    
#     def _get_historical_series_indexed(self, row_idx: int, datos: Dict) -> pd.Series:
#         """Get historical time series with datetime index"""
        
#         df_main = datos['df_main']
#         row = df_main.iloc[row_idx]
        
#         series = pd.Series(dtype=float)
        
#         for date_col in datos['date_columns']:
#             val = row.get(date_col, np.nan)
#             val = pd.to_numeric(val, errors='coerce')
#             series[date_col] = val
        
#         return series
    
#     def _get_p2p_series_indexed(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
#         """Get P2P series with datetime index"""
        
#         df_main = datos['df_main']
#         col_names = datos['col_names']
        
#         customer_col = col_names.get('main_customer')
#         model_col = col_names.get('main_product_model')
        
#         if not customer_col or not model_col:
#             return pd.Series(dtype=float)
        
#         try:
#             mask = (
#                 (df_main[customer_col] == customer) &
#                 (df_main[model_col] == p2p_model)
#             )
            
#             matching_rows = df_main[mask]
            
#             if len(matching_rows) > 0:
#                 row = matching_rows.iloc[0]
#                 series = pd.Series(dtype=float)
                
#                 for date_col in datos['date_columns']:
#                     val = row.get(date_col, np.nan)
#                     val = pd.to_numeric(val, errors='coerce')
#                     series[date_col] = val
                
#                 return series
#         except Exception as e:
#             pass
        
#         return pd.Series(dtype=float)
    
#     def _resultado_vacio(self) -> Dict[str, Any]:
#         """Return empty result structure"""
#         return {
#             'dataframe': pd.DataFrame(),
#             'celulas_actualizadas': 0,
#             'parametros': {},
#             'metadata': {'error': True}
#         }







"""
Forecasting processing engine - Optimized version
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Tuple
import time

# Optional imports for advanced models
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False


class ForecastProcessor:
    """Main class for forecasting processing - Optimized"""
    
    # Calculation base options
    CALC_BASE_P2P = "DE PARA SEGUINTE"
    CALC_BASE_NO_CALC = "Não calcula"
    CALC_BASE_LAUNCH_DEPENDENT = "Depende do mês de Lançamento"
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame], forecast_start_date: datetime, parametros: Dict[str, Any]):
        """
        Initialize forecasting processor
        
        Args:
            dataframes: Dictionary with 'main', 'logics', 'relations'
            forecast_start_date: Forecast start date (from B2)
            parametros: Model configuration parameters
        """
        self.df_main = dataframes['main'].copy()
        self.df_logics = dataframes['logics'].copy()
        self.df_relations = dataframes['relations'].copy()
        self.forecast_start_date = pd.to_datetime(forecast_start_date)
        self.parametros = parametros
        
        # Column names will be detected and mapped
        self.col_names = {}
        
        # Identify date columns in Main sheet
        self.date_columns = self._identify_date_columns()
        
        # Number of months to forecast
        self.forecast_months = 18
        
        # Generate forecast dates
        self.forecast_dates = self._generate_forecast_dates()
        
        # Cache for optimization
        self._logic_cache = {}
        self._growth_factor_cache = {}
        self._p2p_series_cache = {}
    
    def _identify_date_columns(self) -> List[datetime]:
        """Identify columns that are dates in Main DataFrame"""
        date_cols = []
        for col in self.df_main.columns:
            if isinstance(col, datetime):
                date_cols.append(col)
        return sorted(date_cols)
    
    def _generate_forecast_dates(self) -> List[datetime]:
        """Generate list of forecast dates (18 months from start date)"""
        forecast_dates = []
        current_date = self.forecast_start_date
        
        for i in range(self.forecast_months):
            forecast_dates.append(current_date)
            current_date = current_date + relativedelta(months=1)
        
        return forecast_dates
    
    def _detect_column_names(self):
        """Detect actual column names in DataFrames and create mapping"""
        
        column_variations = {
            'customer': ['ID Cliente', 'ID Customer', 'Customer', 'Cliente', 'ID_Customer', 'IDCustomer'],
            'product_model': ['Modelo do Produto', 'Product Model', 'Model', 'Modelo', 'Product_Model'],
            'class': ['Classe', 'Class', 'CLASSE', 'CLASS'],
            'description': ['Descrição do Modelo do Produto', 'Product Description', 'Description', 'Descrição', 'Product_Description'],
            'calc_base': ['Base de Cálculo', 'Calculation Base', 'Base', 'Calculation_Base'],
            'month': ['Mês', 'Month', 'MÊS', 'MONTH'],
            'p2p': ['P2P', 'p2p'],
            'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Launch_Month']
        }
        
        # Detect in Main DataFrame
        for key, variations in column_variations.items():
            for var in variations:
                if var in self.df_main.columns:
                    self.col_names[f'main_{key}'] = var
                    break
        
        # Detect in Logics DataFrame
        for key, variations in column_variations.items():
            for var in variations:
                if var in self.df_logics.columns:
                    self.col_names[f'logics_{key}'] = var
                    break
        
        # Detect in Relations DataFrame
        for key, variations in column_variations.items():
            for var in variations:
                if var in self.df_relations.columns:
                    self.col_names[f'relations_{key}'] = var
                    break
        
        st.info(f"🔍 Detected column mappings: {len(self.col_names)} columns identified")
        
        with st.expander("🔍 Column Detection Details"):
            st.write("**Main sheet columns:**")
            st.write(list(self.df_main.columns[:15]))
            st.write("**Logics sheet columns:**")
            st.write(list(self.df_logics.columns))
            st.write("**Relations sheet columns:**")
            st.write(list(self.df_relations.columns[:10]))
            st.write("**Detected mappings:**")
            st.json(self.col_names)
    
    def ejecutar_forecast(self, modelos_ejecutar: Dict[str, bool]) -> Dict[str, Any]:
        """Execute selected forecasting models"""
        
        resultados = {}
        
        # Detect column names first
        self._detect_column_names()
        
        # Prepare base data
        datos_preparados = self._preparar_datos()
        
        # Pre-build caches for optimization
        self._build_caches(datos_preparados)
        
        # Execute Moving Average
        if modelos_ejecutar.get('media_movil', False):
            with st.spinner("📈 Executing Moving Average (Optimized)..."):
                start_time = time.time()
                resultados['media_movil'] = self._ejecutar_media_movil(datos_preparados)
                resultados['media_movil']['metadata']['execution_time'] = time.time() - start_time
        
        # Execute Exponential Smoothing
        if modelos_ejecutar.get('suavizacao_exponencial', False):
            with st.spinner("📊 Executing Exponential Smoothing..."):
                start_time = time.time()
                resultados['suavizacao_exponencial'] = self._ejecutar_suavizacao(datos_preparados)
                resultados['suavizacao_exponencial']['metadata']['execution_time'] = time.time() - start_time
        
        # Execute ARIMA
        if modelos_ejecutar.get('arima', False):
            with st.spinner("🔬 Executing SARIMA..."):
                start_time = time.time()
                resultados['arima'] = self._ejecutar_arima(datos_preparados)
                resultados['arima']['metadata']['execution_time'] = time.time() - start_time
        
        return resultados
    
    def _preparar_datos(self) -> Dict[str, Any]:
        """Prepare data for forecasting"""
        
        return {
            'df_main': self.df_main,
            'df_logics': self.df_logics,
            'df_relations': self.df_relations,
            'date_columns': self.date_columns,
            'forecast_dates': self.forecast_dates,
            'forecast_start_date': self.forecast_start_date,
            'n_products': len(self.df_main),
            'n_periods': len(self.date_columns),
            'col_names': self.col_names
        }
    
    def _build_caches(self, datos: Dict[str, Any]):
        """Pre-build caches for faster execution - OPTIMIZATION"""
        
        df_logics = datos['df_logics']
        df_relations = datos['df_relations']
        col_names = datos['col_names']
        
        # Cache 1: Logic lookup (class + month -> logic)
        class_col = col_names.get('logics_class')
        month_col = col_names.get('logics_month')
        calc_base_col = col_names.get('logics_calc_base')
        p2p_col = col_names.get('logics_p2p')
        launch_col = col_names.get('logics_launch_month')
        
        if class_col and month_col and class_col in df_logics.columns and month_col in df_logics.columns:
            for idx, row in df_logics.iterrows():
                product_class = row.get(class_col)
                month_date = pd.to_datetime(row.get(month_col)).replace(day=1)
                
                cache_key = (product_class, month_date)
                
                self._logic_cache[cache_key] = (
                    row.get(calc_base_col, self.CALC_BASE_P2P) if calc_base_col else self.CALC_BASE_P2P,
                    row.get(p2p_col, None) if p2p_col else None,
                    row.get(launch_col, None) if launch_col else None
                )
        
        # Cache 2: Growth factors (customer + year -> factor)
        customer_col = col_names.get('relations_customer')
        
        if customer_col and customer_col in df_relations.columns:
            for idx, row in df_relations.iterrows():
                customer = row.get(customer_col)
                
                # Get all year columns
                for col in df_relations.columns:
                    # Check if column is a year (can be string or int)
                    is_year_column = False
                    
                    if isinstance(col, str) and col.isdigit():
                        # String year like '2024', '2025'
                        is_year_column = True
                        year = int(col)
                    elif isinstance(col, int) and 2000 <= col <= 2100:
                        # Direct integer year
                        is_year_column = True
                        year = col
                    
                    if is_year_column:
                        factor = row.get(col)
                        
                        if pd.notna(factor) and isinstance(factor, (int, float)):
                            cache_key = (customer, year)
                            self._growth_factor_cache[cache_key] = float(factor)
        
        # Cache 3: P2P series (customer + p2p_model -> series) - built on demand
        # This is done lazily to avoid loading all series upfront
    
    def _get_logic_for_product_cached(self, customer: str, product_class: str, 
                                     month_date: datetime) -> Tuple[str, str, Any]:
        """Get logic from cache - OPTIMIZED"""
        
        month_start = pd.to_datetime(month_date).replace(day=1)
        cache_key = (product_class, month_start)
        
        if cache_key in self._logic_cache:
            return self._logic_cache[cache_key]
        
        # Default if not found
        return self.CALC_BASE_P2P, None, None
    
    def _get_growth_factor_cached(self, customer: str, year: int) -> float:
        """Get growth factor from cache - OPTIMIZED"""
        
        cache_key = (customer, year)
        
        if cache_key in self._growth_factor_cache:
            return self._growth_factor_cache[cache_key]
        
        return 1.0
    
    def _get_p2p_series_cached(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
        """Get P2P series from cache - OPTIMIZED"""
        
        cache_key = (customer, p2p_model)
        
        if cache_key in self._p2p_series_cache:
            return self._p2p_series_cache[cache_key]
        
        # Build and cache
        series = self._get_p2p_series_indexed(customer, p2p_model, datos)
        self._p2p_series_cache[cache_key] = series
        
        return series
    
    def _get_seasonal_window(self, forecast_date: datetime, historical_series: pd.Series) -> List[float]:
        """Get seasonal window: same month last year + 2 following months"""
        
        same_month_last_year = forecast_date - relativedelta(years=1)
        month_after_1 = same_month_last_year + relativedelta(months=1)
        month_after_2 = same_month_last_year + relativedelta(months=2)
        
        values = []
        
        for target_date in [same_month_last_year, month_after_1, month_after_2]:
            if target_date in historical_series.index:
                val = historical_series[target_date]
                if pd.notna(val) and val > 0:
                    values.append(float(val))
        
        return values
    
    def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Seasonal Moving Average - HEAVILY OPTIMIZED"""
        
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            cells_updated = 0
            processed_products = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
                return self._resultado_vacio()
            
            # Pre-allocate forecast columns with NaN
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # OPTIMIZATION: Extract all historical series at once
            historical_data = {}
            for idx in range(len(df_main)):
                series = pd.Series(dtype=float)
                row = df_main.iloc[idx]
                
                for date_col in datos['date_columns']:
                    val = pd.to_numeric(row.get(date_col, np.nan), errors='coerce')
                    series[date_col] = val
                
                historical_data[idx] = series
            
            # Progress bar
            progress_bar = st.progress(0)
            total_products = len(df_main)
            
            # Process each product row
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Get historical series (already computed)
                historical_series = historical_data[idx]
                
                # Process each forecast month
                for forecast_date in datos['forecast_dates']:
                    
                    # Get logic from cache
                    calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
                        customer, product_class, forecast_date
                    )
                    
                    # Handle "Não calcula"
                    if calc_base == self.CALC_BASE_NO_CALC:
                        forecast_arrays[forecast_date][idx] = 0
                        cells_updated += 1
                        continue
                    
                    # Handle launch dependent
                    if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
                        if pd.notna(launch_month):
                            launch_date = pd.to_datetime(launch_month)
                            if forecast_date < launch_date:
                                forecast_arrays[forecast_date][idx] = 0
                                cells_updated += 1
                                continue
                    
                    # Handle P2P
                    if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
                        # Get historical series (from P2P or same product)
                        if pd.notna(p2p_model) and p2p_model != '':
                            series_to_use = self._get_p2p_series_cached(customer, p2p_model, datos)
                        else:
                            series_to_use = historical_series
                        
                        if len(series_to_use) > 0:
                            # Get seasonal window
                            seasonal_values = self._get_seasonal_window(forecast_date, series_to_use)
                            
                            if len(seasonal_values) > 0:
                                seasonal_avg = np.mean(seasonal_values)
                                
                                # Get growth factor from cache
                                year = forecast_date.year
                                growth_factor = self._get_growth_factor_cached(customer, year)
                                
                                forecasted_value = seasonal_avg * growth_factor
                                forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
                                cells_updated += 1
                            else:
                                # Fallback to simple average
                                simple_avg = series_to_use.mean()
                                year = forecast_date.year
                                growth_factor = self._get_growth_factor_cached(customer, year)
                                forecast_arrays[forecast_date][idx] = max(0, simple_avg * growth_factor)
                                cells_updated += 1
                        else:
                            forecast_arrays[forecast_date][idx] = 0
                            cells_updated += 1
                
                processed_products += 1
                
                # Update progress every 100 products
                if processed_products % 100 == 0:
                    progress_bar.progress(min(processed_products / total_products, 1.0))
            
            # Clear progress bar
            progress_bar.empty()
            
            # Add forecast columns to result
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'seasonal_moving_average',
                    'window': 'same_month_last_year + 2_months',
                    'growth_factors_applied': True,
                    'optimized': True
                },
                'metadata': {
                    'n_products_processed': processed_products,
                    'forecast_months': len(datos['forecast_dates']),
                    'execution_date': datetime.now()
                }
            }
            
        except Exception as e:
            st.error(f"Error in Moving Average: {str(e)}")
            st.exception(e)
            return self._resultado_vacio()
    
    def _ejecutar_suavizacao(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Exponential Smoothing - OPTIMIZED"""
        
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            cells_updated = 0
            processed_products = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns in Main sheet")
                return self._resultado_vacio()
            
            # Pre-allocate forecast columns
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Progress bar
            progress_bar = st.progress(0)
            total_products = len(df_main)
            
            # Process each product row
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Get logic (use first forecast month)
                calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                # Get historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
                else:
                    historical_series = self._get_historical_series_indexed(idx, datos)
                
                if len(historical_series.dropna()) >= 12:
                    
                    try:
                        if STATSMODELS_AVAILABLE:
                            clean_series = historical_series.dropna()
                            clean_series = clean_series[clean_series > 0]
                            
                            if len(clean_series) >= 12:
                                model = ExponentialSmoothing(
                                    clean_series,
                                    seasonal_periods=12,
                                    trend='add',
                                    seasonal='add'
                                )
                                fitted = model.fit()
                                forecast_values = fitted.forecast(steps=18)
                                
                                for i, forecast_date in enumerate(datos['forecast_dates']):
                                    if i < len(forecast_values):
                                        forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
                                        cells_updated += 1
                                
                                processed_products += 1
                        else:
                            # Simplified
                            last_value = historical_series.dropna().iloc[-1]
                            for forecast_date in datos['forecast_dates']:
                                forecast_arrays[forecast_date][idx] = max(0, last_value)
                                cells_updated += 1
                            processed_products += 1
                    
                    except Exception as e:
                        # Fallback
                        avg_value = historical_series.mean()
                        for forecast_date in datos['forecast_dates']:
                            forecast_arrays[forecast_date][idx] = max(0, avg_value)
                            cells_updated += 1
                        processed_products += 1
                
                # Update progress
                if processed_products % 100 == 0:
                    progress_bar.progress(min(processed_products / total_products, 1.0))
            
            progress_bar.empty()
            
            # Add forecast columns
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
                    'seasonal_periods': 12,
                    'optimized': True
                },
                'metadata': {
                    'n_products_processed': processed_products,
                    'forecast_months': len(datos['forecast_dates']),
                    'execution_date': datetime.now()
                }
            }
            
        except Exception as e:
            st.error(f"Error in Exponential Smoothing: {str(e)}")
            st.exception(e)
            return self._resultado_vacio()
    
    def _ejecutar_arima(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SARIMA - OPTIMIZED"""
        
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            arima_params = self.parametros.get('arima_params', (1, 1, 1))
            cells_updated = 0
            processed_products = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns in Main sheet")
                return self._resultado_vacio()
            
            # Pre-allocate
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Progress bar
            progress_bar = st.progress(0)
            total_products = len(df_main)
            
            # Process each product
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Get logic
                calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                # Get historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
                else:
                    historical_series = self._get_historical_series_indexed(idx, datos)
                
                if len(historical_series.dropna()) >= 18:
                    
                    try:
                        if SARIMAX_AVAILABLE:
                            clean_series = historical_series.dropna()
                            clean_series = clean_series[clean_series > 0]
                            
                            if len(clean_series) >= 18:
                                model = SARIMAX(
                                    clean_series,
                                    order=arima_params,
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                
                                fitted = model.fit(disp=False)
                                forecast_values = fitted.forecast(steps=18)
                                
                                for i, forecast_date in enumerate(datos['forecast_dates']):
                                    if i < len(forecast_values):
                                        forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
                                        cells_updated += 1
                                
                                processed_products += 1
                        else:
                            # Simplified trend-based
                            clean_series = historical_series.dropna()
                            if len(clean_series) >= 6:
                                x = np.arange(len(clean_series))
                                y = clean_series.values
                                z = np.polyfit(x, y, 1)
                                trend = z[0]
                                base = y[-1]
                                
                                for i, forecast_date in enumerate(datos['forecast_dates']):
                                    forecast_value = base + trend * (i + 1)
                                    forecast_arrays[forecast_date][idx] = max(0, forecast_value)
                                    cells_updated += 1
                                
                                processed_products += 1
                    
                    except Exception as e:
                        # Fallback
                        avg_value = historical_series.mean()
                        for forecast_date in datos['forecast_dates']:
                            forecast_arrays[forecast_date][idx] = max(0, avg_value)
                            cells_updated += 1
                        processed_products += 1
                
                # Update progress
                if processed_products % 50 == 0:
                    progress_bar.progress(min(processed_products / total_products, 1.0))
            
            progress_bar.empty()
            
            # Add forecast columns
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
                    'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
                    'optimized': True
                },
                'metadata': {
                    'n_products_processed': processed_products,
                    'forecast_months': len(datos['forecast_dates']),
                    'execution_date': datetime.now()
                }
            }
            
        except Exception as e:
            st.error(f"Error in ARIMA: {str(e)}")
            st.exception(e)
            return self._resultado_vacio()
    
    def _get_historical_series_indexed(self, row_idx: int, datos: Dict) -> pd.Series:
        """Get historical time series with datetime index"""
        
        df_main = datos['df_main']
        row = df_main.iloc[row_idx]
        
        series = pd.Series(dtype=float)
        
        for date_col in datos['date_columns']:
            val = row.get(date_col, np.nan)
            val = pd.to_numeric(val, errors='coerce')
            series[date_col] = val
        
        return series
    
    def _get_p2p_series_indexed(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
        """Get P2P series with datetime index"""
        
        df_main = datos['df_main']
        col_names = datos['col_names']
        
        customer_col = col_names.get('main_customer')
        model_col = col_names.get('main_product_model')
        
        if not customer_col or not model_col:
            return pd.Series(dtype=float)
        
        try:
            mask = (
                (df_main[customer_col] == customer) &
                (df_main[model_col] == p2p_model)
            )
            
            matching_rows = df_main[mask]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                series = pd.Series(dtype=float)
                
                for date_col in datos['date_columns']:
                    val = row.get(date_col, np.nan)
                    val = pd.to_numeric(val, errors='coerce')
                    series[date_col] = val
                
                return series
        except Exception as e:
            pass
        
        return pd.Series(dtype=float)
    
    def _resultado_vacio(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'dataframe': pd.DataFrame(),
            'celulas_actualizadas': 0,
            'parametros': {},
            'metadata': {'error': True}
        }