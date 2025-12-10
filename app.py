# """
# Main User Interface - Streamlit App with Filters and Consolidated Chart
# UPDATED: Added preset models functionality
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
#     page_title="Rolling Collection Tool",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'resultados' not in st.session_state:
#     st.session_state.resultados = None
# if 'data_handler' not in st.session_state:
#     st.session_state.data_handler = None
# if 'file_loaded' not in st.session_state:
#     st.session_state.file_loaded = False

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #279c56;
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
#         padding: 1.5rem;
#         border-radius: 0.5rem;
#         margin: 1.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main header
# st.markdown('<h1 class="main-header">Rolling Collection Tool</h1>', unsafe_allow_html=True)
# st.markdown("---")


# def configurar_sidebar():
#     """Configure sidebar with all controls"""
    
#     with st.sidebar:
        
#         # File upload
#         st.subheader("📁 Load file")
#         uploaded_file = st.file_uploader(
#             "Consolidated Excel File", 
#             type=['xlsx'],
#             help="Excel file with sheets: Main, LogicsxMonth, Relations, and Models",
#             key="file_uploader"
#         )
        
#         st.markdown("---")
        
#         # ==================== PRESET MODELS SECTION ====================
#         st.subheader("Model selection mode")
        
#         # ALWAYS show this checkbox (assume Models sheet exists)
#         usar_preestablecidos = st.checkbox(
#             "Use preset models",
#             value=True,  # ACTIVE BY DEFAULT
#             help="Use best model per product from 'Models' sheet",
#             key="usar_preestablecidos_checkbox"
#         )
        
#         st.markdown("---")
        
#         # ==================== MANUAL MODEL SELECTION ====================
#         st.subheader("🔧 Models to execute")
        
#         # Conditional behavior based on preset checkbox
#         if usar_preestablecidos:
#             # DISABLED when using presets
#             modelo_media = st.checkbox(
#                 "Moving average", 
#                 value=False,
#                 disabled=True,
#                 help="Disabled - Using preset models"
#             )
#             modelo_suavizacao = st.checkbox(
#                 "Exponential smoothing", 
#                 value=False,
#                 disabled=True,
#                 help="Disabled - Using preset models"
#             )
#             modelo_arima = st.checkbox(
#                 "SARIMA", 
#                 value=False,
#                 disabled=True,
#                 help="Disabled - Using preset models"
#             )
#         else:
#             # ENABLED when NOT using presets (ORIGINAL FLOW)
#             modelo_media = st.checkbox(
#                 "Moving average", 
#                 value=True, 
#                 help="Seasonal moving average with growth factors"
#             )
#             modelo_suavizacao = st.checkbox(
#                 "Exponential smoothing", 
#                 value=True,
#                 help="Holt-Winters exponential smoothing"
#             )
#             modelo_arima = st.checkbox(
#                 "SARIMA", 
#                 value=False,
#                 help="Seasonal ARIMA time series model"
#             )
        
#         # ==================== PARAMETERS ====================
#         st.markdown("---")
#         st.subheader("⚙️ Parameters")
        
#         parametros = {}
        
#         # ONLY show parameters if NOT using presets
#         if not usar_preestablecidos:
            
#             # ==================== EXPONENTIAL SMOOTHING PARAMETERS ====================
#             if modelo_suavizacao:
#                 st.markdown("**Exponential smoothing:**")
                
#                 # Model Type Selection
#                 smoothing_type = st.selectbox(
#                     "Model type",
#                     ["Holt-Winters (Triple)", "Holt's Linear (Double)", "Simple (SES)"],
#                     index=0,
#                     help="• Triple: Level + Trend + Seasonality\n• Double: Level + Trend\n• Simple: Level only"
#                 )
#                 parametros['smoothing_type'] = smoothing_type
                
#                 # Create columns for parameters
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     # Alpha (always present)
#                     parametros['alpha'] = st.slider(
#                         "α (Level)", 
#                         0.01, 0.99, 0.30, 0.01,
#                         help="Smoothing factor for level\n• 0.1-0.3: Smooth (more weight to past)\n• 0.7-0.9: Reactive (more weight to recent)"
#                     )
                    
#                     # Beta (only for Double and Triple)
#                     if smoothing_type in ["Holt's Linear (Double)", "Holt-Winters (Triple)"]:
#                         parametros['beta'] = st.slider(
#                             "β (Trend)", 
#                             0.01, 0.50, 0.10, 0.01,
#                             help="Smoothing factor for trend\n• Typical: 0.1-0.3 (more conservative than alpha)"
#                         )
                
#                 with col2:
#                     # Gamma and Seasonality Type (only for Triple)
#                     if smoothing_type == "Holt-Winters (Triple)":
#                         parametros['gamma'] = st.slider(
#                             "γ (Seasonality)", 
#                             0.01, 0.50, 0.10, 0.01,
#                             help="Smoothing factor for seasonality\n• Typical: 0.1-0.2 (very conservative)"
#                         )
                        
#                         seasonal_type = st.selectbox(
#                             "Seasonality Type",
#                             ["Additive", "Multiplicative"],
#                             index=0,
#                             help="• Additive: Constant seasonal variations\n• Multiplicative: Proportional seasonal variations"
#                         )
#                         parametros['seasonal_type'] = 'add' if seasonal_type == "Additive" else 'mul'
            
#             # ==================== ARIMA PARAMETERS ====================
#             if modelo_arima:
#                 st.markdown("---")
#                 st.markdown("**SARIMA:**")
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     p = st.selectbox("p (AR)", [0, 1, 2, 3], 1)
#                 with col2:
#                     d = st.selectbox("d (I)", [0, 1, 2], 1)
#                 with col3:
#                     q = st.selectbox("q (MA)", [0, 1, 2, 3], 1)
#                 parametros['arima_params'] = (p, d, q)
        
#         st.markdown("---")       
        
#         return uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros, usar_preestablecidos


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
#         st.info(f"📅 **Forecast start date:** {forecast_start.strftime('%B %d, %Y')}")
    
#     # DataFrame metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         df = dataframes['main']
#         st.metric("📊 Main (Products)", f"{df.shape[0]}")
#     with col2:
#         df = dataframes['logics']
#     with col3:
#         df = dataframes['relations']
    
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


# # def extraer_valores_unicos(df_resultado, customer_col, product_code_col):
# #     """Extract unique values for filters"""
    
# #     customers = []
# #     product_codes = []
    
# #     if customer_col and customer_col in df_resultado.columns:
# #         customers = sorted([str(x) for x in df_resultado[customer_col].dropna().unique().tolist()])
    
# #     if product_code_col and product_code_col in df_resultado.columns:
# #         product_codes = sorted([str(x) for x in df_resultado[product_col].dropna().unique().tolist()])
    
# #     return customers, product_codes

# def extraer_valores_unicos(df_resultado, customer_col, product_code_col):
#     """Extract unique values for filters"""
    
#     customers = []
#     product_codes = []
    
#     if customer_col and customer_col in df_resultado.columns:
#         customers = sorted([str(x) for x in df_resultado[customer_col].dropna().unique().tolist()])
    
#     if product_code_col and product_code_col in df_resultado.columns:
#         # ✅ CORREGIDO: product_code_col (antes decía product_col)
#         product_codes = sorted([str(x) for x in df_resultado[product_code_col].dropna().unique().tolist()])
    
#     return customers, product_codes


# def aplicar_filtros(df: pd.DataFrame, customer_col: str, model_col: str, 
#                    selected_customers: list, selected_models: list) -> pd.DataFrame:
#     """Apply filters to dataframe"""
    
#     df_filtered = df.copy()
    
#     if selected_customers and len(selected_customers) > 0 and customer_col and customer_col in df_filtered.columns:
#         df_filtered = df_filtered[df_filtered[customer_col].astype(str).isin(selected_customers)]
    
#     if selected_models and len(selected_models) > 0 and model_col and model_col in df_filtered.columns:
#         df_filtered = df_filtered[df_filtered[model_col].astype(str).isin(selected_models)]
    
#     return df_filtered


# def crear_grafico_consolidado(resultados: dict, df_filtered_dict: dict):
#     """Create consolidated interactive chart with all forecast models"""
    
#     if not PLOTLY_AVAILABLE:
#         st.warning("⚠️ Plotly not available for charts. Please install: pip install plotly")
#         return
    
#     fig = go.Figure()
    
#     colors = {
#         'media_movil': '#FF6B6B',
#         'suavizacao_exponencial': '#4ECDC4',
#         'arima': '#95E1D3'
#     }
    
#     names = {
#         'media_movil': 'Moving average',
#         'suavizacao_exponencial': 'Exponential smoothing',
#         'arima': 'SARIMA'
#     }
    
#     has_data = False
    
#     for modelo_key, df_filtered in df_filtered_dict.items():
        
#         if df_filtered.empty:
#             continue
        
#         date_cols = [col for col in df_filtered.columns if isinstance(col, datetime)]
        
#         if not date_cols:
#             continue
        
#         forecast_summary = df_filtered[date_cols].sum()
        
#         if len(forecast_summary) == 0 or forecast_summary.sum() == 0:
#             continue
        
#         has_data = True
        
#         fig.add_trace(go.Scatter(
#             x=[d.strftime('%Y-%m') for d in forecast_summary.index],
#             y=forecast_summary.values,
#             mode='lines+markers',
#             name=names.get(modelo_key, modelo_key),
#             line=dict(color=colors.get(modelo_key, '#999999'), width=3),
#             marker=dict(size=8),
#             visible=True,
#             hovertemplate='<b>%{fullData.name}</b><br>' +
#                          'Month: %{x}<br>' +
#                          'Units: %{y:,.0f}<br>' +
#                          '<extra></extra>'
#         ))
    
#     if not has_data:
#         st.warning("⚠️ No data available to display in chart. Try adjusting filters or check if models generated results.")
#         return
    
#     fig.update_layout(
#         title={
#             'text': "",
#             'x': 0.5,
#             'xanchor': 'center',
#             'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial Black'}
#         },
#         xaxis_title="Month",
#         yaxis_title="Total Units Forecasted",
#         hovermode='x unified',
#         height=550,
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1,
#             bgcolor="rgba(255, 255, 255, 0.9)",
#             bordercolor="Gray",
#             borderwidth=2,
#             font=dict(size=12)
#         ),
#         plot_bgcolor='rgba(240, 242, 246, 0.5)',
#         xaxis=dict(
#             showgrid=True, 
#             gridwidth=1, 
#             gridcolor='LightGray',
#             tickangle=-45
#         ),
#         yaxis=dict(
#             showgrid=True, 
#             gridwidth=1, 
#             gridcolor='LightGray',
#             zeroline=True,
#             zerolinewidth=2,
#             zerolinecolor='LightGray'
#         ),
#         font=dict(family="Arial", size=12)
#     )
    
#     st.plotly_chart(fig, use_container_width=True)


# def mostrar_resultados(resultados):
#     """Display forecasting results with global filters and consolidated chart"""
    
#     st.header("📊 Forecast results")
    
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
    
#     first_df = None
#     customer_col = None
#     model_col = None
    
#     for resultado in resultados.values():
#         if 'dataframe' in resultado and not resultado['dataframe'].empty:
#             first_df = resultado['dataframe']
#             break
    
#     if first_df is not None:
#         for col in first_df.columns:
#             col_str = str(col).lower()
#             if 'id' in col_str and 'customer' in col_str:
#                 customer_col = col
#             if 'product' in col_str and 'model' in col_str:
#                 model_col = col
    
#     customers_list = []
#     models_list = []
    
#     if first_df is not None:
#         customers_list, models_list = extraer_valores_unicos(first_df, customer_col, model_col)
    
#     st.markdown("---")
#     st.subheader("🔍 Global filters")
#     st.markdown("*Apply filters to all models and update the consolidated chart*")
    
#     filter_col1, filter_col2 = st.columns(2)
    
#     with filter_col1:
#         if len(customers_list) > 0:
#             selected_customers = st.multiselect(
#                 "Filter by ID Customer",
#                 options=customers_list,
#                 default=[],
#                 key="global_customer_filter",
#                 help="Select one or more customers (empty = show all customers)"
#             )
#         else:
#             selected_customers = []
#             st.info("ℹ️ No customers found in results")
    
#     with filter_col2:
#         if len(models_list) > 0:
#             selected_models = st.multiselect(
#                 "Filter by Product Model",
#                 options=models_list,
#                 default=[],
#                 key="global_model_filter",
#                 help="Select one or more product models (empty = show all models)"
#             )
#         else:
#             selected_models = []
#             st.info("ℹ️ No product models found in results")
    
#     df_filtered_dict = {}
#     total_products_filtered = 0
#     total_products_original = 0
    
#     for modelo_key, resultado in resultados.items():
#         if 'dataframe' in resultado and not resultado['dataframe'].empty:
#             df_resultado = resultado['dataframe']
#             df_filtered = aplicar_filtros(df_resultado, customer_col, model_col, 
#                                          selected_customers, selected_models)
#             df_filtered_dict[modelo_key] = df_filtered
            
#             total_products_original = len(df_resultado)
#             total_products_filtered = len(df_filtered)
    
#     if len(selected_customers) > 0 or len(selected_models) > 0:
#         filter_info = []
#         if len(selected_customers) > 0:
#             filter_info.append(f"**{len(selected_customers)}** customer(s)")
#         if len(selected_models) > 0:
#             filter_info.append(f"**{len(selected_models)}** product model(s)")
        
#         st.success(f"✅ **Active Filters:** {' + '.join(filter_info)} | Showing **{total_products_filtered:,}** of **{total_products_original:,}** products")
#     else:
#         st.info(f"ℹ️ **No filters applied** - Showing all **{total_products_original:,}** products")
    
#     st.markdown("---")
#     st.subheader("📈 Consolidated Forecast Chart - All Models")
    
#     if len(df_filtered_dict) > 0:
#         crear_grafico_consolidado(resultados, df_filtered_dict)
#     else:
#         st.warning("⚠️ No data available for chart")
    
#     st.markdown("---")
#     st.subheader("📋 Detailed results by model")
    
#     tab_names = []
#     for modelo in resultados.keys():
#         if modelo == 'media_movil':
#             tab_names.append("Moving average")
#         elif modelo == 'suavizacao_exponencial':
#             tab_names.append("Exponential smoothing")
#         elif modelo == 'arima':
#             tab_names.append("SARIMA")
    
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
#             df_filtered = df_filtered_dict.get(modelo_key, df_resultado)
            
#             st.subheader("📊 Results table")
            
#             if not df_filtered.empty:
#                 if df_filtered.shape[1] > 20:
#                     cols_to_show = list(df_filtered.columns[:8]) + list(df_filtered.columns[-18:])
#                     df_display = df_filtered[cols_to_show]
#                 else:
#                     df_display = df_filtered
                
#                 st.dataframe(df_display.head(100), use_container_width=True)
#                 st.info(f"📊 Showing **{len(df_filtered):,}** products (Total: **{len(df_resultado):,}**)")
#             else:
#                 st.warning("⚠️ No data matches the selected filters")
            
#             st.markdown("---")
#             st.subheader("💾 Export Data")
            
#             df_to_export = df_filtered if not df_filtered.empty else df_resultado
            
#             col_download1, col_download2 = st.columns(2)
            
#             with col_download1:
#                 csv = df_to_export.to_csv(index=False)
#                 suffix = "_filtered" if len(df_filtered) < len(df_resultado) else "_full"
#                 st.download_button(
#                     label="📥 Download CSV",
#                     data=csv,
#                     file_name=f"forecast_{modelo_key}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
#                     mime="text/csv",
#                     use_container_width=True
#                 )
            
#             with col_download2:
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
#                 except Exception as e:
#                     st.error(f"Error creating Excel file: {str(e)}")


# def mostrar_pantalla_bienvenida():
#     """Display welcome screen when no file is loaded"""
    
#     st.info("📤 **Please load the consolidated Excel file to begin.**")
    
#     st.markdown("### 📋 Required file structure:")
    
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
        
#     st.markdown("---")


# def main():
#     """Main application function"""
    
#     # Get sidebar configurations (NOW includes usar_preestablecidos)
#     uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros, usar_preestablecidos = configurar_sidebar()
    
#     if not uploaded_file:
#         mostrar_pantalla_bienvenida()
#         st.session_state.resultados = None
#         st.session_state.data_handler = None
#         st.session_state.file_loaded = False
#         return
    
#     if not st.session_state.file_loaded or st.session_state.data_handler is None:
#         with st.spinner("🔄 Loading consolidated file..."):
#             data_handler = DataHandler(uploaded_file)
            
#             if not data_handler.cargar_archivo():
#                 for error in data_handler.obtener_errores():
#                     st.error(error)
#                 return
            
#             st.session_state.data_handler = data_handler
#             st.session_state.file_loaded = True
    
#     data_handler = st.session_state.data_handler
    
#     mostrar_info_carga(data_handler)
    
#     # Validation: Only check if models selected when NOT using presets
#     if not usar_preestablecidos and not any([modelo_media, modelo_suavizacao, modelo_arima]):
#         st.warning("⚠️ Please select at least one model to execute.")
#         return
    
#     if st.button("Execute Forecast", type="primary", use_container_width=True):
        
#         with st.spinner("Processing models... This may take a few minutes."):
#             try:
#                 processor = ForecastProcessor(
#                     data_handler.obtener_dataframes(),
#                     data_handler.obtener_fecha_inicio(),
#                     parametros
#                 )
                
#                 # CONDITIONAL EXECUTION: Preset vs Manual
#                 if usar_preestablecidos:
#                     # PRESET MODE (TODO: Implement in next step)
#                     st.info("Using preset models...")
#                     st.warning("⚠️ Preset execution not yet implemented - Coming soon!")
#                     resultados = None
                    
#                 else:
#                     # MANUAL MODE (Original flow)
#                     modelos_ejecutar = {
#                         'media_movil': modelo_media,
#                         'suavizacao_exponencial': modelo_suavizacao,
#                         'arima': modelo_arima
#                     }
                    
#                     resultados = processor.ejecutar_forecast(modelos_ejecutar)
                
#                 if resultados:
#                     st.session_state.resultados = resultados
#                 else:
#                     if not usar_preestablecidos:
#                         st.error("❌ Could not generate results. Please verify input data.")
                    
#             except Exception as e:
#                 st.error(f"❌ Error during processing: {str(e)}")
#                 st.exception(e)
    
#     if st.session_state.resultados is not None: 
#         mostrar_resultados(st.session_state.resultados)


# if __name__ == "__main__":
#     main()







# """
# Main User Interface - Streamlit App with Filters and Consolidated Chart
# UPDATED: Added preset models functionality with consolidated results
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
#     page_title="Rolling Collection Tool",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'resultados' not in st.session_state:
#     st.session_state.resultados = None
# if 'data_handler' not in st.session_state:
#     st.session_state.data_handler = None
# if 'file_loaded' not in st.session_state:
#     st.session_state.file_loaded = False
# if 'usar_preestablecidos' not in st.session_state:
#     st.session_state.usar_preestablecidos = True

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #279c56;
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
#         padding: 1.5rem;
#         border-radius: 0.5rem;
#         margin: 1.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main header
# st.markdown('<h1 class="main-header">Rolling Collection Tool</h1>', unsafe_allow_html=True)
# st.markdown("---")


# def configurar_sidebar():
#     """Configure sidebar with all controls"""
    
#     with st.sidebar:
        
#         # File upload
#         st.subheader("📁 Load file")
#         uploaded_file = st.file_uploader(
#             "Consolidated Excel File", 
#             type=['xlsx'],
#             help="Excel file with sheets: Main, LogicsxMonth, Relations, and Models",
#             key="file_uploader"
#         )
        
#         st.markdown("---")
        
#         # ==================== PRESET MODELS SECTION ====================
#         st.subheader("Model selection mode")
        
#         # ALWAYS show this checkbox (assume Models sheet exists)
#         usar_preestablecidos = st.checkbox(
#             "Use preset models",
#             value=True,  # ACTIVE BY DEFAULT
#             help="Use best model per product from 'Models' sheet",
#             key="usar_preestablecidos_checkbox"
#         )
        
#         st.markdown("---")
        
#         # ==================== MANUAL MODEL SELECTION ====================
#         st.subheader("🔧 Models to execute")
        
#         # Conditional behavior based on preset checkbox
#         if usar_preestablecidos:
#             # DISABLED when using presets
#             modelo_media = st.checkbox(
#                 "Moving average", 
#                 value=False,
#                 disabled=True,
#                 help="Disabled - Using preset models"
#             )
#             modelo_suavizacao = st.checkbox(
#                 "Exponential smoothing", 
#                 value=False,
#                 disabled=True,
#                 help="Disabled - Using preset models"
#             )
#             modelo_arima = st.checkbox(
#                 "SARIMA", 
#                 value=False,
#                 disabled=True,
#                 help="Disabled - Using preset models"
#             )
#         else:
#             # ENABLED when NOT using presets (ORIGINAL FLOW)
#             modelo_media = st.checkbox(
#                 "Moving average", 
#                 value=True, 
#                 help="Seasonal moving average with growth factors"
#             )
#             modelo_suavizacao = st.checkbox(
#                 "Exponential smoothing", 
#                 value=True,
#                 help="Holt-Winters exponential smoothing"
#             )
#             modelo_arima = st.checkbox(
#                 "SARIMA", 
#                 value=False,
#                 help="Seasonal ARIMA time series model"
#             )
        
#         # ==================== PARAMETERS ====================
#         st.markdown("---")
#         st.subheader("⚙️ Parameters")
        
#         parametros = {}
        
#         # ONLY show parameters if NOT using presets
#         if not usar_preestablecidos:
            
#             # ==================== EXPONENTIAL SMOOTHING PARAMETERS ====================
#             if modelo_suavizacao:
#                 st.markdown("**Exponential smoothing:**")
                
#                 # Model Type Selection
#                 smoothing_type = st.selectbox(
#                     "Model type",
#                     ["Holt-Winters (Triple)", "Holt's Linear (Double)", "Simple (SES)"],
#                     index=0,
#                     help="• Triple: Level + Trend + Seasonality\n• Double: Level + Trend\n• Simple: Level only"
#                 )
#                 parametros['smoothing_type'] = smoothing_type
                
#                 # Create columns for parameters
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     # Alpha (always present)
#                     parametros['alpha'] = st.slider(
#                         "α (Level)", 
#                         0.01, 0.99, 0.30, 0.01,
#                         help="Smoothing factor for level\n• 0.1-0.3: Smooth (more weight to past)\n• 0.7-0.9: Reactive (more weight to recent)"
#                     )
                    
#                     # Beta (only for Double and Triple)
#                     if smoothing_type in ["Holt's Linear (Double)", "Holt-Winters (Triple)"]:
#                         parametros['beta'] = st.slider(
#                             "β (Trend)", 
#                             0.01, 0.50, 0.10, 0.01,
#                             help="Smoothing factor for trend\n• Typical: 0.1-0.3 (more conservative than alpha)"
#                         )
                
#                 with col2:
#                     # Gamma and Seasonality Type (only for Triple)
#                     if smoothing_type == "Holt-Winters (Triple)":
#                         parametros['gamma'] = st.slider(
#                             "γ (Seasonality)", 
#                             0.01, 0.50, 0.10, 0.01,
#                             help="Smoothing factor for seasonality\n• Typical: 0.1-0.2 (very conservative)"
#                         )
                        
#                         seasonal_type = st.selectbox(
#                             "Seasonality Type",
#                             ["Additive", "Multiplicative"],
#                             index=0,
#                             help="• Additive: Constant seasonal variations\n• Multiplicative: Proportional seasonal variations"
#                         )
#                         parametros['seasonal_type'] = 'add' if seasonal_type == "Additive" else 'mul'
            
#             # ==================== ARIMA PARAMETERS ====================
#             if modelo_arima:
#                 st.markdown("---")
#                 st.markdown("**SARIMA:**")
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     p = st.selectbox("p (AR)", [0, 1, 2, 3], 1)
#                 with col2:
#                     d = st.selectbox("d (I)", [0, 1, 2], 1)
#                 with col3:
#                     q = st.selectbox("q (MA)", [0, 1, 2, 3], 1)
#                 parametros['arima_params'] = (p, d, q)
        
#         st.markdown("---")       
        
#         return uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros, usar_preestablecidos


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
#         st.info(f"📅 **Forecast start date:** {forecast_start.strftime('%B %d, %Y')}")
    
#     # DataFrame metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         df = dataframes['main']
#         st.metric("📊 Main (Products)", f"{df.shape[0]}")
#     with col2:
#         df = dataframes['logics']
#     with col3:
#         df = dataframes['relations']
    
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


# def extraer_valores_unicos(df_resultado, customer_col, product_code_col):
#     """Extract unique values for filters"""
    
#     customers = []
#     product_codes = []
    
#     if customer_col and customer_col in df_resultado.columns:
#         customers = sorted([str(x) for x in df_resultado[customer_col].dropna().unique().tolist()])
    
#     if product_code_col and product_code_col in df_resultado.columns:
#         product_codes = sorted([str(x) for x in df_resultado[product_code_col].dropna().unique().tolist()])
    
#     return customers, product_codes


# def aplicar_filtros(df: pd.DataFrame, customer_col: str, model_col: str, 
#                    selected_customers: list, selected_models: list) -> pd.DataFrame:
#     """Apply filters to dataframe"""
    
#     df_filtered = df.copy()
    
#     if selected_customers and len(selected_customers) > 0 and customer_col and customer_col in df_filtered.columns:
#         df_filtered = df_filtered[df_filtered[customer_col].astype(str).isin(selected_customers)]
    
#     if selected_models and len(selected_models) > 0 and model_col and model_col in df_filtered.columns:
#         df_filtered = df_filtered[df_filtered[model_col].astype(str).isin(selected_models)]
    
#     return df_filtered


# def crear_grafico_consolidado(df_filtered: pd.DataFrame, usar_preestablecidos: bool = False):
#     """Create consolidated interactive chart"""
    
#     if not PLOTLY_AVAILABLE:
#         st.warning("⚠️ Plotly not available for charts. Please install: pip install plotly")
#         return
    
#     if df_filtered.empty:
#         st.warning("⚠️ No data available to display in chart.")
#         return
    
#     # Get date columns
#     date_cols = [col for col in df_filtered.columns if isinstance(col, datetime)]
    
#     if not date_cols:
#         st.warning("⚠️ No date columns found in data.")
#         return
    
#     # Calculate sum per month
#     forecast_summary = df_filtered[date_cols].sum()
    
#     if len(forecast_summary) == 0 or forecast_summary.sum() == 0:
#         st.warning("⚠️ No forecast data available to display.")
#         return
    
#     # Create figure
#     fig = go.Figure()
    
#     color = '#4CAF50' if usar_preestablecidos else '#FF6B6B'
#     name = 'Preset Models (Consolidated)' if usar_preestablecidos else 'Forecast'
    
#     fig.add_trace(go.Scatter(
#         x=[d.strftime('%Y-%m') for d in forecast_summary.index],
#         y=forecast_summary.values,
#         mode='lines+markers',
#         name=name,
#         line=dict(color=color, width=3),
#         marker=dict(size=8),
#         hovertemplate='<b>' + name + '</b><br>' +
#                      'Month: %{x}<br>' +
#                      'Units: %{y:,.0f}<br>' +
#                      '<extra></extra>'
#     ))
    
#     fig.update_layout(
#         title={
#             'text': "",
#             'x': 0.5,
#             'xanchor': 'center',
#             'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial Black'}
#         },
#         xaxis_title="Month",
#         yaxis_title="Total Units Forecasted",
#         hovermode='x unified',
#         height=550,
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1,
#             bgcolor="rgba(255, 255, 255, 0.9)",
#             bordercolor="Gray",
#             borderwidth=2,
#             font=dict(size=12)
#         ),
#         plot_bgcolor='rgba(240, 242, 246, 0.5)',
#         xaxis=dict(
#             showgrid=True, 
#             gridwidth=1, 
#             gridcolor='LightGray',
#             tickangle=-45
#         ),
#         yaxis=dict(
#             showgrid=True, 
#             gridwidth=1, 
#             gridcolor='LightGray',
#             zeroline=True,
#             zerolinewidth=2,
#             zerolinecolor='LightGray'
#         ),
#         font=dict(family="Arial", size=12)
#     )
    
#     st.plotly_chart(fig, use_container_width=True)


# def mostrar_resultados_preset(resultado_consolidado):
#     """Display consolidated results for preset models (single table)"""
    
#     st.header("📊 Forecast results - Preset models")
    
#     df_resultado = resultado_consolidado.get('dataframe')
#     metadata = resultado_consolidado.get('metadata', {})
    
#     if df_resultado is None or df_resultado.empty:
#         st.warning("⚠️ No data generated")
#         return
    
#     # Metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         total_products = metadata.get('total_products', len(df_resultado))
#         st.metric("📝 Total Products", f"{total_products:,}")
#     with col2:
#         processed = metadata.get('processed', 0)
#         st.metric("✅ Processed", f"{processed:,}")
#     with col3:
#         cells_updated = resultado_consolidado.get('celulas_actualizadas', 0)
#         st.metric("📊 Cells Updated", f"{cells_updated:,}")
#     with col4:
#         forecast_months = metadata.get('forecast_months', 18)
#         st.metric("📅 Forecast Horizon", f"{forecast_months} months")
    
#     # Model distribution
#     if 'model_counts' in metadata:
#         st.markdown("---")
#         st.subheader("📊 Model Usage Distribution")
        
#         model_counts = metadata['model_counts']
#         model_names = {
#             'media_movil': 'Media movil',
#             'suavizacao_exponencial': 'HWAS',
#             'arima': 'SARIMA',
#             'sem_historico': 'Sem histórico',
#             'no_model': 'N/A (No model configured)'
#         }
        
#         col_metrics = st.columns(min(len(model_counts), 5))
#         for i, (model_key, count) in enumerate(model_counts.items()):
#             if count > 0 and i < len(col_metrics):
#                 with col_metrics[i]:
#                     display_name = model_names.get(model_key, model_key)
#                     st.metric(f"🎯 {display_name}", f"{count}")
    
#     # Filters
#     customer_col = None
#     model_col = None
    
#     for col in df_resultado.columns:
#         col_str = str(col).lower()
#         if 'id' in col_str and 'customer' in col_str:
#             customer_col = col
#         if 'product' in col_str and 'model' in col_str:
#             model_col = col
    
#     customers_list, models_list = extraer_valores_unicos(df_resultado, customer_col, model_col)
    
#     st.markdown("---")
#     st.subheader("🔍 Global Filters")
    
#     filter_col1, filter_col2 = st.columns(2)
    
#     with filter_col1:
#         if len(customers_list) > 0:
#             selected_customers = st.multiselect(
#                 "Filter by ID Customer",
#                 options=customers_list,
#                 default=[],
#                 key="preset_customer_filter"
#             )
#         else:
#             selected_customers = []
    
#     with filter_col2:
#         if len(models_list) > 0:
#             selected_models = st.multiselect(
#                 "Filter by Product Model",
#                 options=models_list,
#                 default=[],
#                 key="preset_model_filter"
#             )
#         else:
#             selected_models = []
    
#     df_filtered = aplicar_filtros(df_resultado, customer_col, model_col, selected_customers, selected_models)
    
#     if len(selected_customers) > 0 or len(selected_models) > 0:
#         st.success(f"✅ Showing **{len(df_filtered):,}** of **{len(df_resultado):,}** products")
#     else:
#         st.info(f"ℹ️ Showing all **{len(df_resultado):,}** products")
    
#     # Chart
#     st.markdown("---")
#     st.subheader("📈 Consolidated forecast chart")
#     crear_grafico_consolidado(df_filtered, usar_preestablecidos=True)
    
#     # Table
#     st.markdown("---")
#     st.subheader("📋 Forecast results table")
    
#     if not df_filtered.empty:
#         # Show first columns + Model Used + forecast columns
#         cols_to_show = []
#         for i, col in enumerate(df_filtered.columns):
#             if i < 9 or isinstance(col, datetime):  # First 9 cols (includes Model Used) + date columns
#                 cols_to_show.append(col)
        
#         df_display = df_filtered[cols_to_show] if len(cols_to_show) < len(df_filtered.columns) else df_filtered
#         st.dataframe(df_display.head(100), use_container_width=True)
        
#         if len(df_filtered) > 100:
#             st.info(f"📊 Showing first 100 of {len(df_filtered):,} products")
    
#     # Export
#     st.markdown("---")
#     st.subheader("💾 Export Data")
    
#     df_to_export = df_filtered if not df_filtered.empty else df_resultado
    
#     col_download1, col_download2 = st.columns(2)
    
#     with col_download1:
#         csv = df_to_export.to_csv(index=False)
#         suffix = "_filtered" if len(df_filtered) < len(df_resultado) else ""
#         st.download_button(
#             label="📥 Download CSV",
#             data=csv,
#             file_name=f"forecast_preset_models{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
#             mime="text/csv",
#             use_container_width=True
#         )
    
#     with col_download2:
#         try:
#             from io import BytesIO
#             buffer = BytesIO()
#             with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
#                 df_to_export.to_excel(writer, index=False, sheet_name='Forecast')
            
#             st.download_button(
#                 label="📥 Download Excel",
#                 data=buffer.getvalue(),
#                 file_name=f"forecast_preset_models{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                 use_container_width=True
#             )
#         except Exception as e:
#             st.error(f"Error creating Excel file: {str(e)}")


# def mostrar_resultados_manual(resultados):
#     """Display results for manual model selection (original flow with multiple tabs)"""
    
#     st.header("📊 Forecast Results - Manual model selection")
    
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
    
#     first_df = None
#     customer_col = None
#     model_col = None
    
#     for resultado in resultados.values():
#         if 'dataframe' in resultado and not resultado['dataframe'].empty:
#             first_df = resultado['dataframe']
#             break
    
#     if first_df is not None:
#         for col in first_df.columns:
#             col_str = str(col).lower()
#             if 'id' in col_str and 'customer' in col_str:
#                 customer_col = col
#             if 'product' in col_str and 'model' in col_str:
#                 model_col = col
    
#     customers_list = []
#     models_list = []
    
#     if first_df is not None:
#         customers_list, models_list = extraer_valores_unicos(first_df, customer_col, model_col)
    
#     st.markdown("---")
#     st.subheader("🔍 Global filters")
#     st.markdown("*Apply filters to all models and update charts*")
    
#     filter_col1, filter_col2 = st.columns(2)
    
#     with filter_col1:
#         if len(customers_list) > 0:
#             selected_customers = st.multiselect(
#                 "Filter by ID Customer",
#                 options=customers_list,
#                 default=[],
#                 key="manual_customer_filter"
#             )
#         else:
#             selected_customers = []
#             st.info("ℹ️ No customers found in results")
    
#     with filter_col2:
#         if len(models_list) > 0:
#             selected_models = st.multiselect(
#                 "Filter by Product Model",
#                 options=models_list,
#                 default=[],
#                 key="manual_model_filter"
#             )
#         else:
#             selected_models = []
#             st.info("ℹ️ No product models found in results")
    
#     df_filtered_dict = {}
#     total_products_filtered = 0
#     total_products_original = 0
    
#     for modelo_key, resultado in resultados.items():
#         if 'dataframe' in resultado and not resultado['dataframe'].empty:
#             df_resultado = resultado['dataframe']
#             df_filtered = aplicar_filtros(df_resultado, customer_col, model_col, 
#                                          selected_customers, selected_models)
#             df_filtered_dict[modelo_key] = df_filtered
            
#             total_products_original = len(df_resultado)
#             total_products_filtered = len(df_filtered)
    
#     if len(selected_customers) > 0 or len(selected_models) > 0:
#         st.success(f"✅ Showing **{total_products_filtered:,}** of **{total_products_original:,}** products")
#     else:
#         st.info(f"ℹ️ Showing all **{total_products_original:,}** products")
    
#     st.markdown("---")
#     st.subheader("📋 Detailed Results by Model")
    
#     tab_names = []
#     for modelo in resultados.keys():
#         if modelo == 'media_movil':
#             tab_names.append("Moving Average")
#         elif modelo == 'suavizacao_exponencial':
#             tab_names.append("Exponential Smoothing")
#         elif modelo == 'arima':
#             tab_names.append("SARIMA")
    
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
#             df_filtered = df_filtered_dict.get(modelo_key, df_resultado)
            
#             # Chart for this model
#             st.subheader("📈 Forecast Chart")
#             crear_grafico_consolidado(df_filtered, usar_preestablecidos=False)
            
#             st.markdown("---")
#             st.subheader("📊 Results Table")
            
#             if not df_filtered.empty:
#                 if df_filtered.shape[1] > 20:
#                     cols_to_show = list(df_filtered.columns[:8]) + list(df_filtered.columns[-18:])
#                     df_display = df_filtered[cols_to_show]
#                 else:
#                     df_display = df_filtered
                
#                 st.dataframe(df_display.head(100), use_container_width=True)
#                 st.info(f"📊 Showing **{len(df_filtered):,}** products (Total: **{len(df_resultado):,}**)")
#             else:
#                 st.warning("⚠️ No data matches the selected filters")
            
#             st.markdown("---")
#             st.subheader("💾 Export Data")
            
#             df_to_export = df_filtered if not df_filtered.empty else df_resultado
            
#             col_download1, col_download2 = st.columns(2)
            
#             with col_download1:
#                 csv = df_to_export.to_csv(index=False)
#                 suffix = "_filtered" if len(df_filtered) < len(df_resultado) else "_full"
#                 st.download_button(
#                     label="📥 Download CSV",
#                     data=csv,
#                     file_name=f"forecast_{modelo_key}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
#                     mime="text/csv",
#                     use_container_width=True
#                 )
            
#             with col_download2:
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
#                 except Exception as e:
#                     st.error(f"Error creating Excel file: {str(e)}")


# def mostrar_pantalla_bienvenida():
#     """Display welcome screen when no file is loaded"""
    
#     st.info("📤 **Please load the consolidated Excel file to begin.**")
    
#     st.markdown("### 📋 Required file structure:")
    
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
#         - Forecast logic rules
#         - Calculation bases
#         - P2P relationships
#         """)
    
#     with col3:
#         st.markdown("""
#         **🔗 Sheet: Relations**
#         - Growth factors by customer
#         - Year-based multipliers
#         """)
    
#     # st.markdown("---")
#     # st.markdown("### 🎯 Optional: Preset Models")
#     # st.markdown("""
#     # Add a **'Models'** sheet to enable automatic model selection per product:
#     # - **Column A:** ID Customer
#     # - **Column B:** Product Model
#     # - **Column C:** Best Model (`Media movil`, `HWAS`, `SARIMA`, `Sem histórico`)
#     # """)


# def main():
#     """Main application function"""
    
#     # Get sidebar configurations
#     uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros, usar_preestablecidos = configurar_sidebar()
    
#     if not uploaded_file:
#         mostrar_pantalla_bienvenida()
#         st.session_state.resultados = None
#         st.session_state.data_handler = None
#         st.session_state.file_loaded = False
#         return
    
#     if not st.session_state.file_loaded or st.session_state.data_handler is None:
#         with st.spinner("🔄 Loading consolidated file..."):
#             data_handler = DataHandler(uploaded_file)
            
#             if not data_handler.cargar_archivo():
#                 for error in data_handler.obtener_errores():
#                     st.error(error)
#                 return
            
#             st.session_state.data_handler = data_handler
#             st.session_state.file_loaded = True
    
#     data_handler = st.session_state.data_handler
    
#     mostrar_info_carga(data_handler)
    
#     # Validation: Only check if models selected when NOT using presets
#     if not usar_preestablecidos and not any([modelo_media, modelo_suavizacao, modelo_arima]):
#         st.warning("⚠️ Please select at least one model to execute.")
#         return
    
#     if st.button("Execute forecast", type="primary", use_container_width=True):
        
#         with st.spinner("Processing models... This may take a few minutes."):
#             try:
#                 processor = ForecastProcessor(
#                     data_handler.obtener_dataframes(),
#                     data_handler.obtener_fecha_inicio(),
#                     parametros
#                 )
                
#                 # CONDITIONAL EXECUTION: Preset vs Manual
#                 if usar_preestablecidos:
#                     # PRESET MODE
#                     df_models = data_handler.obtener_dataframes().get('models')
                    
#                     if df_models is None or df_models.empty:
#                         st.error("❌ Models sheet not found or empty. Please add 'Models' sheet or disable preset mode.")
#                         return
                    
#                     # Execute preset forecast
#                     resultado_consolidado = processor.ejecutar_forecast_preset(df_models)
                    
#                     if resultado_consolidado and 'dataframe' in resultado_consolidado:
#                         # Store as dict with single key
#                         st.session_state.resultados = {'preset_consolidado': resultado_consolidado}
#                         st.session_state.usar_preestablecidos = True
#                         st.success("✅ Preset forecast completed successfully!")
#                     else:
#                         st.error("❌ Could not generate results. Check Models sheet configuration.")
#                         return
                    
#                 else:
#                     # MANUAL MODE (Original flow)
#                     modelos_ejecutar = {
#                         'media_movil': modelo_media,
#                         'suavizacao_exponencial': modelo_suavizacao,
#                         'arima': modelo_arima
#                     }
                    
#                     resultados = processor.ejecutar_forecast(modelos_ejecutar)
                    
#                     if resultados:
#                         st.session_state.resultados = resultados
#                         st.session_state.usar_preestablecidos = False
#                         st.success("✅ Manual forecast completed successfully!")
#                     else:
#                         st.error("❌ Could not generate results. Please verify input data.")
#                         return
                    
#             except Exception as e:
#                 st.error(f"❌ Error during processing: {str(e)}")
#                 st.exception(e)
    
#     # Display results
#     if st.session_state.resultados is not None:
#         if st.session_state.usar_preestablecidos:
#             # Show preset results (single consolidated table)
#             mostrar_resultados_preset(st.session_state.resultados['preset_consolidado'])
#         else:
#             # Show manual results (multiple tabs)
#             mostrar_resultados_manual(st.session_state.resultados)


# if __name__ == "__main__":
#     main()












"""
Main User Interface - Streamlit App with Filters and Consolidated Chart
UPDATED: Added preset models functionality with consolidated results
UPDATED: Manual mode shows 3 simultaneous curves in consolidated chart
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict
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
if 'usar_preestablecidos' not in st.session_state:
    st.session_state.usar_preestablecidos = True

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #279c56;
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
st.markdown("---")


def configurar_sidebar():
    """Configure sidebar with all controls"""
    
    with st.sidebar:
        
        # File upload
        st.subheader("📁 Load file")
        uploaded_file = st.file_uploader(
            "Consolidated Excel File", 
            type=['xlsx'],
            help="Excel file with sheets: Main, LogicsxMonth, Relations, and Models",
            key="file_uploader"
        )
        
        st.markdown("---")
        
        # ==================== PRESET MODELS SECTION ====================
        st.subheader("Model selection mode")
        
        # ALWAYS show this checkbox (assume Models sheet exists)
        usar_preestablecidos = st.checkbox(
            "Use preset models",
            value=True,  # ACTIVE BY DEFAULT
            help="Use best model per product from 'Models' sheet",
            key="usar_preestablecidos_checkbox"
        )
        
        st.markdown("---")
        
        # ==================== MANUAL MODEL SELECTION ====================
        st.subheader("🔧 Models to execute")
        
        # Conditional behavior based on preset checkbox
        if usar_preestablecidos:
            # DISABLED when using presets
            modelo_media = st.checkbox(
                "Moving average", 
                value=False,
                disabled=True,
                help="Disabled - Using preset models"
            )
            modelo_suavizacao = st.checkbox(
                "Exponential smoothing", 
                value=False,
                disabled=True,
                help="Disabled - Using preset models"
            )
            modelo_arima = st.checkbox(
                "SARIMA", 
                value=False,
                disabled=True,
                help="Disabled - Using preset models"
            )
        else:
            # ENABLED when NOT using presets (ORIGINAL FLOW)
            modelo_media = st.checkbox(
                "Moving average", 
                value=True, 
                help="Seasonal moving average with growth factors"
            )
            modelo_suavizacao = st.checkbox(
                "Exponential smoothing", 
                value=True,
                help="Holt-Winters exponential smoothing"
            )
            modelo_arima = st.checkbox(
                "SARIMA", 
                value=True,
                help="Seasonal ARIMA time series model"
            )
        
        # ==================== PARAMETERS ====================
        st.markdown("---")
        st.subheader("⚙️ Parameters")
        
        parametros = {}
        
        # ONLY show parameters if NOT using presets
        if not usar_preestablecidos:
            
            # ==================== EXPONENTIAL SMOOTHING PARAMETERS ====================
            if modelo_suavizacao:
                st.markdown("**Exponential smoothing:**")
                
                # Model Type Selection
                smoothing_type = st.selectbox(
                    "Model type",
                    ["Holt-Winters (Triple)", "Holt's Linear (Double)", "Simple (SES)"],
                    index=0,
                    help="• Triple: Level + Trend + Seasonality\n• Double: Level + Trend\n• Simple: Level only"
                )
                parametros['smoothing_type'] = smoothing_type
                
                # Create columns for parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    # Alpha (always present)
                    parametros['alpha'] = st.slider(
                        "α (Level)", 
                        0.01, 0.99, 0.30, 0.01,
                        help="Smoothing factor for level\n• 0.1-0.3: Smooth (more weight to past)\n• 0.7-0.9: Reactive (more weight to recent)"
                    )
                    
                    # Beta (only for Double and Triple)
                    if smoothing_type in ["Holt's Linear (Double)", "Holt-Winters (Triple)"]:
                        parametros['beta'] = st.slider(
                            "β (Trend)", 
                            0.01, 0.50, 0.10, 0.01,
                            help="Smoothing factor for trend\n• Typical: 0.1-0.3 (more conservative than alpha)"
                        )
                
                with col2:
                    # Gamma and Seasonality Type (only for Triple)
                    if smoothing_type == "Holt-Winters (Triple)":
                        parametros['gamma'] = st.slider(
                            "γ (Seasonality)", 
                            0.01, 0.50, 0.10, 0.01,
                            help="Smoothing factor for seasonality\n• Typical: 0.1-0.2 (very conservative)"
                        )
                        
                        seasonal_type = st.selectbox(
                            "Seasonality Type",
                            ["Additive", "Multiplicative"],
                            index=0,
                            help="• Additive: Constant seasonal variations\n• Multiplicative: Proportional seasonal variations"
                        )
                        parametros['seasonal_type'] = 'add' if seasonal_type == "Additive" else 'mul'
            
            # ==================== ARIMA PARAMETERS ====================
            if modelo_arima:
                st.markdown("---")
                st.markdown("**SARIMA:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.selectbox("p (AR)", [0, 1, 2, 3], 1)
                with col2:
                    d = st.selectbox("d (I)", [0, 1, 2], 1)
                with col3:
                    q = st.selectbox("q (MA)", [0, 1, 2, 3], 1)
                parametros['arima_params'] = (p, d, q)
        
        st.markdown("---")       
        
        return uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros, usar_preestablecidos


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
        st.info(f"📅 **Forecast start date:** {forecast_start.strftime('%B %d, %Y')}")
    
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
    with st.expander("👀 Data preview"):
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


def extraer_valores_unicos(df_resultado, customer_col, product_code_col):
    """Extract unique values for filters"""
    
    customers = []
    product_codes = []
    
    if customer_col and customer_col in df_resultado.columns:
        customers = sorted([str(x) for x in df_resultado[customer_col].dropna().unique().tolist()])
    
    if product_code_col and product_code_col in df_resultado.columns:
        product_codes = sorted([str(x) for x in df_resultado[product_code_col].dropna().unique().tolist()])
    
    return customers, product_codes


def aplicar_filtros(df: pd.DataFrame, customer_col: str, model_col: str, 
                   selected_customers: list, selected_models: list) -> pd.DataFrame:
    """Apply filters to dataframe"""
    
    df_filtered = df.copy()
    
    if selected_customers and len(selected_customers) > 0 and customer_col and customer_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[customer_col].astype(str).isin(selected_customers)]
    
    if selected_models and len(selected_models) > 0 and model_col and model_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[model_col].astype(str).isin(selected_models)]
    
    return df_filtered


def crear_grafico_consolidado(df_filtered: pd.DataFrame, usar_preestablecidos: bool = False):
    """Create consolidated interactive chart (for preset mode - single curve)"""
    
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ Plotly not available for charts. Please install: pip install plotly")
        return
    
    if df_filtered.empty:
        st.warning("⚠️ No data available to display in chart.")
        return
    
    # Get date columns
    date_cols = [col for col in df_filtered.columns if isinstance(col, datetime)]
    
    if not date_cols:
        st.warning("⚠️ No date columns found in data.")
        return
    
    # Calculate sum per month
    forecast_summary = df_filtered[date_cols].sum()
    
    if len(forecast_summary) == 0 or forecast_summary.sum() == 0:
        st.warning("⚠️ No forecast data available to display.")
        return
    
    # Create figure
    fig = go.Figure()
    
    color = '#4CAF50' if usar_preestablecidos else '#FF6B6B'
    name = 'Preset Models (Consolidated)' if usar_preestablecidos else 'Forecast'
    
    fig.add_trace(go.Scatter(
        x=[d.strftime('%Y-%m') for d in forecast_summary.index],
        y=forecast_summary.values,
        mode='lines+markers',
        name=name,
        line=dict(color=color, width=3),
        marker=dict(size=8),
        hovertemplate='<b>' + name + '</b><br>' +
                     'Month: %{x}<br>' +
                     'Units: %{y:,.0f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "",
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


def crear_grafico_3_curvas(df_filtered_dict: Dict[str, pd.DataFrame]):
    """
    Create consolidated chart with 3 simultaneous curves (Manual mode only)
    Shows Moving Average, HWAS, and SARIMA together
    """
    
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ Plotly not available for charts. Please install: pip install plotly")
        return
    
    if not df_filtered_dict:
        st.warning("⚠️ No data available to display in chart.")
        return
    
    fig = go.Figure()
    
    # Define colors and names for each model
    model_config = {
        'media_movil': {
            'name': 'Moving Average',
            'color': '#FF6B6B',
            'visible': True
        },
        'suavizacao_exponencial': {
            'name': 'Exponential Smoothing (HWAS)',
            'color': '#4ECDC4',
            'visible': True
        },
        'arima': {
            'name': 'SARIMA',
            'color': '#95E1D3',
            'visible': True
        }
    }
    
    has_data = False
    
    # Add trace for each model
    for modelo_key, df_filtered in df_filtered_dict.items():
        
        if df_filtered.empty or modelo_key not in model_config:
            continue
        
        # Get date columns
        date_cols = [col for col in df_filtered.columns if isinstance(col, datetime)]
        
        if not date_cols:
            continue
        
        # Calculate sum per month
        forecast_summary = df_filtered[date_cols].sum()
        
        if len(forecast_summary) == 0 or forecast_summary.sum() == 0:
            continue
        
        has_data = True
        config = model_config[modelo_key]
        
        fig.add_trace(go.Scatter(
            x=[d.strftime('%Y-%m') for d in forecast_summary.index],
            y=forecast_summary.values,
            mode='lines+markers',
            name=config['name'],
            line=dict(color=config['color'], width=3),
            marker=dict(size=8),
            visible=config['visible'],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Units: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    if not has_data:
        st.warning("⚠️ No data available to display in chart. Try adjusting filters or check if models generated results.")
        return
    
    fig.update_layout(
        title={
            'text': "",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title="Month",
        yaxis_title="Total Units Forecasted",
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Gray",
            borderwidth=2,
            font=dict(size=13)  # ✅ CORREGIDO: Eliminado weight='bold'
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


def mostrar_resultados_preset(resultado_consolidado):
    """Display consolidated results for preset models (single table)"""
    
    st.header("📊 Forecast results - Preset models")
    
    df_resultado = resultado_consolidado.get('dataframe')
    metadata = resultado_consolidado.get('metadata', {})
    
    if df_resultado is None or df_resultado.empty:
        st.warning("⚠️ No data generated")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # total_products = metadata.get('total_products', len(df_resultado))
        # st.metric("📝 Total Products", f"{total_products:,}")
        forecast_months = metadata.get('forecast_months', 18)
        st.metric("📅 Forecast horizon", f"{forecast_months} months")
    # with col2:
    #     processed = metadata.get('processed', 0)
    #     st.metric("✅ Processed", f"{processed:,}")
    # with col3:
    #     cells_updated = resultado_consolidado.get('celulas_actualizadas', 0)
    #     st.metric("📊 Cells Updated", f"{cells_updated:,}")
    # with col4:
    #     forecast_months = metadata.get('forecast_months', 18)
    #     st.metric("📅 Forecast Horizon", f"{forecast_months} months")
    
    # # Model distribution
    # if 'model_counts' in metadata:
    #     st.markdown("---")
    #     st.subheader("📊 Model Usage Distribution")
        
    #     model_counts = metadata['model_counts']
    #     model_names = {
    #         'media_movil': 'Media movil',
    #         'suavizacao_exponencial': 'HWAS',
    #         'arima': 'SARIMA',
    #         'sem_historico': 'Sem histórico',
    #         'no_model': 'N/A (No model configured)'
    #     }
        
    #     col_metrics = st.columns(min(len(model_counts), 5))
    #     for i, (model_key, count) in enumerate(model_counts.items()):
    #         if count > 0 and i < len(col_metrics):
    #             with col_metrics[i]:
    #                 display_name = model_names.get(model_key, model_key)
    #                 st.metric(f"🎯 {display_name}", f"{count}")
    
    # Filters
    customer_col = None
    model_col = None
    
    for col in df_resultado.columns:
        col_str = str(col).lower()
        if 'id' in col_str and 'customer' in col_str:
            customer_col = col
        if 'product' in col_str and 'model' in col_str:
            model_col = col
    
    customers_list, models_list = extraer_valores_unicos(df_resultado, customer_col, model_col)
    
    st.markdown("---")
    st.subheader("🔍 Global filters")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        if len(customers_list) > 0:
            selected_customers = st.multiselect(
                "Filter by ID Customer",
                options=customers_list,
                default=[],
                key="preset_customer_filter"
            )
        else:
            selected_customers = []
    
    with filter_col2:
        if len(models_list) > 0:
            selected_models = st.multiselect(
                "Filter by Product Model",
                options=models_list,
                default=[],
                key="preset_model_filter"
            )
        else:
            selected_models = []
    
    df_filtered = aplicar_filtros(df_resultado, customer_col, model_col, selected_customers, selected_models)
    
    if len(selected_customers) > 0 or len(selected_models) > 0:
        st.success(f"✅ Showing **{len(df_filtered):,}** of **{len(df_resultado):,}** products")
    else:
        st.info(f"ℹ️ Showing all **{len(df_resultado):,}** products")
    
    # Chart
    st.markdown("---")
    st.subheader("📈 Consolidated forecast chart")
    crear_grafico_consolidado(df_filtered, usar_preestablecidos=True)
    
    # Table
    st.markdown("---")
    st.subheader("📋 Forecast results table")
    
    if not df_filtered.empty:
        # Show first columns + Model Used + forecast columns
        cols_to_show = []
        for i, col in enumerate(df_filtered.columns):
            if i < 9 or isinstance(col, datetime):  # First 9 cols (includes Model Used) + date columns
                cols_to_show.append(col)
        
        df_display = df_filtered[cols_to_show] if len(cols_to_show) < len(df_filtered.columns) else df_filtered
        st.dataframe(df_display.head(100), use_container_width=True)
        
        if len(df_filtered) > 100:
            st.info(f"📊 Showing first 100 of {len(df_filtered):,} products")
    
    # Export
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    df_to_export = df_filtered if not df_filtered.empty else df_resultado
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        csv = df_to_export.to_csv(index=False)
        suffix = "_filtered" if len(df_filtered) < len(df_resultado) else ""
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"forecast_preset_models{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_download2:
        try:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_to_export.to_excel(writer, index=False, sheet_name='Forecast')
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f"forecast_preset_models{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")


def mostrar_resultados_manual(resultados):
    """Display results for manual model selection with 3 simultaneous curves"""
    
    st.header("📊 Forecast Results - Manual model selection")
    
    total_cells = sum([r.get('celulas_actualizadas', 0) for r in resultados.values()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # st.metric("📝 Total Cells Updated", f"{total_cells:,}")
        first_result = list(resultados.values())[0]
        forecast_months = first_result.get('metadata', {}).get('forecast_months', 18)
        st.metric("📅 Forecast horizon", f"{forecast_months} months")
    # with col2:
    #     st.metric("🔢 Models Executed", len(resultados))
    # with col3:
    #     if resultados:
    #         first_result = list(resultados.values())[0]
    #         forecast_months = first_result.get('metadata', {}).get('forecast_months', 18)
    #         st.metric("📅 Forecast Horizon", f"{forecast_months} months")
    
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
            if 'id' in col_str and 'customer' in col_str:
                customer_col = col
            if 'product' in col_str and 'model' in col_str:
                model_col = col
    
    customers_list = []
    models_list = []
    
    if first_df is not None:
        customers_list, models_list = extraer_valores_unicos(first_df, customer_col, model_col)
    
    st.markdown("---")
    st.subheader("🔍 Global filters")
    st.markdown("*Apply filters to all models and update the consolidated chart*")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        if len(customers_list) > 0:
            selected_customers = st.multiselect(
                "Filter by ID Customer",
                options=customers_list,
                default=[],
                key="manual_customer_filter"
            )
        else:
            selected_customers = []
            st.info("ℹ️ No customers found in results")
    
    with filter_col2:
        if len(models_list) > 0:
            selected_models = st.multiselect(
                "Filter by Product Model",
                options=models_list,
                default=[],
                key="manual_model_filter"
            )
        else:
            selected_models = []
            st.info("ℹ️ No product models found in results")
    
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
    
    if len(selected_customers) > 0 or len(selected_models) > 0:
        st.success(f"✅ Showing **{total_products_filtered:,}** of **{total_products_original:,}** products")
    else:
        st.info(f"ℹ️ Showing all **{total_products_original:,}** products")
    
    # CONSOLIDATED CHART WITH 3 CURVES
    st.markdown("---")
    st.subheader("📈 Consolidated forecast chart - All models comparison")
    # st.markdown("*Compare all executed models in a single view. Click legend items to show/hide curves.*")
    
    if len(df_filtered_dict) > 0:
        crear_grafico_3_curvas(df_filtered_dict)
    else:
        st.warning("⚠️ No data available for consolidated chart")
    
    # TABS WITH TABLES AND EXPORT (NO INDIVIDUAL CHARTS)
    st.markdown("---")
    st.subheader("📋 Detailed results by model")
    # st.markdown("*View individual model data tables and download options*")
    
    tab_names = []
    for modelo in resultados.keys():
        if modelo == 'media_movil':
            tab_names.append("Moving average")
        elif modelo == 'suavizacao_exponencial':
            tab_names.append("Exponential smoothing")
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
            
            # NO INDIVIDUAL CHART - Only table and export
            st.subheader("Results table")
            
            if not df_filtered.empty:
                if df_filtered.shape[1] > 20:
                    cols_to_show = list(df_filtered.columns[:8]) + list(df_filtered.columns[-18:])
                    df_display = df_filtered[cols_to_show]
                else:
                    df_display = df_filtered
                
                st.dataframe(df_display.head(100), use_container_width=True)
                st.info(f"Showing 100 products (Total: **{len(df_resultado):,}**)")
            else:
                st.warning("⚠️ No data matches the selected filters")
            
            st.markdown("---")
            st.subheader("💾 Export data")
            
            df_to_export = df_filtered if not df_filtered.empty else df_resultado
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
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
    
    st.markdown("### 📋 Required file structure:")
    
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
    
    with col2:
        st.markdown("""
        **⚙️ Sheet: LogicsxMonth**
        - Forecast logic rules
        - Calculation bases
        - P2P relationships
        """)
    
    with col3:
        st.markdown("""
        **🔗 Sheet: Relations**
        - Growth factors by customer
        - Year-based multipliers
        """)


def main():
    """Main application function"""
    
    # Get sidebar configurations
    uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros, usar_preestablecidos = configurar_sidebar()
    
    if not uploaded_file:
        mostrar_pantalla_bienvenida()
        st.session_state.resultados = None
        st.session_state.data_handler = None
        st.session_state.file_loaded = False
        return
    
    if not st.session_state.file_loaded or st.session_state.data_handler is None:
        with st.spinner("🔄 Loading consolidated file..."):
            data_handler = DataHandler(uploaded_file)
            
            if not data_handler.cargar_archivo():
                for error in data_handler.obtener_errores():
                    st.error(error)
                return
            
            st.session_state.data_handler = data_handler
            st.session_state.file_loaded = True
    
    data_handler = st.session_state.data_handler
    
    mostrar_info_carga(data_handler)
    
    # Validation: Only check if models selected when NOT using presets
    if not usar_preestablecidos and not any([modelo_media, modelo_suavizacao, modelo_arima]):
        st.warning("⚠️ Please select at least one model to execute.")
        return
    
    if st.button("Execute forecast", type="primary", use_container_width=True):
        
        with st.spinner("Processing models... This may take a few minutes."):
            try:
                processor = ForecastProcessor(
                    data_handler.obtener_dataframes(),
                    data_handler.obtener_fecha_inicio(),
                    parametros
                )
                
                # CONDITIONAL EXECUTION: Preset vs Manual
                if usar_preestablecidos:
                    # PRESET MODE
                    df_models = data_handler.obtener_dataframes().get('models')
                    
                    if df_models is None or df_models.empty:
                        st.error("❌ Models sheet not found or empty. Please add 'Models' sheet or disable preset mode.")
                        return
                    
                    # Execute preset forecast
                    resultado_consolidado = processor.ejecutar_forecast_preset(df_models)
                    
                    if resultado_consolidado and 'dataframe' in resultado_consolidado:
                        # Store as dict with single key
                        st.session_state.resultados = {'preset_consolidado': resultado_consolidado}
                        st.session_state.usar_preestablecidos = True
                        st.success("✅ Preset forecast completed successfully!")
                    else:
                        st.error("❌ Could not generate results. Check Models sheet configuration.")
                        return
                    
                else:
                    # MANUAL MODE (Original flow)
                    modelos_ejecutar = {
                        'media_movil': modelo_media,
                        'suavizacao_exponencial': modelo_suavizacao,
                        'arima': modelo_arima
                    }
                    
                    resultados = processor.ejecutar_forecast(modelos_ejecutar)
                    
                    if resultados:
                        st.session_state.resultados = resultados
                        st.session_state.usar_preestablecidos = False
                        st.success("✅ Manual forecast completed successfully!")
                    else:
                        st.error("❌ Could not generate results. Please verify input data.")
                        return
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.resultados is not None:
        if st.session_state.usar_preestablecidos:
            # Show preset results (single consolidated table)
            mostrar_resultados_preset(st.session_state.resultados['preset_consolidado'])
        else:
            # Show manual results (3 curves + tabs with tables)
            mostrar_resultados_manual(st.session_state.resultados)


if __name__ == "__main__":
    main()