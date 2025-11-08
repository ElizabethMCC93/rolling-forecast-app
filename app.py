# """
# Interfaz de usuario principal - Streamlit App
# """
# import streamlit as st
# import pandas as pd
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# from data_handler import DataHandler
# from forecast_processor import ForecastProcessor

# # Configuración de página
# st.set_page_config(
#     page_title="Rolling Forecast Tool",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # CSS personalizado
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
# </style>
# """, unsafe_allow_html=True)

# # Header principal
# st.markdown('<h1 class="main-header">🎯 Rolling Forecast Tool</h1>', unsafe_allow_html=True)
# st.markdown("### 📊 Herramienta con 3 Modelos Estadísticos + Lógica de Lançamento")
# st.markdown("---")


# def configurar_sidebar():
#     """Configura el sidebar con todos los controles"""
    
#     with st.sidebar:
#         st.header("⚙️ Configuración")
        
#         # Upload de archivo
#         st.subheader("📁 Cargar Archivo")
#         uploaded_file = st.file_uploader(
#             "Arquivo Excel Consolidado", 
#             type=['xlsx'],
#             help="Archivo Excel con 3 pestañas: Resumo, LogicasxMes y Relaciones"
#         )
        
#         if uploaded_file is None:
#             st.markdown(
#                 '<div class="warning-box">📋 <strong>Pestañas requeridas:</strong><br>'
#                 '• Resumo<br>• LogicasxMes<br>• Relaciones</div>', 
#                 unsafe_allow_html=True
#             )
        
#         # Configuración de fecha base
#         st.subheader("📅 Configuración Temporal")
#         fecha_base = st.date_input(
#             "Data Base", 
#             datetime.now(),
#             help="Fecha base para el cálculo del forecast"
#         )
        
#         # Selección de modelos
#         st.subheader("🔧 Modelos a Ejecutar")
#         modelo_media = st.checkbox("📈 Media Móvil", True)
#         modelo_suavizacao = st.checkbox("📊 Suavização Exponencial", True)
#         modelo_arima = st.checkbox("🔬 ARIMA", True)
        
#         # Parámetros
#         st.subheader("⚙️ Parámetros")
        
#         parametros = {}
        
#         if modelo_suavizacao:
#             parametros['alpha'] = st.slider(
#                 "Alpha (Suavização)", 
#                 0.1, 0.9, 0.3, 0.1,
#                 help="Factor de suavización (0.1 = más suave, 0.9 = más reactivo)"
#             )
        
#         if modelo_arima:
#             st.write("**Parámetros ARIMA (p,d,q):**")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 p = st.selectbox("p (AR)", [0, 1, 2, 3], 1)
#             with col2:
#                 d = st.selectbox("d (I)", [0, 1, 2], 1)
#             with col3:
#                 q = st.selectbox("q (MA)", [0, 1, 2, 3], 1)
#             parametros['arima_params'] = (p, d, q)
        
#         st.markdown("---")
#         st.info("💡 **Nota:** Todos los modelos usan la misma lógica de lançamento.")
        
#         return uploaded_file, fecha_base, modelo_media, modelo_suavizacao, modelo_arima, parametros


# def mostrar_info_carga(data_handler):
#     """Muestra información de los datos cargados"""
    
#     dataframes = data_handler.obtener_dataframes()
    
#     st.markdown(
#         '<div class="success-box">✅ <strong>Archivo cargado exitosamente!</strong></div>', 
#         unsafe_allow_html=True
#     )
    
#     # Métricas de los dataframes
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         df = dataframes['resumo']
#         st.metric("📊 Resumo", f"{df.shape[0]} × {df.shape[1]}")
#     with col2:
#         df = dataframes['logicas']
#         st.metric("⚙️ LogicasxMes", f"{df.shape[0]} × {df.shape[1]}")
#     with col3:
#         df = dataframes['relaciones']
#         st.metric("🔗 Relaciones", f"{df.shape[0]} × {df.shape[1]}")
    
#     # Preview de datos
#     with st.expander("👀 Preview dos Dados"):
#         tab1, tab2, tab3 = st.tabs(["📊 Resumo", "⚙️ LogicasxMes", "🔗 Relaciones"])
        
#         with tab1:
#             df = dataframes['resumo']
#             st.write(f"**Total de columnas:** {df.shape[1]}")
#             st.dataframe(df.head(10), use_container_width=True)
        
#         with tab2:
#             df = dataframes['logicas']
#             st.write(f"**Total de columnas:** {df.shape[1]}")
#             st.dataframe(df.head(10), use_container_width=True)
        
#         with tab3:
#             df = dataframes['relaciones']
#             st.write(f"**Total de columnas:** {df.shape[1]}")
#             st.dataframe(df.head(10), use_container_width=True)


# def mostrar_resultados(resultados):
#     """Muestra los resultados del forecasting"""
    
#     st.header("📊 Resultados del Forecast")
    
#     # Resumen general
#     total_celulas = sum([r.get('celulas_actualizadas', 0) for r in resultados.values()])
#     st.metric("📝 Total de Células Actualizadas", total_celulas)
    
#     # Tabs para cada modelo
#     tab_names = []
#     for modelo in resultados.keys():
#         if modelo == 'media_movil':
#             tab_names.append("📈 Media Móvil")
#         elif modelo == 'suavizacao_exponencial':
#             tab_names.append("📊 Suavización Exp.")
#         elif modelo == 'arima':
#             tab_names.append("🔬 ARIMA")
    
#     if not tab_names:
#         st.warning("No se generaron resultados.")
#         return
    
#     tabs = st.tabs(tab_names)
    
#     for i, (modelo_key, resultado) in enumerate(resultados.items()):
#         with tabs[i]:
#             col1, col2 = st.columns([3, 1])
            
#             with col1:
#                 # Aquí irían los gráficos
#                 st.subheader("📈 Visualización")
#                 if 'dataframe' in resultado and not resultado['dataframe'].empty:
#                     st.dataframe(resultado['dataframe'].head(20), use_container_width=True)
#                 else:
#                     st.info("No hay datos para visualizar")
            
#             with col2:
#                 # Métricas
#                 st.subheader("📊 Métricas")
                
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h4>Células Actualizadas</h4>
#                     <h2>{resultado.get('celulas_actualizadas', 0)}</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 if 'parametros' in resultado:
#                     st.subheader("⚙️ Parámetros")
#                     for param, valor in resultado['parametros'].items():
#                         st.write(f"**{param}:** {valor}")
                
#                 # Botón de descarga
#                 if 'dataframe' in resultado and not resultado['dataframe'].empty:
#                     csv = resultado['dataframe'].to_csv(index=False)
#                     st.download_button(
#                         label="💾 Download CSV",
#                         data=csv,
#                         file_name=f"forecast_{modelo_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
#                         mime="text/csv",
#                         use_container_width=True
#                     )


# def mostrar_pantalla_bienvenida():
#     """Muestra la pantalla cuando no hay archivo cargado"""
    
#     st.info("📤 **Por favor, carga el archivo Excel consolidado para comenzar.**")
    
#     st.markdown("### 📋 Estructura del Archivo Requerido:")
#     st.markdown("El archivo Excel debe contener **3 pestañas** con los siguientes nombres exactos:")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         **📊 Pestaña: Resumo**
#         - Datos principales de productos
#         - Columnas de fechas con valores históricos
#         - Información de clientes y clases
#         """)
    
#     with col2:
#         st.markdown("""
#         **⚙️ Pestaña: LogicasxMes**
#         - Lógicas por clase y mes
#         - Configuración de forecast
#         - Reglas de lançamento
#         """)
    
#     with col3:
#         st.markdown("""
#         **🔗 Pestaña: Relaciones**
#         - Factores por cliente
#         - Multiplicadores por año
#         - Ajustes de crecimiento
#         """)
    
#     st.markdown("---")
#     st.markdown("""
#     <div class="warning-box">
#         <strong>⚠️ Importante:</strong><br>
#         • Los nombres de las pestañas deben ser <strong>exactamente</strong>: Resumo, LogicasxMes, Relaciones<br>
#         • El archivo debe estar en formato <strong>.xlsx</strong><br>
#         • Todas las pestañas deben contener datos
#     </div>
#     """, unsafe_allow_html=True)


# def main():
#     """Función principal de la aplicación"""
    
#     # Configurar sidebar y obtener parámetros
#     uploaded_file, fecha_base, modelo_media, modelo_suavizacao, modelo_arima, parametros = configurar_sidebar()
    
#     # Si no hay archivo, mostrar pantalla de bienvenida
#     if not uploaded_file:
#         mostrar_pantalla_bienvenida()
#         return
    
#     # Cargar datos
#     with st.spinner("🔄 Cargando archivo consolidado..."):
#         data_handler = DataHandler(uploaded_file)
        
#         if not data_handler.cargar_archivo():
#             # Mostrar errores
#             for error in data_handler.obtener_errores():
#                 st.error(error)
#             return
    
#     # Mostrar información de carga
#     mostrar_info_carga(data_handler)
    
#     # Validar que al menos un modelo esté seleccionado
#     if not any([modelo_media, modelo_suavizacao, modelo_arima]):
#         st.warning("⚠️ Por favor, selecciona al menos un modelo para ejecutar.")
#         return
    
#     # Botón para ejecutar forecast
#     if st.button("🚀 Executar Forecast", type="primary", use_container_width=True):
        
#         with st.spinner("Procesando modelos... Esto puede tomar unos minutos."):
#             try:
#                 # Crear procesador
#                 processor = ForecastProcessor(
#                     data_handler.obtener_dataframes(),
#                     fecha_base,
#                     parametros
#                 )
                
#                 # Ejecutar modelos seleccionados
#                 modelos_ejecutar = {
#                     'media_movil': modelo_media,
#                     'suavizacao_exponencial': modelo_suavizacao,
#                     'arima': modelo_arima
#                 }
                
#                 resultados = processor.ejecutar_forecast(modelos_ejecutar)
                
#                 if resultados:
#                     mostrar_resultados(resultados)
#                 else:
#                     st.error("❌ No se pudieron generar resultados. Verifica los datos de entrada.")
                    
#             except Exception as e:
#                 st.error(f"❌ Error durante el procesamiento: {str(e)}")
#                 st.exception(e)


# if __name__ == "__main__":
#     main()








"""
Main User Interface - Streamlit App (English version)
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
    page_title="Rolling Forecast Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎯 Rolling Forecast Tool</h1>', unsafe_allow_html=True)
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
            help="Excel file with 3 sheets: Main, LogicsxMonth, and Relations"
        )
        
        if uploaded_file is None:
            st.markdown(
                '<div class="warning-box">📋 <strong>Required sheets:</strong><br>'
                '• Main (sales data)<br>• LogicsxMonth (logic rules)<br>• Relations (growth factors)</div>', 
                unsafe_allow_html=True
            )
        
        # Model selection
        st.subheader("🔧 Models to Execute")
        modelo_media = st.checkbox("📈 Moving Average", True, 
                                    help="Simple moving average with growth factors")
        modelo_suavizacao = st.checkbox("📊 Exponential Smoothing", True,
                                        help="Exponential smoothing forecast")
        modelo_arima = st.checkbox("🔬 ARIMA", True,
                                   help="ARIMA time series model")
        
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
        
        # Information box
        st.markdown(
            '<div class="info-box">'
            '<strong>💡 How it works:</strong><br>'
            '• 18-month forecast from start date (B2)<br>'
            '• Applies calculation logic per class/month<br>'
            '• Growth factors from Relations sheet<br>'
            '• Supports P2P (previous model) logic'
            '</div>', 
            unsafe_allow_html=True
        )
        
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
        st.metric("📊 Main (Products)", f"{df.shape[0]} rows × {df.shape[1]} cols")
    with col2:
        df = dataframes['logics']
        st.metric("⚙️ LogicsxMonth", f"{df.shape[0]} rules")
    with col3:
        df = dataframes['relations']
        st.metric("🔗 Relations", f"{df.shape[0]} customers")
    
    # Data preview
    with st.expander("👀 Data Preview"):
        tab1, tab2, tab3 = st.tabs(["📊 Main", "⚙️ LogicsxMonth", "🔗 Relations"])
        
        with tab1:
            df = dataframes['main']
            st.write(f"**Columns:** {df.shape[1]} | **Products:** {df.shape[0]}")
            # Show only first columns and first rows
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


def mostrar_resultados(resultados):
    """Display forecasting results"""
    
    st.header("📊 Forecast Results")
    
    # General summary
    total_cells = sum([r.get('celulas_actualizadas', 0) for r in resultados.values()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Total Cells Updated", f"{total_cells:,}")
    with col2:
        st.metric("🔢 Models Executed", len(resultados))
    with col3:
        if resultados:
            first_result = list(resultados.values())[0]
            forecast_months = first_result.get('metadata', {}).get('forecast_months', 18)
            st.metric("📅 Forecast Horizon", f"{forecast_months} months")
    
    # Tabs for each model
    tab_names = []
    for modelo in resultados.keys():
        if modelo == 'media_movil':
            tab_names.append("📈 Moving Average")
        elif modelo == 'suavizacao_exponencial':
            tab_names.append("📊 Exponential Smoothing")
        elif modelo == 'arima':
            tab_names.append("🔬 ARIMA")
    
    if not tab_names:
        st.warning("No results generated.")
        return
    
    tabs = st.tabs(tab_names)
    
    for i, (modelo_key, resultado) in enumerate(resultados.items()):
        with tabs[i]:
            
            if 'error' in resultado.get('metadata', {}):
                st.error("❌ Error processing this model")
                continue
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📊 Results Table")
                
                if 'dataframe' in resultado and not resultado['dataframe'].empty:
                    df_display = resultado['dataframe']
                    
                    # Show first 20 rows and most relevant columns
                    if df_display.shape[1] > 20:
                        # Show first 8 columns + last 10 columns (forecast)
                        cols_to_show = list(df_display.columns[:8]) + list(df_display.columns[-10:])
                        df_display = df_display[cols_to_show]
                    
                    st.dataframe(df_display.head(20), use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📈 Forecast Summary")
                    
                    # Calculate total forecast by month
                    date_cols = [col for col in resultado['dataframe'].columns if isinstance(col, datetime)]
                    if date_cols:
                        forecast_summary = resultado['dataframe'][date_cols].sum()
                        
                        # Create simple chart
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=[d.strftime('%Y-%m') for d in forecast_summary.index],
                                y=forecast_summary.values,
                                mode='lines+markers',
                                name='Total Forecast',
                                line=dict(color='#FF6B6B', width=2),
                                marker=dict(size=6)
                            ))
                            fig.update_layout(
                                title="Total Forecast by Month",
                                xaxis_title="Month",
                                yaxis_title="Units",
                                hovermode='x unified',
                                height=350
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("No data available to display")
            
            with col2:
                # Metrics
                st.subheader("📊 Metrics")
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Cells Updated</h4>
                    <h2>{resultado.get('celulas_actualizadas', 0):,}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Products Processed</h4>
                    <h2>{resultado.get('metadata', {}).get('n_products_processed', 0):,}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                if 'parametros' in resultado:
                    st.subheader("⚙️ Parameters")
                    for param, valor in resultado['parametros'].items():
                        st.write(f"**{param}:** {valor}")
                
                # Download button
                if 'dataframe' in resultado and not resultado['dataframe'].empty:
                    st.subheader("💾 Export")
                    
                    # CSV download
                    csv = resultado['dataframe'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"forecast_{modelo_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Excel download (optional)
                    try:
                        from io import BytesIO
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            resultado['dataframe'].to_excel(writer, index=False, sheet_name='Forecast')
                        
                        st.download_button(
                            label="📥 Download Excel",
                            data=buffer.getvalue(),
                            file_name=f"forecast_{modelo_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except:
                        pass


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
    
    with col2:
        st.markdown("""
        **⚙️ Sheet: LogicsxMonth**
        - Row 2: Column headers
        - Column C: Class
        - Column D: Month (dd/mm/yyyy)
        - Column E: Calculation Base
        - Column G: P2P Model
        - Column H: Launch Month (XF)
        - Data starts from row 3
        """)
    
    with col3:
        st.markdown("""
        **🔗 Sheet: Relations**
        - Row 8: Year headers
        - Column A: ID Customer (from row 9)
        - Columns B+: Growth factors by year
        - Factor values (e.g., 1.05 = 5% growth)
        """)
    
    st.markdown("---")
    
    # Calculation logic explanation
    st.markdown("### 🔍 Calculation Logic:")
    
    st.markdown("""
    <div class="info-box">
    <strong>Calculation Base Options:</strong><br><br>
    
    <strong>1. DE PARA SEGUINTE</strong> (P2P - Previous to Next):<br>
    Uses historical data from the P2P model (column G) for the same customer.<br><br>
    
    <strong>2. Não calcula</strong> (No Calculation):<br>
    Sets forecast to zero or leaves empty.<br><br>
    
    <strong>3. Depende do mês de Lançamento</strong> (Launch Month Dependent):<br>
    Like P2P, but only forecasts from the launch month (column H) forward. 
    Previous months are set to zero.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Notes:</strong><br>
        • Sheet names must be <strong>exactly</strong>: Main, LogicsxMonth, Relations<br>
        • File format must be <strong>.xlsx</strong><br>
        • All sheets must contain data<br>
        • Date format: dd/mm/yyyy (e.g., 01/01/2026)<br>
        • Growth factors in Relations apply only to Moving Average model
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Configure sidebar and get parameters
    uploaded_file, modelo_media, modelo_suavizacao, modelo_arima, parametros = configurar_sidebar()
    
    # If no file, show welcome screen
    if not uploaded_file:
        mostrar_pantalla_bienvenida()
        return
    
    # Load data
    with st.spinner("🔄 Loading consolidated file..."):
        data_handler = DataHandler(uploaded_file)
        
        if not data_handler.cargar_archivo():
            # Show errors
            for error in data_handler.obtener_errores():
                st.error(error)
            return
    
    # Display load information
    mostrar_info_carga(data_handler)
    
    # Validate at least one model is selected
    if not any([modelo_media, modelo_suavizacao, modelo_arima]):
        st.warning("⚠️ Please select at least one model to execute.")
        return
    
    # Execute forecast button
    if st.button("🚀 Execute Forecast", type="primary", use_container_width=True):
        
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
                    mostrar_resultados(resultados)
                else:
                    st.error("❌ Could not generate results. Please verify input data.")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()