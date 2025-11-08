# """
# Motor de procesamiento y forecasting
# Aquí va toda la lógica de negocio y cálculos
# """
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime
# from typing import Dict, Any, List


# class ForecastProcessor:
#     """Clase principal para procesamiento de forecasting"""
    
#     def __init__(self, dataframes: Dict[str, pd.DataFrame], fecha_base: datetime, parametros: Dict[str, Any]):
#         """
#         Inicializa el procesador de forecasting
        
#         Args:
#             dataframes: Diccionario con 'resumo', 'logicas', 'relaciones'
#             fecha_base: Fecha base para el forecast
#             parametros: Parámetros de configuración de modelos
#         """
#         self.df_resumo = dataframes['resumo'].copy()
#         self.df_logicas = dataframes['logicas'].copy()
#         self.df_relaciones = dataframes['relaciones'].copy()
#         self.fecha_base = fecha_base
#         self.parametros = parametros
        
#         # Identificar columnas de fechas
#         self.columnas_fecha = self._identificar_columnas_fecha()
        
#     def _identificar_columnas_fecha(self) -> List:
#         """Identifica columnas que son fechas en el DataFrame de Resumo"""
#         columnas_fecha = []
#         for col in self.df_resumo.columns:
#             if isinstance(col, datetime):
#                 columnas_fecha.append(col)
#         return sorted(columnas_fecha)
    
#     def ejecutar_forecast(self, modelos_ejecutar: Dict[str, bool]) -> Dict[str, Any]:
#         """
#         Ejecuta los modelos de forecasting seleccionados
        
#         Args:
#             modelos_ejecutar: Diccionario con modelos a ejecutar {nombre: True/False}
            
#         Returns:
#             Diccionario con resultados de cada modelo
#         """
#         resultados = {}
        
#         # Preparar datos base
#         datos_preparados = self._preparar_datos()
        
#         # Ejecutar Media Móvil
#         if modelos_ejecutar.get('media_movil', False):
#             with st.spinner("📈 Ejecutando Media Móvil..."):
#                 resultados['media_movil'] = self._ejecutar_media_movil(datos_preparados)
        
#         # Ejecutar Suavização Exponencial
#         if modelos_ejecutar.get('suavizacao_exponencial', False):
#             with st.spinner("📊 Ejecutando Suavização Exponencial..."):
#                 resultados['suavizacao_exponencial'] = self._ejecutar_suavizacao(datos_preparados)
        
#         # Ejecutar ARIMA
#         if modelos_ejecutar.get('arima', False):
#             with st.spinner("🔬 Ejecutando ARIMA..."):
#                 resultados['arima'] = self._ejecutar_arima(datos_preparados)
        
#         return resultados
    
#     def _preparar_datos(self) -> Dict[str, Any]:
#         """
#         Prepara los datos para forecasting
        
#         Returns:
#             Diccionario con datos preparados
#         """
#         # Aquí va la lógica de preparación de datos
#         # Por ahora retornamos estructura básica
        
#         return {
#             'df_resumo': self.df_resumo,
#             'df_logicas': self.df_logicas,
#             'df_relaciones': self.df_relaciones,
#             'columnas_fecha': self.columnas_fecha,
#             'fecha_base': self.fecha_base,
#             'n_productos': len(self.df_resumo),
#             'n_periodos': len(self.columnas_fecha)
#         }
    
#     def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Ejecuta el modelo de Media Móvil
        
#         Args:
#             datos: Datos preparados
            
#         Returns:
#             Resultado del modelo
#         """
#         try:
#             # AQUÍ VA TU LÓGICA DE MEDIA MÓVIL
#             # Por ahora, retornamos estructura de ejemplo
            
#             df_resultado = datos['df_resumo'].copy()
            
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_actualizadas': 0,
#                 'parametros': {
#                     'ventana': 3,
#                     'tipo': 'simple'
#                 },
#                 'metadata': {
#                     'n_productos_procesados': len(df_resultado),
#                     'fecha_ejecucion': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             st.error(f"Error en Media Móvil: {str(e)}")
#             return self._resultado_vacio()
    
#     def _ejecutar_suavizacao(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Ejecuta el modelo de Suavização Exponencial
        
#         Args:
#             datos: Datos preparados
            
#         Returns:
#             Resultado del modelo
#         """
#         try:
#             # AQUÍ VA TU LÓGICA DE SUAVIZAÇÃO
#             alpha = self.parametros.get('alpha', 0.3)
            
#             df_resultado = datos['df_resumo'].copy()
            
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_actualizadas': 0,
#                 'parametros': {
#                     'alpha': alpha
#                 },
#                 'metadata': {
#                     'n_productos_procesados': len(df_resultado),
#                     'fecha_ejecucion': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             st.error(f"Error en Suavização: {str(e)}")
#             return self._resultado_vacio()
    
#     def _ejecutar_arima(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Ejecuta el modelo ARIMA
        
#         Args:
#             datos: Datos preparados
            
#         Returns:
#             Resultado del modelo
#         """
#         try:
#             # AQUÍ VA TU LÓGICA DE ARIMA
#             arima_params = self.parametros.get('arima_params', (1, 1, 1))
            
#             df_resultado = datos['df_resumo'].copy()
            
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_actualizadas': 0,
#                 'parametros': {
#                     'p': arima_params[0],
#                     'd': arima_params[1],
#                     'q': arima_params[2]
#                 },
#                 'metadata': {
#                     'n_productos_procesados': len(df_resultado),
#                     'fecha_ejecucion': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             st.error(f"Error en ARIMA: {str(e)}")
#             return self._resultado_vacio()
    
#     def _resultado_vacio(self) -> Dict[str, Any]:
#         """Retorna estructura de resultado vacío en caso de error"""
#         return {
#             'dataframe': pd.DataFrame(),
#             'celulas_actualizadas': 0,
#             'parametros': {},
#             'metadata': {}
#         }
    
#     # ============================================
#     # MÉTODOS AUXILIARES PARA PROCESAMIENTO
#     # ============================================
    
#     def obtener_serie_temporal(self, fila_index: int) -> pd.Series:
#         """
#         Obtiene la serie temporal de una fila (valores en columnas de fecha)
        
#         Args:
#             fila_index: Índice de la fila
            
#         Returns:
#             Serie temporal con valores numéricos
#         """
#         fila = self.df_resumo.iloc[fila_index]
#         serie = fila[self.columnas_fecha]
        
#         # Convertir a numérico, reemplazando errores con NaN
#         serie = pd.to_numeric(serie, errors='coerce')
        
#         return serie
    
#     def aplicar_logicas_mes(self, clase: str, mes: int) -> Any:
#         """
#         Obtiene la lógica aplicable para una clase y mes
        
#         Args:
#             clase: Clase del producto
#             mes: Mes (1-12)
            
#         Returns:
#             Lógica correspondiente o None
#         """
#         # IMPLEMENTAR según estructura de df_logicas
#         # Por ahora retorna None
#         return None
    
#     def aplicar_factores_relacion(self, cliente: str, año: int) -> float:
#         """
#         Obtiene el factor de relación para un cliente y año
        
#         Args:
#             cliente: Nombre del cliente
#             año: Año
            
#         Returns:
#             Factor multiplicador
#         """
#         # IMPLEMENTAR según estructura de df_relaciones
#         # Por ahora retorna 1.0 (sin cambios)
#         return 1.0









# """
# Forecasting processing engine - Complete business logic
# """
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# from typing import Dict, Any, List, Tuple


# class ForecastProcessor:
#     """Main class for forecasting processing"""
    
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
        
#         # Identify date columns in Main sheet
#         self.date_columns = self._identify_date_columns()
        
#         # Number of months to forecast
#         self.forecast_months = 18
        
#         # Generate forecast dates
#         self.forecast_dates = self._generate_forecast_dates()
        
#         # Column name mapping (Portuguese to English)
#         self.column_mapping = {
#             'ID Cliente': 'ID_Customer',
#             'Modelo do Produto': 'Product_Model',
#             'Classe': 'Class',
#             'Descrição do Modelo do Produto': 'Product_Description',
#             'Base de Cálculo': 'Calculation_Base',
#             'Mês': 'Month',
#             'P2P': 'P2P',
#             'XF': 'Launch_Month'
#         }
    
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
    
#     def ejecutar_forecast(self, modelos_ejecutar: Dict[str, bool]) -> Dict[str, Any]:
#         """
#         Execute selected forecasting models
        
#         Args:
#             modelos_ejecutar: Dictionary with models to execute {name: True/False}
            
#         Returns:
#             Dictionary with results from each model
#         """
#         resultados = {}
        
#         # Prepare base data
#         datos_preparados = self._preparar_datos()
        
#         # Execute Moving Average
#         if modelos_ejecutar.get('media_movil', False):
#             with st.spinner("📈 Executing Moving Average..."):
#                 resultados['media_movil'] = self._ejecutar_media_movil(datos_preparados)
        
#         # Execute Exponential Smoothing
#         if modelos_ejecutar.get('suavizacao_exponencial', False):
#             with st.spinner("📊 Executing Exponential Smoothing..."):
#                 resultados['suavizacao_exponencial'] = self._ejecutar_suavizacao(datos_preparados)
        
#         # Execute ARIMA
#         if modelos_ejecutar.get('arima', False):
#             with st.spinner("🔬 Executing ARIMA..."):
#                 resultados['arima'] = self._ejecutar_arima(datos_preparados)
        
#         return resultados
    
#     def _preparar_datos(self) -> Dict[str, Any]:
#         """
#         Prepare data for forecasting
        
#         Returns:
#             Dictionary with prepared data
#         """
#         # Standardize column names for easier access
#         df_main_clean = self.df_main.copy()
#         df_logics_clean = self.df_logics.copy()
#         df_relations_clean = self.df_relations.copy()
        
#         # Rename columns if they exist with Portuguese names
#         for port_name, eng_name in self.column_mapping.items():
#             if port_name in df_main_clean.columns:
#                 df_main_clean.rename(columns={port_name: eng_name}, inplace=True)
#             if port_name in df_logics_clean.columns:
#                 df_logics_clean.rename(columns={port_name: eng_name}, inplace=True)
#             if port_name in df_relations_clean.columns:
#                 df_relations_clean.rename(columns={port_name: eng_name}, inplace=True)
        
#         return {
#             'df_main': df_main_clean,
#             'df_logics': df_logics_clean,
#             'df_relations': df_relations_clean,
#             'date_columns': self.date_columns,
#             'forecast_dates': self.forecast_dates,
#             'forecast_start_date': self.forecast_start_date,
#             'n_products': len(df_main_clean),
#             'n_periods': len(self.date_columns)
#         }
    
#     def _get_logic_for_product(self, customer: str, product_class: str, month_date: datetime, datos: Dict) -> Tuple[str, str, Any]:
#         """
#         Get applicable logic for a customer, class and month
        
#         Args:
#             customer: Customer ID
#             product_class: Product class
#             month_date: Month to check
#             datos: Prepared data
            
#         Returns:
#             Tuple: (calculation_base, p2p_model, launch_month)
#         """
#         df_logics = datos['df_logics']
        
#         # Filter logics for this class and month
#         month_start = pd.to_datetime(month_date).replace(day=1)
        
#         mask = (
#             (df_logics['Class'] == product_class) &
#             (pd.to_datetime(df_logics['Month']) == month_start)
#         )
        
#         matching_logics = df_logics[mask]
        
#         if len(matching_logics) > 0:
#             logic_row = matching_logics.iloc[0]
#             calc_base = logic_row.get('Calculation_Base', self.CALC_BASE_P2P)
#             p2p_model = logic_row.get('P2P', None)
#             launch_month = logic_row.get('Launch_Month', None)
            
#             return calc_base, p2p_model, launch_month
        
#         # Default: use P2P calculation
#         return self.CALC_BASE_P2P, None, None
    
#     def _get_growth_factor(self, customer: str, year: int, datos: Dict) -> float:
#         """
#         Get growth/decline factor for a customer and year
        
#         Args:
#             customer: Customer ID
#             year: Year
#             datos: Prepared data
            
#         Returns:
#             Growth factor (default 1.0)
#         """
#         df_relations = datos['df_relations']
        
#         # Find customer row
#         if 'ID_Customer' in df_relations.columns:
#             customer_row = df_relations[df_relations['ID_Customer'] == customer]
            
#             if len(customer_row) > 0:
#                 # Check if year column exists
#                 year_col = str(year)
#                 if year_col in df_relations.columns:
#                     factor = customer_row[year_col].iloc[0]
#                     if pd.notna(factor) and isinstance(factor, (int, float)):
#                         return float(factor)
        
#         return 1.0  # Default: no change
    
#     def _calculate_moving_average(self, series: pd.Series, window: int = 3) -> float:
#         """
#         Calculate moving average
        
#         Args:
#             series: Time series
#             window: Window size
            
#         Returns:
#             Moving average value
#         """
#         # Remove NaN values
#         clean_series = series.dropna()
        
#         if len(clean_series) < window:
#             # If not enough data, use simple average
#             return clean_series.mean() if len(clean_series) > 0 else 0.0
        
#         # Calculate moving average of last 'window' periods
#         return clean_series.tail(window).mean()
    
#     def _calculate_exponential_smoothing(self, series: pd.Series, alpha: float = 0.3) -> float:
#         """
#         Calculate exponential smoothing forecast
        
#         Args:
#             series: Time series
#             alpha: Smoothing factor (0-1)
            
#         Returns:
#             Forecasted value
#         """
#         # Remove NaN values
#         clean_series = series.dropna()
        
#         if len(clean_series) == 0:
#             return 0.0
        
#         if len(clean_series) == 1:
#             return clean_series.iloc[0]
        
#         # Initialize with first value
#         forecast = clean_series.iloc[0]
        
#         # Apply exponential smoothing
#         for value in clean_series.iloc[1:]:
#             forecast = alpha * value + (1 - alpha) * forecast
        
#         return forecast
    
#     def _calculate_arima_simple(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> float:
#         """
#         Calculate simple ARIMA forecast
#         Note: This is a simplified version. For production, use statsmodels.
        
#         Args:
#             series: Time series
#             order: ARIMA order (p, d, q)
            
#         Returns:
#             Forecasted value
#         """
#         # Remove NaN values
#         clean_series = series.dropna()
        
#         if len(clean_series) < 3:
#             # Not enough data for ARIMA, fall back to average
#             return clean_series.mean() if len(clean_series) > 0 else 0.0
        
#         # Simplified ARIMA: use weighted average with trend
#         # This is a placeholder - for real ARIMA use statsmodels library
#         recent_values = clean_series.tail(6).values
        
#         # Calculate simple trend
#         if len(recent_values) >= 2:
#             trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
#             base_value = recent_values[-1]
#             return max(0, base_value + trend)
        
#         return recent_values[-1]






#     # ============================================
#     # MÉTODOS DE EJECUCIÓN DE MODELOS
#     # ============================================
    
#     def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute Moving Average model with growth factors
        
#         Args:
#             datos: Prepared data
            
#         Returns:
#             Model result
#         """
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
            
#             cells_updated = 0
#             processed_products = 0
            
#             # Add forecast columns to result
#             for forecast_date in datos['forecast_dates']:
#                 df_result[forecast_date] = np.nan
            
#             # Process each product row
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 # Get product information
#                 customer = row.get('ID_Customer', '')
#                 product_model = row.get('Product_Model', '')
#                 product_class = row.get('Class', '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Process each forecast month
#                 for forecast_date in datos['forecast_dates']:
                    
#                     # Get logic for this class and month
#                     calc_base, p2p_model, launch_month = self._get_logic_for_product(
#                         customer, product_class, forecast_date, datos
#                     )
                    
#                     # Handle "Não calcula" (No calculation)
#                     if calc_base == self.CALC_BASE_NO_CALC:
#                         df_result.loc[idx, forecast_date] = 0
#                         cells_updated += 1
#                         continue
                    
#                     # Handle "Depende do mês de Lançamento" (Launch dependent)
#                     if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 # Before launch: set to 0
#                                 df_result.loc[idx, forecast_date] = 0
#                                 cells_updated += 1
#                                 continue
#                         # After launch: continue with P2P logic
                    
#                     # Handle "DE PARA SEGUINTE" (P2P) or launch-dependent after launch
#                     if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         # Determine which model to use for historical data
#                         if pd.notna(p2p_model) and p2p_model != '':
#                             # Use P2P model data
#                             historical_series = self._get_p2p_series(
#                                 customer, p2p_model, datos
#                             )
#                         else:
#                             # Use same product's historical data
#                             historical_series = self._get_historical_series(idx, datos)
                        
#                         # Calculate moving average
#                         if len(historical_series.dropna()) > 0:
#                             ma_value = self._calculate_moving_average(historical_series, window=3)
                            
#                             # Apply growth factor from Relations
#                             year = forecast_date.year
#                             growth_factor = self._get_growth_factor(customer, year, datos)
                            
#                             forecasted_value = ma_value * growth_factor
                            
#                             df_result.loc[idx, forecast_date] = max(0, forecasted_value)
#                             cells_updated += 1
#                         else:
#                             df_result.loc[idx, forecast_date] = 0
#                             cells_updated += 1
                
#                 processed_products += 1
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'window': 3,
#                     'type': 'simple',
#                     'growth_factors_applied': True
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
    
#     def _ejecutar_suavizacao(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute Exponential Smoothing model
        
#         Args:
#             datos: Prepared data
            
#         Returns:
#             Model result
#         """
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
            
#             alpha = self.parametros.get('alpha', 0.3)
#             cells_updated = 0
#             processed_products = 0
            
#             # Add forecast columns
#             for forecast_date in datos['forecast_dates']:
#                 df_result[forecast_date] = np.nan
            
#             # Process each product row
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get('ID_Customer', '')
#                 product_model = row.get('Product_Model', '')
#                 product_class = row.get('Class', '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Process each forecast month
#                 for forecast_date in datos['forecast_dates']:
                    
#                     calc_base, p2p_model, launch_month = self._get_logic_for_product(
#                         customer, product_class, forecast_date, datos
#                     )
                    
#                     if calc_base == self.CALC_BASE_NO_CALC:
#                         df_result.loc[idx, forecast_date] = 0
#                         cells_updated += 1
#                         continue
                    
#                     if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 df_result.loc[idx, forecast_date] = 0
#                                 cells_updated += 1
#                                 continue
                    
#                     if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         if pd.notna(p2p_model) and p2p_model != '':
#                             historical_series = self._get_p2p_series(customer, p2p_model, datos)
#                         else:
#                             historical_series = self._get_historical_series(idx, datos)
                        
#                         if len(historical_series.dropna()) > 0:
#                             es_value = self._calculate_exponential_smoothing(historical_series, alpha)
#                             df_result.loc[idx, forecast_date] = max(0, es_value)
#                             cells_updated += 1
#                         else:
#                             df_result.loc[idx, forecast_date] = 0
#                             cells_updated += 1
                
#                 processed_products += 1
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'alpha': alpha,
#                     'type': 'simple_exponential_smoothing'
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
    
#     def _ejecutar_arima(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute ARIMA model
        
#         Args:
#             datos: Prepared data
            
#         Returns:
#             Model result
#         """
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
            
#             arima_params = self.parametros.get('arima_params', (1, 1, 1))
#             cells_updated = 0
#             processed_products = 0
            
#             # Add forecast columns
#             for forecast_date in datos['forecast_dates']:
#                 df_result[forecast_date] = np.nan
            
#             # Process each product row
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get('ID_Customer', '')
#                 product_model = row.get('Product_Model', '')
#                 product_class = row.get('Class', '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Process each forecast month
#                 for forecast_date in datos['forecast_dates']:
                    
#                     calc_base, p2p_model, launch_month = self._get_logic_for_product(
#                         customer, product_class, forecast_date, datos
#                     )
                    
#                     if calc_base == self.CALC_BASE_NO_CALC:
#                         df_result.loc[idx, forecast_date] = 0
#                         cells_updated += 1
#                         continue
                    
#                     if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 df_result.loc[idx, forecast_date] = 0
#                                 cells_updated += 1
#                                 continue
                    
#                     if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         if pd.notna(p2p_model) and p2p_model != '':
#                             historical_series = self._get_p2p_series(customer, p2p_model, datos)
#                         else:
#                             historical_series = self._get_historical_series(idx, datos)
                        
#                         if len(historical_series.dropna()) > 0:
#                             arima_value = self._calculate_arima_simple(historical_series, arima_params)
#                             df_result.loc[idx, forecast_date] = max(0, arima_value)
#                             cells_updated += 1
#                         else:
#                             df_result.loc[idx, forecast_date] = 0
#                             cells_updated += 1
                
#                 processed_products += 1
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'p': arima_params[0],
#                     'd': arima_params[1],
#                     'q': arima_params[2],
#                     'type': 'simplified_arima'
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
    
#     def _get_historical_series(self, row_idx: int, datos: Dict) -> pd.Series:
#         """
#         Get historical time series for a product row
        
#         Args:
#             row_idx: Row index
#             datos: Prepared data
            
#         Returns:
#             Time series with numeric values
#         """
#         df_main = datos['df_main']
#         row = df_main.iloc[row_idx]
        
#         # Extract values from date columns
#         series = row[datos['date_columns']]
        
#         # Convert to numeric
#         series = pd.to_numeric(series, errors='coerce')
        
#         return series
    
#     def _get_p2p_series(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
#         """
#         Get historical series from P2P model
        
#         Args:
#             customer: Customer ID
#             p2p_model: P2P model name
#             datos: Prepared data
            
#         Returns:
#             Time series from P2P model
#         """
#         df_main = datos['df_main']
        
#         # Find the P2P model row for this customer
#         mask = (
#             (df_main['ID_Customer'] == customer) &
#             (df_main['Product_Model'] == p2p_model)
#         )
        
#         matching_rows = df_main[mask]
        
#         if len(matching_rows) > 0:
#             # Use first match
#             row = matching_rows.iloc[0]
#             series = row[datos['date_columns']]
#             series = pd.to_numeric(series, errors='coerce')
#             return series
        
#         # If not found, return empty series
#         return pd.Series(dtype=float)
    
#     def _resultado_vacio(self) -> Dict[str, Any]:
#         """Return empty result structure in case of error"""
#         return {
#             'dataframe': pd.DataFrame(),
#             'celulas_actualizadas': 0,
#             'parametros': {},
#             'metadata': {
#                 'error': True
#             }
#         }










"""
Forecasting processing engine - Complete business logic
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Tuple


class ForecastProcessor:
    """Main class for forecasting processing"""
    
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
        """
        Detect actual column names in DataFrames and create mapping
        """
        # Possible column names (English and Portuguese variations)
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
        
        # Debug: Show detected columns
        st.info(f"🔍 Detected column mappings: {len(self.col_names)} columns identified")
        
        # Show in expander
        with st.expander("🔍 Column Detection Details"):
            st.write("**Main sheet columns:**")
            st.write(list(self.df_main.columns[:15]))  # Show first 15
            st.write("**Logics sheet columns:**")
            st.write(list(self.df_logics.columns))
            st.write("**Relations sheet columns:**")
            st.write(list(self.df_relations.columns[:10]))  # Show first 10
            st.write("**Detected mappings:**")
            st.json(self.col_names)
    
    def ejecutar_forecast(self, modelos_ejecutar: Dict[str, bool]) -> Dict[str, Any]:
        """
        Execute selected forecasting models
        
        Args:
            modelos_ejecutar: Dictionary with models to execute {name: True/False}
            
        Returns:
            Dictionary with results from each model
        """
        resultados = {}
        
        # Detect column names first
        self._detect_column_names()
        
        # Prepare base data
        datos_preparados = self._preparar_datos()
        
        # Execute Moving Average
        if modelos_ejecutar.get('media_movil', False):
            with st.spinner("📈 Executing Moving Average..."):
                resultados['media_movil'] = self._ejecutar_media_movil(datos_preparados)
        
        # Execute Exponential Smoothing
        if modelos_ejecutar.get('suavizacao_exponencial', False):
            with st.spinner("📊 Executing Exponential Smoothing..."):
                resultados['suavizacao_exponencial'] = self._ejecutar_suavizacao(datos_preparados)
        
        # Execute ARIMA
        if modelos_ejecutar.get('arima', False):
            with st.spinner("🔬 Executing ARIMA..."):
                resultados['arima'] = self._ejecutar_arima(datos_preparados)
        
        return resultados
    
    def _preparar_datos(self) -> Dict[str, Any]:
        """
        Prepare data for forecasting
        
        Returns:
            Dictionary with prepared data
        """
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
    
    def _get_column_value(self, df: pd.DataFrame, row_or_series, column_key: str, df_name: str):
        """
        Safely get column value using detected column names
        
        Args:
            df: DataFrame
            row_or_series: Row or Series
            column_key: Key in col_names mapping (e.g., 'customer', 'class')
            df_name: Name of dataframe ('main', 'logics', 'relations')
            
        Returns:
            Column value or None
        """
        full_key = f'{df_name}_{column_key}'
        
        if full_key in self.col_names:
            col_name = self.col_names[full_key]
            if col_name in df.columns:
                if isinstance(row_or_series, pd.Series):
                    return row_or_series.get(col_name, None)
                else:
                    return row_or_series[col_name] if col_name in row_or_series else None
        
        return None
    
    def _get_logic_for_product(self, customer: str, product_class: str, month_date: datetime, datos: Dict) -> Tuple[str, str, Any]:
        """
        Get applicable logic for a customer, class and month
        
        Args:
            customer: Customer ID
            product_class: Product class
            month_date: Month to check
            datos: Prepared data
            
        Returns:
            Tuple: (calculation_base, p2p_model, launch_month)
        """
        df_logics = datos['df_logics']
        col_names = datos['col_names']
        
        # Get column names
        class_col = col_names.get('logics_class')
        month_col = col_names.get('logics_month')
        calc_base_col = col_names.get('logics_calc_base')
        p2p_col = col_names.get('logics_p2p')
        launch_col = col_names.get('logics_launch_month')
        
        if not class_col or class_col not in df_logics.columns:
            # If no class column found, return default
            return self.CALC_BASE_P2P, None, None
        
        if not month_col or month_col not in df_logics.columns:
            return self.CALC_BASE_P2P, None, None
        
        # Filter logics for this class and month
        month_start = pd.to_datetime(month_date).replace(day=1)
        
        try:
            mask = (
                (df_logics[class_col] == product_class) &
                (pd.to_datetime(df_logics[month_col]) == month_start)
            )
            
            matching_logics = df_logics[mask]
            
            if len(matching_logics) > 0:
                logic_row = matching_logics.iloc[0]
                
                calc_base = logic_row.get(calc_base_col, self.CALC_BASE_P2P) if calc_base_col else self.CALC_BASE_P2P
                p2p_model = logic_row.get(p2p_col, None) if p2p_col else None
                launch_month = logic_row.get(launch_col, None) if launch_col else None
                
                return calc_base, p2p_model, launch_month
        except Exception as e:
            st.warning(f"Error getting logic: {str(e)}")
        
        # Default: use P2P calculation
        return self.CALC_BASE_P2P, None, None
    
    def _get_growth_factor(self, customer: str, year: int, datos: Dict) -> float:
        """
        Get growth/decline factor for a customer and year
        
        Args:
            customer: Customer ID
            year: Year
            datos: Prepared data
            
        Returns:
            Growth factor (default 1.0)
        """
        df_relations = datos['df_relations']
        col_names = datos['col_names']
        
        customer_col = col_names.get('relations_customer')
        
        if not customer_col or customer_col not in df_relations.columns:
            return 1.0
        
        # Find customer row
        try:
            customer_row = df_relations[df_relations[customer_col] == customer]
            
            if len(customer_row) > 0:
                # Check if year column exists
                year_col = str(year)
                if year_col in df_relations.columns:
                    factor = customer_row[year_col].iloc[0]
                    if pd.notna(factor) and isinstance(factor, (int, float)):
                        return float(factor)
        except Exception as e:
            pass
        
        return 1.0  # Default: no change
    
    def _calculate_moving_average(self, series: pd.Series, window: int = 3) -> float:
        """
        Calculate moving average
        
        Args:
            series: Time series
            window: Window size
            
        Returns:
            Moving average value
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < window:
            # If not enough data, use simple average
            return clean_series.mean() if len(clean_series) > 0 else 0.0
        
        # Calculate moving average of last 'window' periods
        return clean_series.tail(window).mean()
    
    def _calculate_exponential_smoothing(self, series: pd.Series, alpha: float = 0.3) -> float:
        """
        Calculate exponential smoothing forecast
        
        Args:
            series: Time series
            alpha: Smoothing factor (0-1)
            
        Returns:
            Forecasted value
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return 0.0
        
        if len(clean_series) == 1:
            return clean_series.iloc[0]
        
        # Initialize with first value
        forecast = clean_series.iloc[0]
        
        # Apply exponential smoothing
        for value in clean_series.iloc[1:]:
            forecast = alpha * value + (1 - alpha) * forecast
        
        return forecast
    
    def _calculate_arima_simple(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> float:
        """
        Calculate simple ARIMA forecast
        Note: This is a simplified version. For production, use statsmodels.
        
        Args:
            series: Time series
            order: ARIMA order (p, d, q)
            
        Returns:
            Forecasted value
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 3:
            # Not enough data for ARIMA, fall back to average
            return clean_series.mean() if len(clean_series) > 0 else 0.0
        
        # Simplified ARIMA: use weighted average with trend
        recent_values = clean_series.tail(6).values
        
        # Calculate simple trend
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
            base_value = recent_values[-1]
            return max(0, base_value + trend)
        
        return recent_values[-1]
    
    # ============================================
    # MÉTODOS DE EJECUCIÓN DE MODELOS
    # ============================================
    
    def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Moving Average model with growth factors
        
        Args:
            datos: Prepared data
            
        Returns:
            Model result
        """
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            cells_updated = 0
            processed_products = 0
            
            # Get column names
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
                return self._resultado_vacio()
            
            # Add forecast columns to result
            for forecast_date in datos['forecast_dates']:
                df_result[forecast_date] = np.nan
            
            # Process each product row
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                # Get product information
                customer = row.get(customer_col, '')
                product_model = row.get(model_col, '') if model_col else ''
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Process each forecast month
                for forecast_date in datos['forecast_dates']:
                    
                    # Get logic for this class and month
                    calc_base, p2p_model, launch_month = self._get_logic_for_product(
                        customer, product_class, forecast_date, datos
                    )
                    
                    # Handle "Não calcula" (No calculation)
                    if calc_base == self.CALC_BASE_NO_CALC:
                        df_result.loc[idx, forecast_date] = 0
                        cells_updated += 1
                        continue
                    
                    # Handle "Depende do mês de Lançamento" (Launch dependent)
                    if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
                        if pd.notna(launch_month):
                            launch_date = pd.to_datetime(launch_month)
                            if forecast_date < launch_date:
                                # Before launch: set to 0
                                df_result.loc[idx, forecast_date] = 0
                                cells_updated += 1
                                continue
                        # After launch: continue with P2P logic
                    
                    # Handle "DE PARA SEGUINTE" (P2P) or launch-dependent after launch
                    if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
                        # Determine which model to use for historical data
                        if pd.notna(p2p_model) and p2p_model != '':
                            # Use P2P model data
                            historical_series = self._get_p2p_series(
                                customer, p2p_model, datos
                            )
                        else:
                            # Use same product's historical data
                            historical_series = self._get_historical_series(idx, datos)
                        
                        # Calculate moving average
                        if len(historical_series.dropna()) > 0:
                            ma_value = self._calculate_moving_average(historical_series, window=3)
                            
                            # Apply growth factor from Relations
                            year = forecast_date.year
                            growth_factor = self._get_growth_factor(customer, year, datos)
                            
                            forecasted_value = ma_value * growth_factor
                            
                            df_result.loc[idx, forecast_date] = max(0, forecasted_value)
                            cells_updated += 1
                        else:
                            df_result.loc[idx, forecast_date] = 0
                            cells_updated += 1
                
                processed_products += 1
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'window': 3,
                    'type': 'simple',
                    'growth_factors_applied': True
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
        """
        Execute Exponential Smoothing model
        
        Args:
            datos: Prepared data
            
        Returns:
            Model result
        """
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            alpha = self.parametros.get('alpha', 0.3)
            cells_updated = 0
            processed_products = 0
            
            # Get column names
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns in Main sheet")
                return self._resultado_vacio()
            
            # Add forecast columns
            for forecast_date in datos['forecast_dates']:
                df_result[forecast_date] = np.nan
            
            # Process each product row
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_model = row.get(model_col, '') if model_col else ''
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Process each forecast month
                for forecast_date in datos['forecast_dates']:
                    
                    calc_base, p2p_model, launch_month = self._get_logic_for_product(
                        customer, product_class, forecast_date, datos
                    )
                    
                    if calc_base == self.CALC_BASE_NO_CALC:
                        df_result.loc[idx, forecast_date] = 0
                        cells_updated += 1
                        continue
                    
                    if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
                        if pd.notna(launch_month):
                            launch_date = pd.to_datetime(launch_month)
                            if forecast_date < launch_date:
                                df_result.loc[idx, forecast_date] = 0
                                cells_updated += 1
                                continue
                    
                    if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
                        if pd.notna(p2p_model) and p2p_model != '':
                            historical_series = self._get_p2p_series(customer, p2p_model, datos)
                        else:
                            historical_series = self._get_historical_series(idx, datos)
                        
                        if len(historical_series.dropna()) > 0:
                            es_value = self._calculate_exponential_smoothing(historical_series, alpha)
                            df_result.loc[idx, forecast_date] = max(0, es_value)
                            cells_updated += 1
                        else:
                            df_result.loc[idx, forecast_date] = 0
                            cells_updated += 1
                
                processed_products += 1
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'alpha': alpha,
                    'type': 'simple_exponential_smoothing'
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
        """
        Execute ARIMA model
        
        Args:
            datos: Prepared data
            
        Returns:
            Model result
        """
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            arima_params = self.parametros.get('arima_params', (1, 1, 1))
            cells_updated = 0
            processed_products = 0
            
            # Get column names
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns in Main sheet")
                return self._resultado_vacio()
            
            # Add forecast columns
            for forecast_date in datos['forecast_dates']:
                df_result[forecast_date] = np.nan
            
            # Process each product row
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_model = row.get(model_col, '') if model_col else ''
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Process each forecast month
                for forecast_date in datos['forecast_dates']:
                    
                    calc_base, p2p_model, launch_month = self._get_logic_for_product(
                        customer, product_class, forecast_date, datos
                    )
                    
                    if calc_base == self.CALC_BASE_NO_CALC:
                        df_result.loc[idx, forecast_date] = 0
                        cells_updated += 1
                        continue
                    
                    if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
                        if pd.notna(launch_month):
                            launch_date = pd.to_datetime(launch_month)
                            if forecast_date < launch_date:
                                df_result.loc[idx, forecast_date] = 0
                                cells_updated += 1
                                continue
                    
                    if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
                        if pd.notna(p2p_model) and p2p_model != '':
                            historical_series = self._get_p2p_series(customer, p2p_model, datos)
                        else:
                            historical_series = self._get_historical_series(idx, datos)
                        
                        if len(historical_series.dropna()) > 0:
                            arima_value = self._calculate_arima_simple(historical_series, arima_params)
                            df_result.loc[idx, forecast_date] = max(0, arima_value)
                            cells_updated += 1
                        else:
                            df_result.loc[idx, forecast_date] = 0
                            cells_updated += 1
                
                processed_products += 1
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'p': arima_params[0],
                    'd': arima_params[1],
                    'q': arima_params[2],
                    'type': 'simplified_arima'
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
    
    # ============================================
    # AUXILIARY METHODS
    # ============================================
    
    def _get_historical_series(self, row_idx: int, datos: Dict) -> pd.Series:
        """
        Get historical time series for a product row
        
        Args:
            row_idx: Row index
            datos: Prepared data
            
        Returns:
            Time series with numeric values
        """
        df_main = datos['df_main']
        row = df_main.iloc[row_idx]
        
        # Extract values from date columns
        series = row[datos['date_columns']]
        
        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce')
        
        return series
    
    def _get_p2p_series(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
        """
        Get historical series from P2P model
        
        Args:
            customer: Customer ID
            p2p_model: P2P model name
            datos: Prepared data
            
        Returns:
            Time series from P2P model
        """
        df_main = datos['df_main']
        col_names = datos['col_names']
        
        customer_col = col_names.get('main_customer')
        model_col = col_names.get('main_product_model')
        
        if not customer_col or not model_col:
            return pd.Series(dtype=float)
        
        # Find the P2P model row for this customer
        try:
            mask = (
                (df_main[customer_col] == customer) &
                (df_main[model_col] == p2p_model)
            )
            
            matching_rows = df_main[mask]
            
            if len(matching_rows) > 0:
                # Use first match
                row = matching_rows.iloc[0]
                series = row[datos['date_columns']]
                series = pd.to_numeric(series, errors='coerce')
                return series
        except Exception as e:
            pass
        
        # If not found, return empty series
        return pd.Series(dtype=float)
    
    def _resultado_vacio(self) -> Dict[str, Any]:
        """Return empty result structure in case of error"""
        return {
            'dataframe': pd.DataFrame(),
            'celulas_actualizadas': 0,
            'parametros': {},
            'metadata': {
                'error': True
            }
        }