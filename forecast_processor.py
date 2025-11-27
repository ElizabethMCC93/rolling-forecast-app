# """
# Forecasting processing engine - Optimized version with reference product and start date logic
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
#         self._reference_series_cache = {}
    
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
#             'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Launch_Month'],
#             'reference_model': ['Modelo de Referência', 'Reference Model', 'Referencia', 'Reference'],
#             'start_date': ['Data Início', 'Start Date', 'XF', 'Início']
#         }
        
#         # Map column indices for Main sheet (we know the structure)
#         # A=0 (ID Customer), B=1 (Product Model), G=6 (Reference Model), H=7 (Start Date)
#         main_cols = list(self.df_main.columns)
        
#         if len(main_cols) > 0:
#             self.col_names['main_customer'] = main_cols[0]  # Column A
#         if len(main_cols) > 1:
#             self.col_names['main_product_model'] = main_cols[1]  # Column B
#         if len(main_cols) > 5:
#             self.col_names['main_class'] = main_cols[5]  # Column F (assuming)
#         if len(main_cols) > 6:
#             self.col_names['main_reference_model'] = main_cols[6]  # Column G
#         if len(main_cols) > 7:
#             self.col_names['main_start_date'] = main_cols[7]  # Column H (XF)
        
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
#             st.write("**Detected mappings:**")
#             st.json(self.col_names)
    
#     def _parse_start_date(self, value) -> datetime:
#         """Parse start date from column H in format 01/mm/yyyy or other formats"""
#         if pd.isna(value):
#             return None
        
#         try:
#             if isinstance(value, (datetime, pd.Timestamp)):
#                 return pd.to_datetime(value)
            
#             if isinstance(value, str):
#                 value = value.strip()
#                 # Try dd/mm/yyyy format
#                 try:
#                     return pd.to_datetime(value, format='%d/%m/%Y')
#                 except:
#                     pass
#                 # Try other formats
#                 try:
#                     return pd.to_datetime(value, dayfirst=True)
#                 except:
#                     pass
            
#             return None
#         except:
#             return None
    
#     def _should_forecast_in_month(self, start_date: datetime, forecast_date: datetime) -> bool:
#         """Check if forecast should be applied in this month based on start date"""
#         if start_date is None:
#             return True
        
#         # Normalize both dates to first day of month
#         start_normalized = datetime(start_date.year, start_date.month, 1)
#         forecast_normalized = datetime(forecast_date.year, forecast_date.month, 1)
        
#         return forecast_normalized >= start_normalized
    
#     def _get_reference_product_series(self, customer: str, reference_model: str, datos: Dict) -> pd.Series:
#         """
#         Get historical series for reference product (same customer, model from column G)
#         """
#         cache_key = (customer, reference_model)
        
#         if cache_key in self._reference_series_cache:
#             return self._reference_series_cache[cache_key]
        
#         df_main = datos['df_main']
#         col_names = datos['col_names']
        
#         customer_col = col_names.get('main_customer')
#         model_col = col_names.get('main_product_model')
        
#         if not customer_col or not model_col:
#             return pd.Series(dtype=float)
        
#         try:
#             # Search for product with same customer and model = reference_model
#             mask = (
#                 (df_main[customer_col].astype(str).str.strip() == str(customer).strip()) &
#                 (df_main[model_col].astype(str).str.strip() == str(reference_model).strip())
#             )
            
#             matching_rows = df_main[mask]
            
#             if len(matching_rows) > 0:
#                 row = matching_rows.iloc[0]
#                 series = pd.Series(dtype=float)
                
#                 for date_col in datos['date_columns']:
#                     val = row.get(date_col, np.nan)
#                     val = pd.to_numeric(val, errors='coerce')
#                     series[date_col] = val
                
#                 self._reference_series_cache[cache_key] = series
#                 return series
#         except Exception:
#             pass
        
#         return pd.Series(dtype=float)
    
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
#             with st.spinner("📈 Executing Moving Average..."):
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
#         # Cache 4: Reference product series - built on demand
    
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
    
#     def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Seasonal Moving Average with reference product and start date logic"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
#             reference_col = col_names.get('main_reference_model')
#             start_date_col = col_names.get('main_start_date')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # OPTIMIZATION: Extract all historical series at once
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
#                 product_model = row.get(model_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Get start date from column H (if exists)
#                 start_date = None
#                 if start_date_col:
#                     start_date_value = row.get(start_date_col)
#                     start_date = self._parse_start_date(start_date_value)
                
#                 # Check if there's a reference product in column G
#                 reference_model = None
#                 if reference_col:
#                     reference_value = row.get(reference_col)
#                     if pd.notna(reference_value):
#                         reference_str = str(reference_value).strip()
#                         product_model_str = str(product_model).strip()
                        
#                         # Only use reference if it's different from current model and not empty/dash
#                         if reference_str != '' and reference_str != '-' and reference_str != product_model_str:
#                             reference_model = reference_str
                
#                 # Get historical series (own or reference)
#                 if reference_model:
#                     # Use reference product series
#                     historical_series = self._get_reference_product_series(customer, reference_model, datos)
#                     if len(historical_series) == 0 or historical_series.isna().all():
#                         # Fallback to own series if reference not found
#                         historical_series = historical_data[idx]
#                 else:
#                     # Use own series
#                     historical_series = historical_data[idx]
                
#                 # Process each forecast month
#                 for forecast_date in datos['forecast_dates']:
                    
#                     # Check if we should forecast in this month based on start date
#                     if not self._should_forecast_in_month(start_date, forecast_date):
#                         # Leave as NaN (skip this month)
#                         continue
                    
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
                        
#                         # Get historical series (from P2P or already determined series)
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
#                     'reference_product_logic': True,
#                     'start_date_restriction': True,
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








# """
# Forecasting processing engine - Optimized version with reference product and start date logic
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
#         self._reference_series_cache = {}
    
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
#             'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Launch_Month'],
#             'reference_model': ['Modelo de Referência', 'Reference Model', 'Referencia', 'Reference'],
#             'start_date': ['Data Início', 'Start Date', 'XF', 'Início']
#         }
        
#         # Map column indices for Main sheet (we know the structure)
#         # A=0 (ID Customer), B=1 (Product Model), F=5 (Class), G=6 (Reference Model), H=7 (Start Date)
#         main_cols = list(self.df_main.columns)
        
#         if len(main_cols) > 0:
#             self.col_names['main_customer'] = main_cols[0]  # Column A
#         if len(main_cols) > 1:
#             self.col_names['main_product_model'] = main_cols[1]  # Column B
#         if len(main_cols) > 5:
#             self.col_names['main_class'] = main_cols[5]  # Column F
#         if len(main_cols) > 6:
#             self.col_names['main_reference_model'] = main_cols[6]  # Column G
#         if len(main_cols) > 7:
#             self.col_names['main_start_date'] = main_cols[7]  # Column H (XF)
        
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
#             st.write("**Detected mappings:**")
#             st.json(self.col_names)
    
#     def _parse_start_date(self, value) -> datetime:
#         """Parse start date from column H in format 01/mm/yyyy or other formats"""
#         if pd.isna(value):
#             return None
        
#         try:
#             if isinstance(value, (datetime, pd.Timestamp)):
#                 return pd.to_datetime(value)
            
#             if isinstance(value, str):
#                 value = value.strip()
#                 # Try dd/mm/yyyy format
#                 try:
#                     return pd.to_datetime(value, format='%d/%m/%Y')
#                 except:
#                     pass
#                 # Try other formats
#                 try:
#                     return pd.to_datetime(value, dayfirst=True)
#                 except:
#                     pass
            
#             return None
#         except:
#             return None
    
#     def _should_forecast_in_month(self, start_date: datetime, forecast_date: datetime) -> bool:
#         """Check if forecast should be applied in this month based on start date"""
#         if start_date is None:
#             return True
        
#         # Normalize both dates to first day of month
#         start_normalized = datetime(start_date.year, start_date.month, 1)
#         forecast_normalized = datetime(forecast_date.year, forecast_date.month, 1)
        
#         return forecast_normalized >= start_normalized
    
#     def _get_reference_product_series(self, customer: str, reference_model: str, datos: Dict) -> pd.Series:
#         """
#         Get historical series for reference product (same ID Customer, model from column G)
#         IMPORTANT: Does NOT filter by classification - gets data even if reference is "out of line"
#         """
#         cache_key = (customer, reference_model)
        
#         if cache_key in self._reference_series_cache:
#             return self._reference_series_cache[cache_key]
        
#         df_main = datos['df_main']
#         col_names = datos['col_names']
        
#         customer_col = col_names.get('main_customer')
#         model_col = col_names.get('main_product_model')
        
#         if not customer_col or not model_col:
#             return pd.Series(dtype=float)
        
#         try:
#             # Search for product with:
#             # - Same ID Customer (column A)
#             # - Model (column B) = reference_model (value from column G)
#             # NO classification filter - we want data even if it's "out of line"
#             mask = (
#                 (df_main[customer_col].astype(str).str.strip() == str(customer).strip()) &
#                 (df_main[model_col].astype(str).str.strip() == str(reference_model).strip())
#             )
            
#             matching_rows = df_main[mask]
            
#             if len(matching_rows) > 0:
#                 row = matching_rows.iloc[0]
#                 series = pd.Series(dtype=float)
                
#                 for date_col in datos['date_columns']:
#                     val = row.get(date_col, np.nan)
#                     val = pd.to_numeric(val, errors='coerce')
#                     series[date_col] = val
                
#                 self._reference_series_cache[cache_key] = series
#                 return series
#             else:
#                 # Only print if reference model is not empty/dash (actual reference that wasn't found)
#                 if reference_model and reference_model != '-' and reference_model.strip() != '':
#                     print(f"[INFO] Reference product not found: Customer='{customer}', Reference Model='{reference_model}'")
#         except Exception as e:
#             print(f"[ERROR] Exception in _get_reference_product_series: {e}")
        
#         return pd.Series(dtype=float)
    
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
#             with st.spinner("📈 Executing Moving Average..."):
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
#         # Cache 4: Reference product series - built on demand
    
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
    
#     def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Seasonal Moving Average with reference product and start date logic"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
#             reference_col = col_names.get('main_reference_model')
#             start_date_col = col_names.get('main_start_date')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # OPTIMIZATION: Extract all historical series at once
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
#                 product_model = row.get(model_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # CRITICAL: Check classification BEFORE any processing
#                 # Get logic for first forecast month to check if it's "Não calcula"
#                 calc_base_check, _, _ = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # If classification is "Não calcula", skip this product entirely (leave all months as NaN)
#                 if calc_base_check == self.CALC_BASE_NO_CALC:
#                     skipped_no_calc += 1
#                     continue
                
#                 # Get start date from column H (if exists)
#                 start_date = None
#                 if start_date_col:
#                     start_date_value = row.get(start_date_col)
#                     start_date = self._parse_start_date(start_date_value)
                
#                 # Check if there's a reference product in column G
#                 reference_model = None
#                 if reference_col:
#                     reference_value = row.get(reference_col)
#                     if pd.notna(reference_value):
#                         reference_str = str(reference_value).strip()
#                         product_model_str = str(product_model).strip()
                        
#                         # Only use reference if it's different from current model and not empty/dash
#                         if reference_str != '' and reference_str != '-' and reference_str != product_model_str:
#                             reference_model = reference_str
                
#                 # Get historical series (own or reference)
#                 if reference_model:
#                     # Use reference product series (even if reference is "out of line")
#                     historical_series = self._get_reference_product_series(customer, reference_model, datos)
#                     if len(historical_series) == 0 or historical_series.isna().all():
#                         # Fallback to own series if reference not found
#                         historical_series = historical_data[idx]
#                 else:
#                     # Use own series
#                     historical_series = historical_data[idx]
                
#                 # Process each forecast month
#                 for forecast_date in datos['forecast_dates']:
                    
#                     # Check if we should forecast in this month based on start date
#                     if not self._should_forecast_in_month(start_date, forecast_date):
#                         # Leave as NaN (skip this month)
#                         continue
                    
#                     # Get logic from cache
#                     calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                         customer, product_class, forecast_date
#                     )
                    
#                     # Double-check "Não calcula" (shouldn't reach here, but safety check)
#                     if calc_base == self.CALC_BASE_NO_CALC:
#                         continue
                    
#                     # Handle launch dependent
#                     if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 # Leave as NaN before launch date
#                                 continue
                    
#                     # Handle P2P
#                     if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         # Get historical series (from P2P or already determined series)
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
#                                 if pd.notna(simple_avg) and simple_avg > 0:
#                                     year = forecast_date.year
#                                     growth_factor = self._get_growth_factor_cached(customer, year)
#                                     forecast_arrays[forecast_date][idx] = max(0, simple_avg * growth_factor)
#                                     cells_updated += 1
                
#                 processed_products += 1
                
#                 # Update progress every 100 products
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             # Clear progress bar
#             progress_bar.empty()
            
#             # Add forecast columns to result
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'seasonal_moving_average',
#                     'window': 'same_month_last_year + 2_months',
#                     'growth_factors_applied': True,
#                     'reference_product_logic': True,
#                     'start_date_restriction': True,
#                     'skipped_no_calc': skipped_no_calc,
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
    
#     def _ejecutar_suavizacao(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Exponential Smoothing with classification check"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
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
                
#                 # CRITICAL: Check classification first
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Skip if "Não calcula"
#                 if calc_base == self.CALC_BASE_NO_CALC:
#                     skipped_no_calc += 1
#                     continue
                
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
                    
#                     except Exception:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                
#                 # Update progress
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
#                     'seasonal_periods': 12,
#                     'skipped_no_calc': skipped_no_calc,
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
    
#     def _ejecutar_arima(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute SARIMA with classification check"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             arima_params = self.parametros.get('arima_params', (1, 1, 1))
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
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
                
#                 # CRITICAL: Check classification first
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Skip if "Não calcula"
#                 if calc_base == self.CALC_BASE_NO_CALC:
#                     skipped_no_calc += 1
#                     continue
                
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
                    
#                     except Exception:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                
#                 # Update progress
#                 if processed_products % 50 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
#                     'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
#                     'skipped_no_calc': skipped_no_calc,
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
#         except Exception:
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








# """
# Forecasting processing engine - Fixed version
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
#         self._reference_series_cache = {}
    
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
#             'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Launch_Month'],
#             'reference_model': ['Modelo de Referência', 'Reference Model', 'Referencia', 'Reference'],
#             'start_date': ['Data Início', 'Start Date', 'XF', 'Início']
#         }
        
#         # Map column indices for Main sheet (we know the structure)
#         # A=0 (ID Customer), B=1 (Product Model), F=5 (Class), G=6 (Reference Model), H=7 (Start Date)
#         main_cols = list(self.df_main.columns)
        
#         if len(main_cols) > 0:
#             self.col_names['main_customer'] = main_cols[0]  # Column A
#         if len(main_cols) > 1:
#             self.col_names['main_product_model'] = main_cols[1]  # Column B
#         if len(main_cols) > 5:
#             self.col_names['main_class'] = main_cols[5]  # Column F
#         if len(main_cols) > 6:
#             self.col_names['main_reference_model'] = main_cols[6]  # Column G
#         if len(main_cols) > 7:
#             self.col_names['main_start_date'] = main_cols[7]  # Column H (XF)
        
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
#             st.write("**Detected mappings:**")
#             st.json(self.col_names)
    
#     def _parse_start_date(self, value) -> datetime:
#         """Parse start date from column H in format 01/mm/yyyy or other formats"""
#         if pd.isna(value):
#             return None
        
#         try:
#             if isinstance(value, (datetime, pd.Timestamp)):
#                 return pd.to_datetime(value)
            
#             if isinstance(value, str):
#                 value = value.strip()
#                 # Try dd/mm/yyyy format
#                 try:
#                     return pd.to_datetime(value, format='%d/%m/%Y')
#                 except:
#                     pass
#                 # Try other formats
#                 try:
#                     return pd.to_datetime(value, dayfirst=True)
#                 except:
#                     pass
            
#             return None
#         except:
#             return None
    
#     def _should_forecast_in_month(self, start_date: datetime, forecast_date: datetime) -> bool:
#         """Check if forecast should be applied in this month based on start date"""
#         if start_date is None:
#             return True
        
#         # Normalize both dates to first day of month
#         start_normalized = datetime(start_date.year, start_date.month, 1)
#         forecast_normalized = datetime(forecast_date.year, forecast_date.month, 1)
        
#         return forecast_normalized >= start_normalized
    
#     def _has_no_calc_in_any_month(self, customer: str, product_class: str, forecast_dates: List[datetime]) -> bool:
#         """
#         CRITICAL: Check if product has "Não calcula" classification in ANY forecast month.
#         If yes, skip product entirely across ALL models.
#         """
#         for forecast_date in forecast_dates:
#             calc_base, _, _ = self._get_logic_for_product_cached(customer, product_class, forecast_date)
#             if calc_base == self.CALC_BASE_NO_CALC:
#                 return True
#         return False
    
#     def _get_reference_product_series(self, customer: str, reference_model: str, datos: Dict) -> pd.Series:
#         """
#         Get historical series for reference product (same ID Customer, model from column G)
#         IMPORTANT: Does NOT filter by classification - gets data even if reference is "out of line"
#         """
#         cache_key = (customer, reference_model)
        
#         if cache_key in self._reference_series_cache:
#             return self._reference_series_cache[cache_key]
        
#         df_main = datos['df_main']
#         col_names = datos['col_names']
        
#         customer_col = col_names.get('main_customer')
#         model_col = col_names.get('main_product_model')
        
#         if not customer_col or not model_col:
#             return pd.Series(dtype=float)
        
#         try:
#             # Search for product with:
#             # - Same ID Customer (column A)
#             # - Model (column B) = reference_model (value from column G)
#             # NO classification filter - we want data even if it's "out of line"
#             mask = (
#                 (df_main[customer_col].astype(str).str.strip() == str(customer).strip()) &
#                 (df_main[model_col].astype(str).str.strip() == str(reference_model).strip())
#             )
            
#             matching_rows = df_main[mask]
            
#             if len(matching_rows) > 0:
#                 row = matching_rows.iloc[0]
#                 series = pd.Series(dtype=float)
                
#                 for date_col in datos['date_columns']:
#                     val = row.get(date_col, np.nan)
#                     val = pd.to_numeric(val, errors='coerce')
#                     series[date_col] = val
                
#                 self._reference_series_cache[cache_key] = series
#                 return series
#         except Exception:
#             pass
        
#         return pd.Series(dtype=float)
    
#     def _determine_historical_series(self, idx: int, row: pd.Series, datos: Dict) -> pd.Series:
#         """
#         Determine which historical series to use based on column G (reference model).
#         Priority:
#         1. If column G != column B and valid: use reference product series
#         2. Otherwise: use own product series
#         """
#         col_names = datos['col_names']
#         customer_col = col_names.get('main_customer')
#         model_col = col_names.get('main_product_model')
#         reference_col = col_names.get('main_reference_model')
        
#         customer = row.get(customer_col, '')
#         product_model = row.get(model_col, '')
        
#         # Check if there's a reference product in column G
#         reference_model = None
#         if reference_col:
#             reference_value = row.get(reference_col)
#             if pd.notna(reference_value):
#                 reference_str = str(reference_value).strip()
#                 product_model_str = str(product_model).strip()
                
#                 # Only use reference if it's different from current model and not empty/dash
#                 if reference_str != '' and reference_str != '-' and reference_str != product_model_str:
#                     reference_model = reference_str
        
#         # Get historical series (reference or own)
#         if reference_model:
#             # Use reference product series (even if reference is "out of line")
#             historical_series = self._get_reference_product_series(customer, reference_model, datos)
#             if len(historical_series) == 0 or historical_series.isna().all():
#                 # Fallback to own series if reference not found
#                 historical_series = self._get_historical_series_indexed(idx, datos)
#         else:
#             # Use own series
#             historical_series = self._get_historical_series_indexed(idx, datos)
        
#         return historical_series
    
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
#             with st.spinner("📈 Executing Moving Average..."):
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
    
#     def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Seasonal Moving Average with reference product and start date logic"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
#             start_date_col = col_names.get('main_start_date')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns with NaN
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
                
#                 # CRITICAL: Check if product has "Não calcula" in ANY month - skip entirely if true
#                 if self._has_no_calc_in_any_month(customer, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     # Leave ALL months as NaN - do not fill with 0
#                     continue
                
#                 # Get start date from column H (if exists)
#                 start_date = None
#                 if start_date_col:
#                     start_date_value = row.get(start_date_col)
#                     start_date = self._parse_start_date(start_date_value)
                
#                 # Determine historical series (handles reference product logic from column G)
#                 historical_series = self._determine_historical_series(idx, row, datos)
                
#                 # Process each forecast month
#                 for forecast_date in datos['forecast_dates']:
                    
#                     # Check if we should forecast in this month based on start date
#                     if not self._should_forecast_in_month(start_date, forecast_date):
#                         # Leave as NaN (skip this month) - do not fill with 0
#                         continue
                    
#                     # Get logic from cache
#                     calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                         customer, product_class, forecast_date
#                     )
                    
#                     # Handle launch dependent
#                     if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 # Leave as NaN before launch date - do not fill with 0
#                                 continue
                    
#                     # Handle P2P
#                     if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         # Get historical series (from P2P or already determined series)
#                         if pd.notna(p2p_model) and p2p_model != '':
#                             series_to_use = self._get_p2p_series_cached(customer, p2p_model, datos)
#                         else:
#                             series_to_use = historical_series
                        
#                         if len(series_to_use) > 0 and not series_to_use.isna().all():
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
#                                 if pd.notna(simple_avg) and simple_avg > 0:
#                                     year = forecast_date.year
#                                     growth_factor = self._get_growth_factor_cached(customer, year)
#                                     forecast_arrays[forecast_date][idx] = max(0, simple_avg * growth_factor)
#                                     cells_updated += 1
                
#                 processed_products += 1
                
#                 # Update progress every 100 products
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             # Clear progress bar
#             progress_bar.empty()
            
#             # Add forecast columns to result
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if skipped_no_calc > 0:
#                 st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'seasonal_moving_average',
#                     'window': 'same_month_last_year + 2_months',
#                     'growth_factors_applied': True,
#                     'reference_product_logic': True,
#                     'start_date_restriction': True,
#                     'skipped_no_calc': skipped_no_calc,
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
    
#     def _ejecutar_suavizacao(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Exponential Smoothing with classification check and reference product logic"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns with NaN
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
                
#                 # CRITICAL: Check if product has "Não calcula" in ANY month - skip entirely if true
#                 if self._has_no_calc_in_any_month(customer, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     # Leave ALL months as NaN - do not fill with 0
#                     continue
                
#                 # Get logic (use first forecast month)
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Determine historical series (handles reference product logic from column G)
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     # P2P has priority
#                     historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     # Use reference product logic (column G) or own series
#                     historical_series = self._determine_historical_series(idx, row, datos)
                
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
#                             if pd.notna(last_value) and last_value > 0:
#                                 for forecast_date in datos['forecast_dates']:
#                                     forecast_arrays[forecast_date][idx] = max(0, last_value)
#                                     cells_updated += 1
#                                 processed_products += 1
                    
#                     except Exception:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                
#                 # Update progress
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if skipped_no_calc > 0:
#                 st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
#                     'seasonal_periods': 12,
#                     'reference_product_logic': True,
#                     'skipped_no_calc': skipped_no_calc,
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
    
#     def _ejecutar_arima(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute SARIMA with classification check and reference product logic"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             arima_params = self.parametros.get('arima_params', (1, 1, 1))
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate with NaN
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
                
#                 # CRITICAL: Check if product has "Não calcula" in ANY month - skip entirely if true
#                 if self._has_no_calc_in_any_month(customer, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     # Leave ALL months as NaN - do not fill with 0
#                     continue
                
#                 # Get logic
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Determine historical series (handles reference product logic from column G)
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     # P2P has priority
#                     historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     # Use reference product logic (column G) or own series
#                     historical_series = self._determine_historical_series(idx, row, datos)
                
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
                    
#                     except Exception:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                
#                 # Update progress
#                 if processed_products % 50 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if skipped_no_calc > 0:
#                 st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
#                     'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
#                     'reference_product_logic': True,
#                     'skipped_no_calc': skipped_no_calc,
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
#         except Exception:
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











# """
# Forecasting processing engine - DEBUG to TERMINAL only
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
    
#     # DEBUG: Target record for detailed logging (TERMINAL ONLY)
#     DEBUG_TARGET_MODEL = "4145766"
#     DEBUG_TARGET_CUSTOMER = "ALPASHOP"
    
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
#         self._reference_series_cache = {}
        
#         # DEBUG: Log initialization to TERMINAL
#         print(f"\n{'='*80}")
#         print(f"🔍 DEBUG INIT")
#         print(f"  Forecast Start Date: {self.forecast_start_date.strftime('%Y-%m-%d')}")
#         print(f"  Forecast Months: {self.forecast_months}")
#         print(f"  Date Columns Range: {self.date_columns[0].strftime('%Y-%m')} to {self.date_columns[-1].strftime('%Y-%m')}")
#         print(f"{'='*80}\n")
    
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
#             'forecast_start_month': ['Mês Início Previsão', 'Forecast Start Month', 'Mes Inicio', 'Month Start'],
#             'p2p': ['P2P', 'p2p'],
#             'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Launch_Month'],
#             'reference_model': ['Modelo de Referência', 'Reference Model', 'Referencia', 'Reference'],
#             'start_date': ['Data Início', 'Start Date', 'XF', 'Início']
#         }
        
#         # Map column indices for Main sheet (we know the structure)
#         # A=0 (ID Customer), B=1 (Product Model), F=5 (Class), G=6 (Reference Model), H=7 (Start Date)
#         main_cols = list(self.df_main.columns)
        
#         if len(main_cols) > 0:
#             self.col_names['main_customer'] = main_cols[0]  # Column A
#         if len(main_cols) > 1:
#             self.col_names['main_product_model'] = main_cols[1]  # Column B
#         if len(main_cols) > 5:
#             self.col_names['main_class'] = main_cols[5]  # Column F
#         if len(main_cols) > 6:
#             self.col_names['main_reference_model'] = main_cols[6]  # Column G
#         if len(main_cols) > 7:
#             self.col_names['main_start_date'] = main_cols[7]  # Column H (XF)
        
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
    
#     def _parse_start_date(self, value) -> datetime:
#         """Parse start date from column H in format 01/mm/yyyy or other formats"""
#         if pd.isna(value):
#             return None
        
#         try:
#             if isinstance(value, (datetime, pd.Timestamp)):
#                 return pd.to_datetime(value)
            
#             if isinstance(value, str):
#                 value = value.strip()
#                 # Try dd/mm/yyyy format
#                 try:
#                     return pd.to_datetime(value, format='%d/%m/%Y')
#                 except:
#                     pass
#                 # Try other formats
#                 try:
#                     return pd.to_datetime(value, dayfirst=True)
#                 except:
#                     pass
            
#             return None
#         except:
#             return None
    
#     def _should_forecast_in_month(self, start_date: datetime, forecast_date: datetime) -> bool:
#         """Check if forecast should be applied in this month based on start date"""
#         if start_date is None:
#             return True
        
#         # Normalize both dates to first day of month
#         start_normalized = datetime(start_date.year, start_date.month, 1)
#         forecast_normalized = datetime(forecast_date.year, forecast_date.month, 1)
        
#         return forecast_normalized >= start_normalized
    
#     def _has_no_calc_in_any_month(self, customer: str, product_model: str, product_class: str, forecast_dates: List[datetime]) -> bool:
#         """
#         CRITICAL: Check if product has "Não calcula" classification in ANY forecast month.
#         If yes, skip product entirely across ALL models.
        
#         DEBUG: Logs details for target model to TERMINAL
#         """
#         is_debug_target = (str(product_model).strip() == self.DEBUG_TARGET_MODEL and 
#                           str(customer).strip() == self.DEBUG_TARGET_CUSTOMER)
        
#         if is_debug_target:
#             print(f"\n{'='*80}")
#             print(f"🔍 DEBUG _has_no_calc_in_any_month")
#             print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
        
#         for forecast_date in forecast_dates:
#             calc_base, _, _ = self._get_logic_for_product_cached(customer, product_class, forecast_date)
            
#             if is_debug_target:
#                 print(f"  {forecast_date.strftime('%Y-%m')}: {calc_base}")
            
#             if calc_base == self.CALC_BASE_NO_CALC:
#                 if is_debug_target:
#                     print(f"  ❌ FOUND 'Não calcula' - SKIPPING")
#                     print(f"{'='*80}\n")
#                 return True
        
#         if is_debug_target:
#             print(f"  ✅ No 'Não calcula' - PROCESSING")
#             print(f"{'='*80}\n")
        
#         return False
    
#     def _get_reference_product_series(self, customer: str, reference_model: str, datos: Dict) -> pd.Series:
#         """
#         Get historical series for reference product (same ID Customer, model from column G)
#         IMPORTANT: Does NOT filter by classification - gets data even if reference is "out of line"
#         """
#         cache_key = (customer, reference_model)
        
#         if cache_key in self._reference_series_cache:
#             return self._reference_series_cache[cache_key]
        
#         df_main = datos['df_main']
#         col_names = datos['col_names']
        
#         customer_col = col_names.get('main_customer')
#         model_col = col_names.get('main_product_model')
        
#         if not customer_col or not model_col:
#             return pd.Series(dtype=float)
        
#         try:
#             mask = (
#                 (df_main[customer_col].astype(str).str.strip() == str(customer).strip()) &
#                 (df_main[model_col].astype(str).str.strip() == str(reference_model).strip())
#             )
            
#             matching_rows = df_main[mask]
            
#             if len(matching_rows) > 0:
#                 row = matching_rows.iloc[0]
#                 series = pd.Series(dtype=float)
                
#                 for date_col in datos['date_columns']:
#                     val = row.get(date_col, np.nan)
#                     val = pd.to_numeric(val, errors='coerce')
#                     series[date_col] = val
                
#                 self._reference_series_cache[cache_key] = series
#                 return series
#         except Exception:
#             pass
        
#         return pd.Series(dtype=float)
    
#     def _determine_historical_series(self, idx: int, row: pd.Series, datos: Dict) -> pd.Series:
#         """
#         Determine which historical series to use based on column G (reference model).
#         Priority:
#         1. If column G != column B and valid: use reference product series
#         2. Otherwise: use own product series
        
#         DEBUG: Logs details for target model to TERMINAL
#         """
#         col_names = datos['col_names']
#         customer_col = col_names.get('main_customer')
#         model_col = col_names.get('main_product_model')
#         reference_col = col_names.get('main_reference_model')
        
#         customer = row.get(customer_col, '')
#         product_model = row.get(model_col, '')
        
#         is_debug_target = (str(product_model).strip() == self.DEBUG_TARGET_MODEL and 
#                           str(customer).strip() == self.DEBUG_TARGET_CUSTOMER)
        
#         if is_debug_target:
#             print(f"\n{'='*80}")
#             print(f"🔍 DEBUG _determine_historical_series")
#             print(f"  Model: {product_model}")
        
#         # Check if there's a reference product in column G
#         reference_model = None
#         if reference_col:
#             reference_value = row.get(reference_col)
#             if pd.notna(reference_value):
#                 reference_str = str(reference_value).strip()
#                 product_model_str = str(product_model).strip()
                
#                 if is_debug_target:
#                     print(f"  Reference (Col G): '{reference_str}'")
#                     print(f"  Current (Col B): '{product_model_str}'")
                
#                 # Only use reference if it's different from current model and not empty/dash
#                 if reference_str != '' and reference_str != '-' and reference_str != product_model_str:
#                     reference_model = reference_str
#                     if is_debug_target:
#                         print(f"  ✅ USING REFERENCE: {reference_model}")
        
#         # Get historical series (reference or own)
#         if reference_model:
#             historical_series = self._get_reference_product_series(customer, reference_model, datos)
#             if len(historical_series) == 0 or historical_series.isna().all():
#                 if is_debug_target:
#                     print(f"  ⚠️ Reference not found, fallback to OWN")
#                 historical_series = self._get_historical_series_indexed(idx, datos)
#         else:
#             if is_debug_target:
#                 print(f"  ✅ USING OWN series")
#             historical_series = self._get_historical_series_indexed(idx, datos)
        
#         if is_debug_target:
#             non_null = historical_series.dropna()
#             print(f"  Series: {len(non_null)} values")
#             if len(non_null) > 0:
#                 print(f"  Range: {non_null.index.min().strftime('%Y-%m')} to {non_null.index.max().strftime('%Y-%m')}")
#                 print(f"  Sample: {dict(list(non_null.head(3).items()))}")
#             print(f"{'='*80}\n")
        
#         return historical_series
    
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
#             with st.spinner("📈 Executing Moving Average..."):
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
#         """
#         Pre-build caches for faster execution - CORRECTED to use forecast_start_month
        
#         CRITICAL FIX: Cache key now uses (forecast_start_month, class, month) instead of just (class, month)
        
#         DEBUG to TERMINAL only
#         """
        
#         df_logics = datos['df_logics']
#         df_relations = datos['df_relations']
#         col_names = datos['col_names']
        
#         print(f"\n{'='*80}")
#         print(f"🔍 DEBUG _build_caches")
        
#         # Cache 1: Logic lookup (forecast_start_month + class + month -> logic)
#         forecast_start_col = col_names.get('logics_forecast_start_month')
#         class_col = col_names.get('logics_class')
#         month_col = col_names.get('logics_month')
#         calc_base_col = col_names.get('logics_calc_base')
#         p2p_col = col_names.get('logics_p2p')
#         launch_col = col_names.get('logics_launch_month')
        
#         print(f"  LogicsxMonth Columns:")
#         print(f"    Forecast Start: {forecast_start_col}")
#         print(f"    Class: {class_col}")
#         print(f"    Month: {month_col}")
#         print(f"    Calc Base: {calc_base_col}")
        
#         # CRITICAL: If forecast_start_col is missing, try to detect it from column B
#         if not forecast_start_col and len(df_logics.columns) > 1:
#             forecast_start_col = df_logics.columns[1]  # Assume column B
#             col_names['logics_forecast_start_month'] = forecast_start_col
#             print(f"    ⚠️ Auto-detected Forecast Start as: {forecast_start_col}")
        
#         if class_col and month_col and class_col in df_logics.columns and month_col in df_logics.columns:
            
#             total_rows = 0
#             matching_forecast_start = 0
            
#             for idx, row in df_logics.iterrows():
#                 total_rows += 1
                
#                 product_class = row.get(class_col)
#                 month_date = pd.to_datetime(row.get(month_col)).replace(day=1)
                
#                 # Get forecast start month (column B in LogicsxMonth)
#                 forecast_start_month = None
#                 if forecast_start_col:
#                     forecast_start_value = row.get(forecast_start_col)
#                     if pd.notna(forecast_start_value):
#                         forecast_start_month = pd.to_datetime(forecast_start_value).replace(day=1)
                
#                 # CORRECTED: Cache key now includes forecast_start_month
#                 if forecast_start_month:
#                     cache_key = (forecast_start_month, product_class, month_date)
                    
#                     # Count how many match our forecast start date
#                     if forecast_start_month == self.forecast_start_date:
#                         matching_forecast_start += 1
#                 else:
#                     # Fallback to old structure if forecast_start_month is missing
#                     cache_key = (product_class, month_date)
                
#                 calc_base_value = row.get(calc_base_col, self.CALC_BASE_P2P) if calc_base_col else self.CALC_BASE_P2P
#                 p2p_value = row.get(p2p_col, None) if p2p_col else None
#                 launch_value = row.get(launch_col, None) if launch_col else None
                
#                 self._logic_cache[cache_key] = (calc_base_value, p2p_value, launch_value)
            
#             print(f"  ✅ Built logic cache:")
#             print(f"    Total entries: {len(self._logic_cache)}")
#             print(f"    Total rows processed: {total_rows}")
#             print(f"    Matching forecast start ({self.forecast_start_date.strftime('%Y-%m')}): {matching_forecast_start}")
#         else:
#             print(f"  ❌ Could not build logic cache - missing columns")
        
#         # Cache 2: Growth factors (customer + year -> factor)
#         customer_col = col_names.get('relations_customer')
        
#         if customer_col and customer_col in df_relations.columns:
#             for idx, row in df_relations.iterrows():
#                 customer = row.get(customer_col)
                
#                 # Get all year columns
#                 for col in df_relations.columns:
#                     is_year_column = False
                    
#                     if isinstance(col, str) and col.isdigit():
#                         is_year_column = True
#                         year = int(col)
#                     elif isinstance(col, int) and 2000 <= col <= 2100:
#                         is_year_column = True
#                         year = col
                    
#                     if is_year_column:
#                         factor = row.get(col)
                        
#                         if pd.notna(factor) and isinstance(factor, (int, float)):
#                             cache_key = (customer, year)
#                             self._growth_factor_cache[cache_key] = float(factor)
            
#             print(f"  ✅ Built growth factor cache: {len(self._growth_factor_cache)} entries")
        
#         print(f"{'='*80}\n")
    
#     def _get_logic_for_product_cached(self, customer: str, product_class: str, 
#                                      month_date: datetime) -> Tuple[str, str, Any]:
#         """
#         Get logic from cache - CORRECTED to use forecast_start_month
        
#         CRITICAL FIX: Now tries both new structure (forecast_start, class, month) 
#         and old structure (class, month) for backward compatibility
#         """
        
#         month_start = pd.to_datetime(month_date).replace(day=1)
        
#         # Try new structure first: (forecast_start_date, class, month)
#         cache_key_new = (self.forecast_start_date, product_class, month_start)
        
#         if cache_key_new in self._logic_cache:
#             return self._logic_cache[cache_key_new]
        
#         # Fallback to old structure: (class, month)
#         cache_key_old = (product_class, month_start)
        
#         if cache_key_old in self._logic_cache:
#             return self._logic_cache[cache_key_old]
        
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
    
#     def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute Seasonal Moving Average with reference product and start date logic
        
#         DEBUG: Detailed logging for target model 4145766 to TERMINAL
#         """
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
#             start_date_col = col_names.get('main_start_date')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Progress bar
#             progress_bar = st.progress(0)
#             total_products = len(df_main)
            
#             # Process each product row
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get(customer_col, '')
#                 product_model = row.get(model_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # DEBUG: Check if this is our target record
#                 is_debug_target = (str(product_model).strip() == self.DEBUG_TARGET_MODEL and 
#                                   str(customer).strip() == self.DEBUG_TARGET_CUSTOMER)
                
#                 if is_debug_target:
#                     print(f"\n{'='*80}")
#                     print(f"🎯 TARGET PRODUCT PROCESSING")
#                     print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
#                     print(f"  Row Index: {idx}")
#                     print(f"{'='*80}\n")
                
#                 # CRITICAL: Check if product has "Não calcula" in ANY month - skip entirely if true
#                 if self._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     continue
                
#                 # Get start date from column H (if exists)
#                 start_date = None
#                 if start_date_col:
#                     start_date_value = row.get(start_date_col)
#                     start_date = self._parse_start_date(start_date_value)
#                     if is_debug_target:
#                         print(f"  Start Date (Col H): {start_date.strftime('%Y-%m-%d') if start_date else 'None'}\n")
                
#                 # Determine historical series (handles reference product logic from column G)
#                 historical_series = self._determine_historical_series(idx, row, datos)
                
#                 # Process each forecast month
#                 for i, forecast_date in enumerate(datos['forecast_dates']):
                    
#                     if is_debug_target:
#                         print(f"  --- Forecast Month {i+1}/{len(datos['forecast_dates'])}: {forecast_date.strftime('%Y-%m')} ---")
                    
#                     # Check if we should forecast in this month based on start date
#                     if not self._should_forecast_in_month(start_date, forecast_date):
#                         if is_debug_target:
#                             print(f"    SKIP: Before start date\n")
#                         continue
                    
#                     # Get logic from cache
#                     calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                         customer, product_class, forecast_date
#                     )
                    
#                     if is_debug_target:
#                         print(f"    Calc Base: {calc_base}")
#                         print(f"    P2P: {p2p_model if pd.notna(p2p_model) else 'None'}")
#                         print(f"    Launch: {launch_month if pd.notna(launch_month) else 'None'}")
                    
#                     # Handle launch dependent
#                     if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 if is_debug_target:
#                                     print(f"    SKIP: Before launch date\n")
#                                 continue
                    
#                     # Handle P2P
#                     if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         # Get historical series (from P2P or already determined series)
#                         if pd.notna(p2p_model) and p2p_model != '':
#                             series_to_use = self._get_p2p_series_cached(customer, p2p_model, datos)
#                             if is_debug_target:
#                                 print(f"    Using P2P series: {p2p_model}")
#                         else:
#                             series_to_use = historical_series
#                             if is_debug_target:
#                                 print(f"    Using own/reference series")
                        
#                         if len(series_to_use) > 0 and not series_to_use.isna().all():
#                             # Get seasonal window
#                             seasonal_values = self._get_seasonal_window(forecast_date, series_to_use)
                            
#                             # DEBUG: Show seasonal lookup details
#                             if is_debug_target:
#                                 lookup_dates = [
#                                     forecast_date - relativedelta(years=1),
#                                     forecast_date - relativedelta(years=1) + relativedelta(months=1),
#                                     forecast_date - relativedelta(years=1) + relativedelta(months=2)
#                                 ]
#                                 print(f"    Seasonal lookup:")
#                                 for ld in lookup_dates:
#                                     val = series_to_use.get(ld, np.nan)
#                                     exists = "✅" if ld in series_to_use.index and pd.notna(val) and val > 0 else "❌"
#                                     print(f"      {ld.strftime('%Y-%m')}: {exists} {val if pd.notna(val) else 'N/A'}")
#                                 print(f"    Seasonal values found: {seasonal_values}")
                            
#                             if len(seasonal_values) > 0:
#                                 seasonal_avg = np.mean(seasonal_values)
                                
#                                 # Get growth factor from cache
#                                 year = forecast_date.year
#                                 growth_factor = self._get_growth_factor_cached(customer, year)
                                
#                                 forecasted_value = seasonal_avg * growth_factor
#                                 forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
#                                 cells_updated += 1
                                
#                                 if is_debug_target:
#                                     print(f"    Avg: {seasonal_avg:.2f} × Factor: {growth_factor} = {forecasted_value:.2f}\n")
#                             else:
#                                 # Fallback to simple average
#                                 simple_avg = series_to_use.mean()
#                                 if pd.notna(simple_avg) and simple_avg > 0:
#                                     year = forecast_date.year
#                                     growth_factor = self._get_growth_factor_cached(customer, year)
#                                     forecasted_value = simple_avg * growth_factor
#                                     forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
#                                     cells_updated += 1
                                    
#                                     if is_debug_target:
#                                         print(f"    FALLBACK: Avg={simple_avg:.2f} × Factor={growth_factor} = {forecasted_value:.2f}\n")
                
#                 processed_products += 1
                
#                 # Update progress every 100 products
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             # Clear progress bar
#             progress_bar.empty()
            
#             # Add forecast columns to result
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if skipped_no_calc > 0:
#                 st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'seasonal_moving_average',
#                     'window': 'same_month_last_year + 2_months',
#                     'growth_factors_applied': True,
#                     'reference_product_logic': True,
#                     'start_date_restriction': True,
#                     'skipped_no_calc': skipped_no_calc,
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
    
#     def _ejecutar_suavizacao(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute Exponential Smoothing with classification check and reference product logic"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Progress bar
#             progress_bar = st.progress(0)
#             total_products = len(df_main)
            
#             # Process each product row
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get(customer_col, '')
#                 product_model = row.get(model_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # CRITICAL: Check if product has "Não calcula" in ANY month - skip entirely if true
#                 if self._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     continue
                
#                 # Get logic (use first forecast month)
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Determine historical series (handles reference product logic from column G)
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     historical_series = self._determine_historical_series(idx, row, datos)
                
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
#                             if pd.notna(last_value) and last_value > 0:
#                                 for forecast_date in datos['forecast_dates']:
#                                     forecast_arrays[forecast_date][idx] = max(0, last_value)
#                                     cells_updated += 1
#                                 processed_products += 1
                    
#                     except Exception:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                
#                 # Update progress
#                 if processed_products % 100 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if skipped_no_calc > 0:
#                 st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
#                     'seasonal_periods': 12,
#                     'reference_product_logic': True,
#                     'skipped_no_calc': skipped_no_calc,
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
    
#     def _ejecutar_arima(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute SARIMA with classification check and reference product logic"""
        
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             arima_params = self.parametros.get('arima_params', (1, 1, 1))
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 st.error("❌ Could not find required columns in Main sheet")
#                 return self._resultado_vacio()
            
#             # Pre-allocate with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Progress bar
#             progress_bar = st.progress(0)
#             total_products = len(df_main)
            
#             # Process each product
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get(customer_col, '')
#                 product_model = row.get(model_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # CRITICAL: Check if product has "Não calcula" in ANY month - skip entirely if true
#                 if self._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     continue
                
#                 # Get logic
#                 calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Determine historical series (handles reference product logic from column G)
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     historical_series = self._determine_historical_series(idx, row, datos)
                
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
                    
#                     except Exception:
#                         # Fallback
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                
#                 # Update progress
#                 if processed_products % 50 == 0:
#                     progress_bar.progress(min(processed_products / total_products, 1.0))
            
#             progress_bar.empty()
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if skipped_no_calc > 0:
#                 st.info(f"ℹ️ Skipped {skipped_no_calc} products with 'Não calcula' classification")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
#                     'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
#                     'reference_product_logic': True,
#                     'skipped_no_calc': skipped_no_calc,
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
#         except Exception:
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
Forecasting processing engine - Modular version with FIXED column detection
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Tuple
import time

# Import models
from models import MovingAverageModel, ExponentialSmoothingModel, ARIMAModel


class ForecastProcessor:
    """Main class for forecasting processing - Modular and Optimized"""
    
    # Calculation base options
    CALC_BASE_P2P = "DE PARA SEGUINTE"
    CALC_BASE_NO_CALC = "Não calcula"
    CALC_BASE_LAUNCH_DEPENDENT = "Depende do mês de Lançamento"
    
    # DEBUG: Target record
    DEBUG_TARGET_MODEL = "4145766"
    DEBUG_TARGET_CUSTOMER = "ALPASHOP"
    
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
        self._reference_series_cache = {}
        
        # Initialize models
        self.moving_average_model = MovingAverageModel(self)
        self.exponential_smoothing_model = ExponentialSmoothingModel(self)
        self.arima_model = ARIMAModel(self)
        
        # DEBUG: Log initialization
        print(f"\n{'='*80}")
        print(f"🔍 FORECAST PROCESSOR INITIALIZED")
        print(f"  Forecast Start: {self.forecast_start_date.strftime('%Y-%m-%d')}")
        print(f"  Forecast Months: {self.forecast_months}")
        if len(self.date_columns) > 0:
            print(f"  Historical Range: {self.date_columns[0].strftime('%Y-%m')} to {self.date_columns[-1].strftime('%Y-%m')}")
        print(f"{'='*80}\n")
    
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
        
        CRITICAL FIX: Enhanced detection for LogicsxMonth columns with MORE variations
        """
        
        # EXPANDED column variations (Portuguese + Spanish + English)
        column_variations = {
            'customer': ['ID Cliente', 'ID Customer', 'Customer', 'Cliente', 'ID_Customer', 'IDCustomer'],
            'product_model': ['Modelo do Produto', 'Product Model', 'Model', 'Modelo', 'Product_Model'],
            'class': ['Classe', 'Class', 'CLASSE', 'CLASS', 'Clase'],
            'description': ['Descrição do Produto', 'Product Description', 'Description', 'Descrição', 'Product_Description'],
            'calc_base': ['Base de Cálculo', 'Base de Calculo', 'Calculation Base', 'Base', 'Calculation_Base', 'Base Calculo'],
            'month': ['Mês', 'Mes', 'Month', 'MÊS', 'MONTH', 'MES'],
            'forecast_start_month': [
                'Mês Início Previsão', 'Mes Inicio Prevision', 'Forecast Start Month', 
                'Mes Inicio', 'Month Start', 'Mês Início', 'Mes de forecast',
                'Início Previsão', 'Start Month', 'Forecast Start'
            ],
            'p2p': ['P2P', 'p2p', 'DE PARA', 'De Para'],
            'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Mes de Lanzamiento', 'Launch_Month'],
            'reference_model': ['Modelo de Referência', 'Modelo de Referencia', 'Reference Model', 'Referencia', 'Reference'],
            'start_date': ['Data Início', 'Data Inicio', 'Start Date', 'XF', 'Início', 'Inicio']
        }
        
        # Map column indices for Main sheet (known structure)
        main_cols = list(self.df_main.columns)
        
        if len(main_cols) > 0:
            self.col_names['main_customer'] = main_cols[0]  # Column A
        if len(main_cols) > 1:
            self.col_names['main_product_model'] = main_cols[1]  # Column B
        if len(main_cols) > 5:
            self.col_names['main_class'] = main_cols[5]  # Column F
        if len(main_cols) > 6:
            self.col_names['main_reference_model'] = main_cols[6]  # Column G
        if len(main_cols) > 7:
            self.col_names['main_start_date'] = main_cols[7]  # Column H (XF)
        
        # Detect in Logics DataFrame - WITH DEBUG
        print(f"\n{'='*80}")
        print(f"🔍 DETECTING LogicsxMonth COLUMNS")
        print(f"  Available columns: {list(self.df_logics.columns)}")
        
        for key, variations in column_variations.items():
            for var in variations:
                if var in self.df_logics.columns:
                    self.col_names[f'logics_{key}'] = var
                    print(f"  ✅ {key}: '{var}'")
                    break
            else:
                if key in ['forecast_start_month', 'class', 'month', 'calc_base']:
                    print(f"  ❌ {key}: NOT FOUND")
        
        # FALLBACK: If critical columns not found, use column indices
        if 'logics_forecast_start_month' not in self.col_names and len(self.df_logics.columns) > 1:
            self.col_names['logics_forecast_start_month'] = self.df_logics.columns[1]  # Column B
            print(f"  ⚠️ forecast_start_month: AUTO-DETECTED as Column B = '{self.df_logics.columns[1]}'")
        
        if 'logics_class' not in self.col_names and len(self.df_logics.columns) > 2:
            self.col_names['logics_class'] = self.df_logics.columns[2]  # Column C
            print(f"  ⚠️ class: AUTO-DETECTED as Column C = '{self.df_logics.columns[2]}'")
        
        if 'logics_month' not in self.col_names and len(self.df_logics.columns) > 3:
            self.col_names['logics_month'] = self.df_logics.columns[3]  # Column D
            print(f"  ⚠️ month: AUTO-DETECTED as Column D = '{self.df_logics.columns[3]}'")
        
        if 'logics_calc_base' not in self.col_names and len(self.df_logics.columns) > 4:
            self.col_names['logics_calc_base'] = self.df_logics.columns[4]  # Column E
            print(f"  ⚠️ calc_base: AUTO-DETECTED as Column E = '{self.df_logics.columns[4]}'")
        
        print(f"{'='*80}\n")
        
        # Detect in Relations DataFrame
        for key, variations in column_variations.items():
            for var in variations:
                if var in self.df_relations.columns:
                    self.col_names[f'relations_{key}'] = var
                    break
        
        st.info(f"🔍 Detected {len(self.col_names)} column mappings")
    
    def _parse_start_date(self, value) -> datetime:
        """Parse start date from column H in format 01/mm/yyyy or other formats"""
        if pd.isna(value):
            return None
        
        try:
            if isinstance(value, (datetime, pd.Timestamp)):
                return pd.to_datetime(value)
            
            if isinstance(value, str):
                value = value.strip()
                try:
                    return pd.to_datetime(value, format='%d/%m/%Y')
                except:
                    pass
                try:
                    return pd.to_datetime(value, dayfirst=True)
                except:
                    pass
            
            return None
        except:
            return None
    
    def _should_forecast_in_month(self, start_date: datetime, forecast_date: datetime) -> bool:
        """Check if forecast should be applied in this month based on start date"""
        if start_date is None:
            return True
        
        start_normalized = datetime(start_date.year, start_date.month, 1)
        forecast_normalized = datetime(forecast_date.year, forecast_date.month, 1)
        
        return forecast_normalized >= start_normalized
    
    def _has_no_calc_in_any_month(self, customer: str, product_model: str, product_class: str, forecast_dates: List[datetime]) -> bool:
        """
        Check if product has "Não calcula" classification in ANY forecast month
        
        DEBUG: Logs for target model
        """
        is_debug_target = (str(product_model).strip() == self.DEBUG_TARGET_MODEL and 
                          str(customer).strip() == self.DEBUG_TARGET_CUSTOMER)
        
        if is_debug_target:
            print(f"\n{'='*80}")
            print(f"🔍 CHECK NO CALC")
            print(f"  Model: {product_model}, Class: {product_class}")
        
        for forecast_date in forecast_dates:
            calc_base, _, _ = self._get_logic_for_product_cached(customer, product_class, forecast_date)
            
            if is_debug_target:
                print(f"  {forecast_date.strftime('%Y-%m')}: {calc_base}")
            
            if calc_base == self.CALC_BASE_NO_CALC:
                if is_debug_target:
                    print(f"  ❌ FOUND 'Não calcula' - SKIP")
                    print(f"{'='*80}\n")
                return True
        
        if is_debug_target:
            print(f"  ✅ No 'Não calcula' - PROCESS")
            print(f"{'='*80}\n")
        
        return False
    
    def _get_reference_product_series(self, customer: str, reference_model: str, datos: Dict) -> pd.Series:
        """Get historical series for reference product"""
        cache_key = (customer, reference_model)
        
        if cache_key in self._reference_series_cache:
            return self._reference_series_cache[cache_key]
        
        df_main = datos['df_main']
        col_names = datos['col_names']
        
        customer_col = col_names.get('main_customer')
        model_col = col_names.get('main_product_model')
        
        if not customer_col or not model_col:
            return pd.Series(dtype=float)
        
        try:
            mask = (
                (df_main[customer_col].astype(str).str.strip() == str(customer).strip()) &
                (df_main[model_col].astype(str).str.strip() == str(reference_model).strip())
            )
            
            matching_rows = df_main[mask]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                series = pd.Series(dtype=float)
                
                for date_col in datos['date_columns']:
                    val = row.get(date_col, np.nan)
                    val = pd.to_numeric(val, errors='coerce')
                    series[date_col] = val
                
                self._reference_series_cache[cache_key] = series
                return series
        except Exception:
            pass
        
        return pd.Series(dtype=float)
    
    def _determine_historical_series(self, idx: int, row: pd.Series, datos: Dict) -> pd.Series:
        """Determine which historical series to use based on column G (reference model)"""
        col_names = datos['col_names']
        customer_col = col_names.get('main_customer')
        model_col = col_names.get('main_product_model')
        reference_col = col_names.get('main_reference_model')
        
        customer = row.get(customer_col, '')
        product_model = row.get(model_col, '')
        
        is_debug_target = (str(product_model).strip() == self.DEBUG_TARGET_MODEL and 
                          str(customer).strip() == self.DEBUG_TARGET_CUSTOMER)
        
        if is_debug_target:
            print(f"\n{'='*80}")
            print(f"🔍 DETERMINE SERIES")
            print(f"  Model: {product_model}")
        
        # Check if there's a reference product in column G
        reference_model = None
        if reference_col:
            reference_value = row.get(reference_col)
            if pd.notna(reference_value):
                reference_str = str(reference_value).strip()
                product_model_str = str(product_model).strip()
                
                if is_debug_target:
                    print(f"  Ref (Col G): '{reference_str}'")
                    print(f"  Current (Col B): '{product_model_str}'")
                
                if reference_str != '' and reference_str != '-' and reference_str != product_model_str:
                    reference_model = reference_str
                    if is_debug_target:
                        print(f"  ✅ USE REFERENCE: {reference_model}")
        
        # Get historical series
        if reference_model:
            historical_series = self._get_reference_product_series(customer, reference_model, datos)
            if len(historical_series) == 0 or historical_series.isna().all():
                if is_debug_target:
                    print(f"  ⚠️ Ref not found, fallback to OWN")
                historical_series = self._get_historical_series_indexed(idx, datos)
        else:
            if is_debug_target:
                print(f"  ✅ USE OWN series")
            historical_series = self._get_historical_series_indexed(idx, datos)
        
        if is_debug_target:
            non_null = historical_series.dropna()
            print(f"  Series: {len(non_null)} values")
            if len(non_null) > 0:
                print(f"  Range: {non_null.index.min().strftime('%Y-%m')} to {non_null.index.max().strftime('%Y-%m')}")
            print(f"{'='*80}\n")
        
        return historical_series
    
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
            with st.spinner("📈 Executing Moving Average..."):
                start_time = time.time()
                resultados['media_movil'] = self.moving_average_model.execute(datos_preparados)
                resultados['media_movil']['metadata']['execution_time'] = time.time() - start_time
        
        # Execute Exponential Smoothing
        if modelos_ejecutar.get('suavizacao_exponencial', False):
            with st.spinner("📊 Executing Exponential Smoothing..."):
                start_time = time.time()
                resultados['suavizacao_exponencial'] = self.exponential_smoothing_model.execute(datos_preparados)
                resultados['suavizacao_exponencial']['metadata']['execution_time'] = time.time() - start_time
        
        # Execute ARIMA
        if modelos_ejecutar.get('arima', False):
            with st.spinner("🔬 Executing SARIMA..."):
                start_time = time.time()
                resultados['arima'] = self.arima_model.execute(datos_preparados)
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
        """Pre-build caches with CORRECTED structure for forecast_start_month"""
        
        df_logics = datos['df_logics']
        df_relations = datos['df_relations']
        col_names = datos['col_names']
        
        print(f"\n{'='*80}")
        print(f"🔍 BUILDING CACHES")
        
        # Cache 1: Logic lookup
        forecast_start_col = col_names.get('logics_forecast_start_month')
        class_col = col_names.get('logics_class')
        month_col = col_names.get('logics_month')
        calc_base_col = col_names.get('logics_calc_base')
        p2p_col = col_names.get('logics_p2p')
        launch_col = col_names.get('logics_launch_month')
        
        print(f"  Columns:")
        print(f"    Forecast Start: {forecast_start_col}")
        print(f"    Class: {class_col}")
        print(f"    Month: {month_col}")
        print(f"    Calc Base: {calc_base_col}")
        
        if class_col and month_col and class_col in df_logics.columns and month_col in df_logics.columns:
            
            total_rows = 0
            matching_count = 0
            
            for idx, row in df_logics.iterrows():
                total_rows += 1
                
                product_class = row.get(class_col)
                month_date = pd.to_datetime(row.get(month_col)).replace(day=1)
                
                # Get forecast start month
                forecast_start_month = None
                if forecast_start_col and forecast_start_col in df_logics.columns:
                    forecast_start_value = row.get(forecast_start_col)
                    if pd.notna(forecast_start_value):
                        try:
                            forecast_start_month = pd.to_datetime(forecast_start_value).replace(day=1)
                        except:
                            pass
                
                # Build cache key
                if forecast_start_month:
                    cache_key = (forecast_start_month, product_class, month_date)
                    
                    if forecast_start_month == self.forecast_start_date:
                        matching_count += 1
                else:
                    cache_key = (product_class, month_date)
                
                calc_base_value = row.get(calc_base_col, self.CALC_BASE_P2P) if calc_base_col else self.CALC_BASE_P2P
                p2p_value = row.get(p2p_col, None) if p2p_col else None
                launch_value = row.get(launch_col, None) if launch_col else None
                
                self._logic_cache[cache_key] = (calc_base_value, p2p_value, launch_value)
            
            print(f"  ✅ Logic cache:")
            print(f"    Total entries: {len(self._logic_cache)}")
            print(f"    Total rows: {total_rows}")
            print(f"    Matching forecast start: {matching_count}")
        else:
            print(f"  ❌ Could not build logic cache - missing columns")
        
        # Cache 2: Growth factors
        customer_col = col_names.get('relations_customer')
        
        if customer_col and customer_col in df_relations.columns:
            for idx, row in df_relations.iterrows():
                customer = row.get(customer_col)
                
                for col in df_relations.columns:
                    is_year_column = False
                    
                    if isinstance(col, str) and col.isdigit():
                        is_year_column = True
                        year = int(col)
                    elif isinstance(col, int) and 2000 <= col <= 2100:
                        is_year_column = True
                        year = col
                    
                    if is_year_column:
                        factor = row.get(col)
                        
                        if pd.notna(factor) and isinstance(factor, (int, float)):
                            cache_key = (customer, year)
                            self._growth_factor_cache[cache_key] = float(factor)
            
            print(f"  ✅ Growth factor cache: {len(self._growth_factor_cache)} entries")
        
        print(f"{'='*80}\n")
    
    def _get_logic_for_product_cached(self, customer: str, product_class: str, month_date: datetime) -> Tuple[str, str, Any]:
        """Get logic from cache with fallback"""
        
        month_start = pd.to_datetime(month_date).replace(day=1)
        
        # Try new structure: (forecast_start, class, month)
        cache_key_new = (self.forecast_start_date, product_class, month_start)
        
        if cache_key_new in self._logic_cache:
            return self._logic_cache[cache_key_new]
        
        # Fallback: (class, month)
        cache_key_old = (product_class, month_start)
        
        if cache_key_old in self._logic_cache:
            return self._logic_cache[cache_key_old]
        
        # Default
        return self.CALC_BASE_P2P, None, None
    
    def _get_growth_factor_cached(self, customer: str, year: int) -> float:
        """Get growth factor from cache"""
        cache_key = (customer, year)
        return self._growth_factor_cache.get(cache_key, 1.0)
    
    def _get_p2p_series_cached(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
        """Get P2P series from cache"""
        cache_key = (customer, p2p_model)
        
        if cache_key in self._p2p_series_cache:
            return self._p2p_series_cache[cache_key]
        
        series = self._get_p2p_series_indexed(customer, p2p_model, datos)
        self._p2p_series_cache[cache_key] = series
        
        return series
    
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
        except Exception:
            pass
        
        return pd.Series(dtype=float)