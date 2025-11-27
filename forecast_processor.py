# """
# Forecasting processing engine - Modular version with FIXED column detection
# """
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# from typing import Dict, Any, List, Tuple
# import time

# # Import models
# from models import MovingAverageModel, ExponentialSmoothingModel, ARIMAModel


# class ForecastProcessor:
#     """Main class for forecasting processing - Modular and Optimized"""
    
#     # Calculation base options
#     CALC_BASE_P2P = "DE PARA SEGUINTE"
#     CALC_BASE_NO_CALC = "Não calcula"
#     CALC_BASE_LAUNCH_DEPENDENT = "Depende do mês de Lançamento"
    
#     # DEBUG: Target record
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
        
#         # Initialize models
#         self.moving_average_model = MovingAverageModel(self)
#         self.exponential_smoothing_model = ExponentialSmoothingModel(self)
#         self.arima_model = ARIMAModel(self)
        
#         # DEBUG: Log initialization
#         print(f"\n{'='*80}")
#         print(f"🔍 FORECAST PROCESSOR INITIALIZED")
#         print(f"  Forecast Start: {self.forecast_start_date.strftime('%Y-%m-%d')}")
#         print(f"  Forecast Months: {self.forecast_months}")
#         if len(self.date_columns) > 0:
#             print(f"  Historical Range: {self.date_columns[0].strftime('%Y-%m')} to {self.date_columns[-1].strftime('%Y-%m')}")
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
#         """
#         Detect actual column names in DataFrames and create mapping
        
#         CRITICAL FIX: Enhanced detection for LogicsxMonth columns with MORE variations
#         """
        
#         # EXPANDED column variations (Portuguese + Spanish + English)
#         column_variations = {
#             'customer': ['ID Cliente', 'ID Customer', 'Customer', 'Cliente', 'ID_Customer', 'IDCustomer'],
#             'product_model': ['Modelo do Produto', 'Product Model', 'Model', 'Modelo', 'Product_Model'],
#             'class': ['Classe', 'Class', 'CLASSE', 'CLASS', 'Clase'],
#             'description': ['Descrição do Produto', 'Product Description', 'Description', 'Descrição', 'Product_Description'],
#             'calc_base': ['Base de Cálculo', 'Base de Calculo', 'Calculation Base', 'Base', 'Calculation_Base', 'Base Calculo'],
#             'month': ['Mês', 'Mes', 'Month', 'MÊS', 'MONTH', 'MES'],
#             'forecast_start_month': [
#                 'Mês Início Previsão', 'Mes Inicio Prevision', 'Forecast Start Month', 
#                 'Mes Inicio', 'Month Start', 'Mês Início', 'Mes de forecast',
#                 'Início Previsão', 'Start Month', 'Forecast Start'
#             ],
#             'p2p': ['P2P', 'p2p', 'DE PARA', 'De Para'],
#             'launch_month': ['XF', 'Launch Month', 'Mês de Lançamento', 'Mes de Lanzamiento', 'Launch_Month'],
#             'reference_model': ['Modelo de Referência', 'Modelo de Referencia', 'Reference Model', 'Referencia', 'Reference'],
#             'start_date': ['Data Início', 'Data Inicio', 'Start Date', 'XF', 'Início', 'Inicio']
#         }
        
#         # Map column indices for Main sheet (known structure)
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
        
#         # Detect in Logics DataFrame - WITH DEBUG
#         print(f"\n{'='*80}")
#         print(f"🔍 DETECTING LogicsxMonth COLUMNS")
#         print(f"  Available columns: {list(self.df_logics.columns)}")
        
#         for key, variations in column_variations.items():
#             for var in variations:
#                 if var in self.df_logics.columns:
#                     self.col_names[f'logics_{key}'] = var
#                     print(f"  ✅ {key}: '{var}'")
#                     break
#             else:
#                 if key in ['forecast_start_month', 'class', 'month', 'calc_base']:
#                     print(f"  ❌ {key}: NOT FOUND")
        
#         # FALLBACK: If critical columns not found, use column indices
#         if 'logics_forecast_start_month' not in self.col_names and len(self.df_logics.columns) > 1:
#             self.col_names['logics_forecast_start_month'] = self.df_logics.columns[1]  # Column B
#             print(f"  ⚠️ forecast_start_month: AUTO-DETECTED as Column B = '{self.df_logics.columns[1]}'")
        
#         if 'logics_class' not in self.col_names and len(self.df_logics.columns) > 2:
#             self.col_names['logics_class'] = self.df_logics.columns[2]  # Column C
#             print(f"  ⚠️ class: AUTO-DETECTED as Column C = '{self.df_logics.columns[2]}'")
        
#         if 'logics_month' not in self.col_names and len(self.df_logics.columns) > 3:
#             self.col_names['logics_month'] = self.df_logics.columns[3]  # Column D
#             print(f"  ⚠️ month: AUTO-DETECTED as Column D = '{self.df_logics.columns[3]}'")
        
#         if 'logics_calc_base' not in self.col_names and len(self.df_logics.columns) > 4:
#             self.col_names['logics_calc_base'] = self.df_logics.columns[4]  # Column E
#             print(f"  ⚠️ calc_base: AUTO-DETECTED as Column E = '{self.df_logics.columns[4]}'")
        
#         print(f"{'='*80}\n")
        
#         # Detect in Relations DataFrame
#         for key, variations in column_variations.items():
#             for var in variations:
#                 if var in self.df_relations.columns:
#                     self.col_names[f'relations_{key}'] = var
#                     break
        
#         st.info(f"🔍 Detected {len(self.col_names)} column mappings")
    
#     def _parse_start_date(self, value) -> datetime:
#         """Parse start date from column H in format 01/mm/yyyy or other formats"""
#         if pd.isna(value):
#             return None
        
#         try:
#             if isinstance(value, (datetime, pd.Timestamp)):
#                 return pd.to_datetime(value)
            
#             if isinstance(value, str):
#                 value = value.strip()
#                 try:
#                     return pd.to_datetime(value, format='%d/%m/%Y')
#                 except:
#                     pass
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
        
#         start_normalized = datetime(start_date.year, start_date.month, 1)
#         forecast_normalized = datetime(forecast_date.year, forecast_date.month, 1)
        
#         return forecast_normalized >= start_normalized
    
#     def _has_no_calc_in_any_month(self, customer: str, product_model: str, product_class: str, forecast_dates: List[datetime]) -> bool:
#         """
#         Check if product has "Não calcula" classification in ANY forecast month
        
#         DEBUG: Logs for target model
#         """
#         is_debug_target = (str(product_model).strip() == self.DEBUG_TARGET_MODEL and 
#                           str(customer).strip() == self.DEBUG_TARGET_CUSTOMER)
        
#         if is_debug_target:
#             print(f"\n{'='*80}")
#             print(f"🔍 CHECK NO CALC")
#             print(f"  Model: {product_model}, Class: {product_class}")
        
#         for forecast_date in forecast_dates:
#             calc_base, _, _ = self._get_logic_for_product_cached(customer, product_class, forecast_date)
            
#             if is_debug_target:
#                 print(f"  {forecast_date.strftime('%Y-%m')}: {calc_base}")
            
#             if calc_base == self.CALC_BASE_NO_CALC:
#                 if is_debug_target:
#                     print(f"  ❌ FOUND 'Não calcula' - SKIP")
#                     print(f"{'='*80}\n")
#                 return True
        
#         if is_debug_target:
#             print(f"  ✅ No 'Não calcula' - PROCESS")
#             print(f"{'='*80}\n")
        
#         return False
    
#     def _get_reference_product_series(self, customer: str, reference_model: str, datos: Dict) -> pd.Series:
#         """Get historical series for reference product"""
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
#         """Determine which historical series to use based on column G (reference model)"""
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
#             print(f"🔍 DETERMINE SERIES")
#             print(f"  Model: {product_model}")
        
#         # Check if there's a reference product in column G
#         reference_model = None
#         if reference_col:
#             reference_value = row.get(reference_col)
#             if pd.notna(reference_value):
#                 reference_str = str(reference_value).strip()
#                 product_model_str = str(product_model).strip()
                
#                 if is_debug_target:
#                     print(f"  Ref (Col G): '{reference_str}'")
#                     print(f"  Current (Col B): '{product_model_str}'")
                
#                 if reference_str != '' and reference_str != '-' and reference_str != product_model_str:
#                     reference_model = reference_str
#                     if is_debug_target:
#                         print(f"  ✅ USE REFERENCE: {reference_model}")
        
#         # Get historical series
#         if reference_model:
#             historical_series = self._get_reference_product_series(customer, reference_model, datos)
#             if len(historical_series) == 0 or historical_series.isna().all():
#                 if is_debug_target:
#                     print(f"  ⚠️ Ref not found, fallback to OWN")
#                 historical_series = self._get_historical_series_indexed(idx, datos)
#         else:
#             if is_debug_target:
#                 print(f"  ✅ USE OWN series")
#             historical_series = self._get_historical_series_indexed(idx, datos)
        
#         if is_debug_target:
#             non_null = historical_series.dropna()
#             print(f"  Series: {len(non_null)} values")
#             if len(non_null) > 0:
#                 print(f"  Range: {non_null.index.min().strftime('%Y-%m')} to {non_null.index.max().strftime('%Y-%m')}")
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
#                 resultados['media_movil'] = self.moving_average_model.execute(datos_preparados)
#                 resultados['media_movil']['metadata']['execution_time'] = time.time() - start_time
        
#         # Execute Exponential Smoothing
#         if modelos_ejecutar.get('suavizacao_exponencial', False):
#             with st.spinner("📊 Executing Exponential Smoothing..."):
#                 start_time = time.time()
#                 resultados['suavizacao_exponencial'] = self.exponential_smoothing_model.execute(datos_preparados)
#                 resultados['suavizacao_exponencial']['metadata']['execution_time'] = time.time() - start_time
        
#         # Execute ARIMA
#         if modelos_ejecutar.get('arima', False):
#             with st.spinner("🔬 Executing SARIMA..."):
#                 start_time = time.time()
#                 resultados['arima'] = self.arima_model.execute(datos_preparados)
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
#         """Pre-build caches with CORRECTED structure for forecast_start_month"""
        
#         df_logics = datos['df_logics']
#         df_relations = datos['df_relations']
#         col_names = datos['col_names']
        
#         print(f"\n{'='*80}")
#         print(f"🔍 BUILDING CACHES")
        
#         # Cache 1: Logic lookup
#         forecast_start_col = col_names.get('logics_forecast_start_month')
#         class_col = col_names.get('logics_class')
#         month_col = col_names.get('logics_month')
#         calc_base_col = col_names.get('logics_calc_base')
#         p2p_col = col_names.get('logics_p2p')
#         launch_col = col_names.get('logics_launch_month')
        
#         print(f"  Columns:")
#         print(f"    Forecast Start: {forecast_start_col}")
#         print(f"    Class: {class_col}")
#         print(f"    Month: {month_col}")
#         print(f"    Calc Base: {calc_base_col}")
        
#         if class_col and month_col and class_col in df_logics.columns and month_col in df_logics.columns:
            
#             total_rows = 0
#             matching_count = 0
            
#             for idx, row in df_logics.iterrows():
#                 total_rows += 1
                
#                 product_class = row.get(class_col)
#                 month_date = pd.to_datetime(row.get(month_col)).replace(day=1)
                
#                 # Get forecast start month
#                 forecast_start_month = None
#                 if forecast_start_col and forecast_start_col in df_logics.columns:
#                     forecast_start_value = row.get(forecast_start_col)
#                     if pd.notna(forecast_start_value):
#                         try:
#                             forecast_start_month = pd.to_datetime(forecast_start_value).replace(day=1)
#                         except:
#                             pass
                
#                 # Build cache key
#                 if forecast_start_month:
#                     cache_key = (forecast_start_month, product_class, month_date)
                    
#                     if forecast_start_month == self.forecast_start_date:
#                         matching_count += 1
#                 else:
#                     cache_key = (product_class, month_date)
                
#                 calc_base_value = row.get(calc_base_col, self.CALC_BASE_P2P) if calc_base_col else self.CALC_BASE_P2P
#                 p2p_value = row.get(p2p_col, None) if p2p_col else None
#                 launch_value = row.get(launch_col, None) if launch_col else None
                
#                 self._logic_cache[cache_key] = (calc_base_value, p2p_value, launch_value)
            
#             print(f"  ✅ Logic cache:")
#             print(f"    Total entries: {len(self._logic_cache)}")
#             print(f"    Total rows: {total_rows}")
#             print(f"    Matching forecast start: {matching_count}")
#         else:
#             print(f"  ❌ Could not build logic cache - missing columns")
        
#         # Cache 2: Growth factors
#         customer_col = col_names.get('relations_customer')
        
#         if customer_col and customer_col in df_relations.columns:
#             for idx, row in df_relations.iterrows():
#                 customer = row.get(customer_col)
                
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
            
#             print(f"  ✅ Growth factor cache: {len(self._growth_factor_cache)} entries")
        
#         print(f"{'='*80}\n")
    
#     def _get_logic_for_product_cached(self, customer: str, product_class: str, month_date: datetime) -> Tuple[str, str, Any]:
#         """Get logic from cache with fallback"""
        
#         month_start = pd.to_datetime(month_date).replace(day=1)
        
#         # Try new structure: (forecast_start, class, month)
#         cache_key_new = (self.forecast_start_date, product_class, month_start)
        
#         if cache_key_new in self._logic_cache:
#             return self._logic_cache[cache_key_new]
        
#         # Fallback: (class, month)
#         cache_key_old = (product_class, month_start)
        
#         if cache_key_old in self._logic_cache:
#             return self._logic_cache[cache_key_old]
        
#         # Default
#         return self.CALC_BASE_P2P, None, None
    
#     def _get_growth_factor_cached(self, customer: str, year: int) -> float:
#         """Get growth factor from cache"""
#         cache_key = (customer, year)
#         return self._growth_factor_cache.get(cache_key, 1.0)
    
#     def _get_p2p_series_cached(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
#         """Get P2P series from cache"""
#         cache_key = (customer, p2p_model)
        
#         if cache_key in self._p2p_series_cache:
#             return self._p2p_series_cache[cache_key]
        
#         series = self._get_p2p_series_indexed(customer, p2p_model, datos)
#         self._p2p_series_cache[cache_key] = series
        
#         return series
    
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
    
    # DEBUG: Target record - UPDATED
    DEBUG_TARGET_MODEL = "TC0001973"
    DEBUG_TARGET_CUSTOMER = "ATACADO ESPECIALIZADO"
    
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
        print(f"  DEBUG TARGET: Model={self.DEBUG_TARGET_MODEL}, Customer={self.DEBUG_TARGET_CUSTOMER}")
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
            'class': ['Classe', 'Class', 'CLASSE', 'CLASS', 'Clase', 'Tipo de producto', 'Tipo'],
            'description': ['Descrição do Produto', 'Product Description', 'Description', 'Descrição', 'Product_Description'],
            'calc_base': ['Base de Cálculo', 'Base de Calculo', 'Calculation Base', 'Base', 'Calculation_Base', 'Base Calculo', 'Base de cálculo'],
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
        """
        Determine which historical series to use based on column G (reference model)
        
        ENHANCED DEBUG: Shows clearly if using reference or own series
        """
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
            print(f"🔍 DETERMINE HISTORICAL SERIES")
            print(f"  Current Model (Col B): '{product_model}'")
        
        # Check if there's a reference product in column G
        reference_model = None
        if reference_col:
            reference_value = row.get(reference_col)
            if pd.notna(reference_value):
                reference_str = str(reference_value).strip()
                product_model_str = str(product_model).strip()
                
                if is_debug_target:
                    print(f"  Reference Model (Col G): '{reference_str}'")
                
                if reference_str != '' and reference_str != '-' and reference_str != product_model_str:
                    reference_model = reference_str
                    if is_debug_target:
                        print(f"  ✅ DECISION: USE REFERENCE '{reference_model}'")
                        print(f"     (Will search for Customer={customer} + Model={reference_model})")
        
        # Get historical series
        if reference_model:
            historical_series = self._get_reference_product_series(customer, reference_model, datos)
            if len(historical_series) == 0 or historical_series.isna().all():
                if is_debug_target:
                    print(f"  ⚠️ REFERENCE NOT FOUND - Fallback to OWN series")
                historical_series = self._get_historical_series_indexed(idx, datos)
        else:
            if is_debug_target:
                print(f"  ✅ DECISION: USE OWN series (no reference)")
            historical_series = self._get_historical_series_indexed(idx, datos)
        
        if is_debug_target:
            non_null = historical_series.dropna()
            print(f"  RESULT:")
            print(f"    Total values: {len(non_null)}")
            if len(non_null) > 0:
                print(f"    Date range: {non_null.index.min().strftime('%Y-%m')} to {non_null.index.max().strftime('%Y-%m')}")
                print(f"    Sample: {dict(list(non_null.head(3).items()))}")
            else:
                print(f"    ❌ NO DATA FOUND")
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