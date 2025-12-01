# """
# ARIMA Forecasting Model
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, Any

# try:
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     SARIMAX_AVAILABLE = True
# except ImportError:
#     SARIMAX_AVAILABLE = False


# class ARIMAModel:
#     """SARIMA forecasting model"""
    
#     def __init__(self, processor):
#         """
#         Initialize model with reference to main processor
        
#         Args:
#             processor: ForecastProcessor instance
#         """
#         self.processor = processor
    
#     def execute(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute SARIMA forecasting
        
#         Args:
#             datos: Prepared data dictionary
            
#         Returns:
#             Dictionary with results
#         """
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             arima_params = self.processor.parametros.get('arima_params', (1, 1, 1))
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 print("❌ ERROR: Could not find required columns in Main sheet")
#                 return self._empty_result()
            
#             # Pre-allocate with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Process each product
#             for idx in range(len(df_main)):
#                 row = df_main.iloc[idx]
                
#                 customer = row.get(customer_col, '')
#                 product_model = row.get(model_col, '')
#                 product_class = row.get(class_col, '')
                
#                 if pd.isna(customer) or pd.isna(product_class):
#                     continue
                
#                 # Check if product has "Não calcula"
#                 if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     continue
                
#                 # Get logic
#                 calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Determine historical series
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     historical_series = self.processor._determine_historical_series(idx, row, datos)
                
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
#                         # Fallback to average
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
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
#             print(f"❌ ERROR in ARIMA: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return self._empty_result()
    
#     def _empty_result(self) -> Dict[str, Any]:
#         """Return empty result structure"""
#         return {
#             'dataframe': pd.DataFrame(),
#             'celulas_actualizadas': 0,
#             'parametros': {},
#             'metadata': {'error': True}
#         }






# """
# SARIMA Forecasting Model - FIXED with same logic as other models
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, Any, Optional

# try:
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     SARIMAX_AVAILABLE = True
# except ImportError:
#     SARIMAX_AVAILABLE = False


# class ARIMAModel:
#     """SARIMA forecasting model"""
    
#     # DEBUG: Target record
#     DEBUG_TARGET_MODEL = "TC0001973"
#     DEBUG_TARGET_CUSTOMER = "ATACADO ESPECIALIZADO"
    
#     def __init__(self, processor):
#         """
#         Initialize model with reference to main processor
        
#         Args:
#             processor: ForecastProcessor instance
#         """
#         self.processor = processor
    
#     def execute(self, datos: Dict[str, Any], moving_avg_result: Optional[Dict] = None) -> Dict[str, Any]:
#         """
#         Execute SARIMA forecasting
        
#         CRITICAL FIXES:
#         1. Accept zero values (>= 0 instead of > 0)
#         2. Extend series with Moving Average if insufficient data
#         3. Require minimum 24 months for SARIMA (2 seasonal cycles)
#         4. Follow same logic as Exponential Smoothing
        
#         Args:
#             datos: Prepared data dictionary
#             moving_avg_result: Results from Moving Average model (for series extension)
            
#         Returns:
#             Dictionary with results
#         """
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             arima_params = self.processor.parametros.get('arima_params', (1, 1, 1))
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
#             extended_series_count = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 print("❌ ERROR: Could not find required columns in Main sheet")
#                 return self._empty_result()
            
#             # Pre-allocate with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Get Moving Average forecast dataframe for series extension
#             moving_avg_df = None
#             if moving_avg_result and 'dataframe' in moving_avg_result:
#                 moving_avg_df = moving_avg_result['dataframe']
            
#             # Process each product
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
#                     print(f"🎯 [SARIMA] TARGET PRODUCT FOUND")
#                     print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
#                     print(f"  Row: {idx}")
#                     print(f"{'='*80}\n")
                
#                 # Check if product has "Não calcula"
#                 if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     if is_debug_target:
#                         print(f"  ⏭️ SKIPPED: Has 'Não calcula'\n")
#                     continue
                
#                 # Get logic (use first forecast month)
#                 calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 if is_debug_target:
#                     print(f"  Calc Base: {calc_base}")
#                     print(f"  P2P: {p2p_model if pd.notna(p2p_model) else 'None'}")
                
#                 # Determine historical series
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
#                     if is_debug_target:
#                         print(f"  Using P2P series: {p2p_model}")
#                 else:
#                     historical_series = self.processor._determine_historical_series(idx, row, datos)
#                     if is_debug_target:
#                         print(f"  Using own/reference series")
                
#                 # Check minimum data requirement (SARIMA needs 24+ for 2 seasonal cycles)
#                 non_null_count = len(historical_series.dropna())
                
#                 if is_debug_target:
#                     print(f"\n  DATA CHECK:")
#                     print(f"    Non-null values (historical): {non_null_count}")
#                     print(f"    Required minimum: 24 (2 seasonal cycles)")
                
#                 # CRITICAL FIX: Extend series with Moving Average if insufficient data
#                 if non_null_count < 24:
#                     if moving_avg_df is not None and is_debug_target:
#                         print(f"    ⚠️ INSUFFICIENT - Attempting to extend with Moving Average forecast...")
                    
#                     if moving_avg_df is not None:
#                         extended_series = historical_series.copy()
#                         extended_count = 0
                        
#                         # Add forecast values from Moving Average to extend the series
#                         for forecast_date in sorted(datos['forecast_dates']):
#                             if forecast_date not in extended_series.index:
#                                 # Get value from Moving Average result
#                                 if forecast_date in moving_avg_df.columns:
#                                     moving_avg_val = moving_avg_df.iloc[idx][forecast_date]
#                                     if pd.notna(moving_avg_val):
#                                         extended_series[forecast_date] = moving_avg_val
#                                         extended_count += 1
                        
#                         # Check if extension was successful
#                         new_non_null_count = len(extended_series.dropna())
                        
#                         if is_debug_target:
#                             print(f"    Extended with {extended_count} Moving Avg values")
#                             print(f"    New total: {new_non_null_count} values")
                        
#                         if new_non_null_count >= 24:
#                             historical_series = extended_series
#                             non_null_count = new_non_null_count
#                             extended_series_count += 1
                            
#                             if is_debug_target:
#                                 print(f"    ✅ EXTENSION SUCCESSFUL - Proceeding with SARIMA")
#                         else:
#                             if is_debug_target:
#                                 print(f"    ❌ Still insufficient ({new_non_null_count} < 24)")
                
#                 # Process if we have enough data
#                 if non_null_count >= 24:
                    
#                     if is_debug_target:
#                         print(f"    ✅ SUFFICIENT DATA - Proceeding with SARIMA forecast")
                    
#                     try:
#                         if SARIMAX_AVAILABLE:
#                             # CRITICAL FIX 1: Accept zeros (>= 0 instead of > 0)
#                             clean_series = historical_series.dropna()
#                             clean_series = clean_series[clean_series >= 0]
                            
#                             if is_debug_target:
#                                 print(f"    Cleaned series: {len(clean_series)} values >= 0")
                            
#                             if len(clean_series) >= 24:
#                                 if is_debug_target:
#                                     print(f"    Fitting SARIMA model with order={arima_params}, seasonal=(1,1,1,12)...")
                                
#                                 # Fit SARIMA model
#                                 model = SARIMAX(
#                                     clean_series,
#                                     order=arima_params,
#                                     seasonal_order=(1, 1, 1, 12),
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False
#                                 )
                                
#                                 fitted = model.fit(disp=False, maxiter=100)
#                                 forecast_values = fitted.forecast(steps=18)
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ SARIMA model fitted successfully")
#                                     print(f"    Forecast sample (first 3): {list(forecast_values[:3])}")
                                
#                                 # Apply forecast values
#                                 for i, forecast_date in enumerate(datos['forecast_dates']):
#                                     if i < len(forecast_values):
#                                         forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
#                                         cells_updated += 1
                                
#                                 processed_products += 1
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ Forecast applied to {len(forecast_values)} months\n")
#                             else:
#                                 if is_debug_target:
#                                     print(f"    ❌ After cleaning, only {len(clean_series)} values - INSUFFICIENT")
                        
#                         else:
#                             # Fallback: Simplified trend-based (if statsmodels not available)
#                             if is_debug_target:
#                                 print(f"    ⚠️ statsmodels NOT available - using trend-based fallback")
                            
#                             # FIXED: Accept zeros
#                             clean_series = historical_series.dropna()
#                             clean_series = clean_series[clean_series >= 0]
                            
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
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ Trend-based forecast applied")
                    
#                     except Exception as e:
#                         if is_debug_target:
#                             print(f"    ❌ ERROR during SARIMA fitting: {str(e)}")
                        
#                         # Fallback to average (FIXED: accept zeros)
#                         avg_value = historical_series[historical_series >= 0].mean()
#                         if pd.notna(avg_value):
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                            
#                             if is_debug_target:
#                                 print(f"    ⚠️ Fallback to average: {avg_value:.2f}")
#                 else:
#                     if is_debug_target:
#                         print(f"    ❌ INSUFFICIENT DATA - Skipping product")
#                         print(f"       (Need at least 24 values, have {non_null_count})\n")
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if extended_series_count > 0:
#                 print(f"ℹ️ Extended {extended_series_count} products with Moving Average data to meet SARIMA requirements")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
#                     'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
#                     'seasonal_order': '(1,1,1,12)',
#                     'reference_product_logic': True,
#                     'skipped_no_calc': skipped_no_calc,
#                     'extended_series_count': extended_series_count,
#                     'optimized': True,
#                     'accepts_zero_values': True,
#                     'extends_with_moving_avg': True,
#                     'minimum_data_points': 24
#                 },
#                 'metadata': {
#                     'n_products_processed': processed_products,
#                     'forecast_months': len(datos['forecast_dates']),
#                     'execution_date': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             print(f"❌ ERROR in ARIMA: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return self._empty_result()
    
#     def _empty_result(self) -> Dict[str, Any]:
#         """Return empty result structure"""
#         return {
#             'dataframe': pd.DataFrame(),
#             'celulas_actualizadas': 0,
#             'parametros': {},
#             'metadata': {'error': True}
#         }

















# """
# ARIMA Forecasting Model - Simple version WITHOUT seasonality
# Recommended for datasets with < 36 months
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, Any, Optional

# try:
#     from statsmodels.tsa.arima.model import ARIMA
#     ARIMA_AVAILABLE = True
# except ImportError:
#     ARIMA_AVAILABLE = False


# class ARIMAModel:
#     """ARIMA forecasting model (non-seasonal)"""
    
#     # DEBUG: Target record
#     DEBUG_TARGET_MODEL = "TC0001973"
#     DEBUG_TARGET_CUSTOMER = "ATACADO ESPECIALIZADO"
    
#     def __init__(self, processor):
#         """
#         Initialize model with reference to main processor
        
#         Args:
#             processor: ForecastProcessor instance
#         """
#         self.processor = processor
    
#     def execute(self, datos: Dict[str, Any], moving_avg_result: Optional[Dict] = None) -> Dict[str, Any]:
#         """
#         Execute ARIMA forecasting (simple, non-seasonal)
        
#         OPTIMIZED FOR: 12-24 months of historical data
        
#         Args:
#             datos: Prepared data dictionary
#             moving_avg_result: Results from Moving Average model (for series extension)
            
#         Returns:
#             Dictionary with results
#         """
#         try:
#             df_main = datos['df_main'].copy()
#             df_result = df_main.copy()
#             col_names = datos['col_names']
            
#             # ARIMA parameters: (p, d, q)
#             # Default: (1, 1, 1) - conservative for short series
#             arima_params = self.processor.parametros.get('arima_params', (1, 1, 1))
            
#             cells_updated = 0
#             processed_products = 0
#             skipped_no_calc = 0
#             extended_series_count = 0
            
#             customer_col = col_names.get('main_customer')
#             model_col = col_names.get('main_product_model')
#             class_col = col_names.get('main_class')
            
#             if not customer_col or not class_col:
#                 print("❌ ERROR: Could not find required columns in Main sheet")
#                 return self._empty_result()
            
#             # Pre-allocate with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Get Moving Average forecast dataframe for series extension
#             moving_avg_df = None
#             if moving_avg_result and 'dataframe' in moving_avg_result:
#                 moving_avg_df = moving_avg_result['dataframe']
            
#             # Process each product
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
#                     print(f"🎯 [ARIMA] TARGET PRODUCT FOUND")
#                     print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
#                     print(f"  Row: {idx}")
#                     print(f"{'='*80}\n")
                
#                 # Check if product has "Não calcula"
#                 if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     if is_debug_target:
#                         print(f"  ⏭️ SKIPPED: Has 'Não calcula'\n")
#                     continue
                
#                 # Get logic (use first forecast month)
#                 calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 if is_debug_target:
#                     print(f"  Calc Base: {calc_base}")
#                     print(f"  P2P: {p2p_model if pd.notna(p2p_model) else 'None'}")
                
#                 # Determine historical series
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
#                     if is_debug_target:
#                         print(f"  Using P2P series: {p2p_model}")
#                 else:
#                     historical_series = self.processor._determine_historical_series(idx, row, datos)
#                     if is_debug_target:
#                         print(f"  Using own/reference series")
                
#                 # Check minimum data requirement (ARIMA needs 12+ months minimum)
#                 non_null_count = len(historical_series.dropna())
                
#                 if is_debug_target:
#                     print(f"\n  DATA CHECK:")
#                     print(f"    Non-null values (historical): {non_null_count}")
#                     print(f"    Required minimum: 12 (ARIMA non-seasonal)")
                
#                 # Extend series with Moving Average if insufficient data
#                 if non_null_count < 12:
#                     if moving_avg_df is not None and is_debug_target:
#                         print(f"    ⚠️ INSUFFICIENT - Attempting to extend with Moving Average forecast...")
                    
#                     if moving_avg_df is not None:
#                         extended_series = historical_series.copy()
#                         extended_count = 0
                        
#                         # Add forecast values from Moving Average to extend the series
#                         for forecast_date in sorted(datos['forecast_dates']):
#                             if forecast_date not in extended_series.index:
#                                 # Get value from Moving Average result
#                                 if forecast_date in moving_avg_df.columns:
#                                     moving_avg_val = moving_avg_df.iloc[idx][forecast_date]
#                                     if pd.notna(moving_avg_val):
#                                         extended_series[forecast_date] = moving_avg_val
#                                         extended_count += 1
                        
#                         # Check if extension was successful
#                         new_non_null_count = len(extended_series.dropna())
                        
#                         if is_debug_target:
#                             print(f"    Extended with {extended_count} Moving Avg values")
#                             print(f"    New total: {new_non_null_count} values")
                        
#                         if new_non_null_count >= 12:
#                             historical_series = extended_series
#                             non_null_count = new_non_null_count
#                             extended_series_count += 1
                            
#                             if is_debug_target:
#                                 print(f"    ✅ EXTENSION SUCCESSFUL - Proceeding with ARIMA")
#                         else:
#                             if is_debug_target:
#                                 print(f"    ❌ Still insufficient ({new_non_null_count} < 12)")
                
#                 # Process if we have enough data
#                 if non_null_count >= 12:
                    
#                     if is_debug_target:
#                         print(f"    ✅ SUFFICIENT DATA - Proceeding with ARIMA forecast")
                    
#                     try:
#                         if ARIMA_AVAILABLE:
#                             # Accept zeros (>= 0)
#                             clean_series = historical_series.dropna()
#                             clean_series = clean_series[clean_series >= 0]
                            
#                             if is_debug_target:
#                                 print(f"    Cleaned series: {len(clean_series)} values >= 0")
                            
#                             if len(clean_series) >= 12:
#                                 if is_debug_target:
#                                     print(f"    Fitting ARIMA model with order={arima_params}...")
                                
#                                 # Fit ARIMA model (NO seasonal component)
#                                 model = ARIMA(
#                                     clean_series,
#                                     order=arima_params,
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False
#                                 )
                                
#                                 fitted = model.fit()
#                                 forecast_values = fitted.forecast(steps=18)
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ ARIMA model fitted successfully")
#                                     print(f"    Forecast sample (first 3): {list(forecast_values[:3])}")
                                
#                                 # Apply forecast values
#                                 for i, forecast_date in enumerate(datos['forecast_dates']):
#                                     if i < len(forecast_values):
#                                         forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
#                                         cells_updated += 1
                                
#                                 processed_products += 1
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ Forecast applied to {len(forecast_values)} months\n")
#                             else:
#                                 if is_debug_target:
#                                     print(f"    ❌ After cleaning, only {len(clean_series)} values - INSUFFICIENT")
                        
#                         else:
#                             # Fallback: Simplified trend-based (if statsmodels not available)
#                             if is_debug_target:
#                                 print(f"    ⚠️ statsmodels NOT available - using trend-based fallback")
                            
#                             clean_series = historical_series.dropna()
#                             clean_series = clean_series[clean_series >= 0]
                            
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
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ Trend-based forecast applied")
                    
#                     except Exception as e:
#                         if is_debug_target:
#                             print(f"    ❌ ERROR during ARIMA fitting: {str(e)}")
                        
#                         # Fallback to average (accept zeros)
#                         avg_value = historical_series[historical_series >= 0].mean()
#                         if pd.notna(avg_value):
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                            
#                             if is_debug_target:
#                                 print(f"    ⚠️ Fallback to average: {avg_value:.2f}")
#                 else:
#                     if is_debug_target:
#                         print(f"    ❌ INSUFFICIENT DATA - Skipping product")
#                         print(f"       (Need at least 12 values, have {non_null_count})\n")
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
#             if extended_series_count > 0:
#                 print(f"ℹ️ Extended {extended_series_count} products with Moving Average data to meet ARIMA requirements")
            
#             return {
#                 'dataframe': df_result,
#                 'celulas_actualizadas': cells_updated,
#                 'parametros': {
#                     'method': 'ARIMA (non-seasonal)',
#                     'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
#                     'seasonal': False,
#                     'reference_product_logic': True,
#                     'skipped_no_calc': skipped_no_calc,
#                     'extended_series_count': extended_series_count,
#                     'optimized': True,
#                     'accepts_zero_values': True,
#                     'extends_with_moving_avg': True,
#                     'minimum_data_points': 12
#                 },
#                 'metadata': {
#                     'n_products_processed': processed_products,
#                     'forecast_months': len(datos['forecast_dates']),
#                     'execution_date': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             print(f"❌ ERROR in ARIMA: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return self._empty_result()
    
#     def _empty_result(self) -> Dict[str, Any]:
#         """Return empty result structure"""
#         return {
#             'dataframe': pd.DataFrame(),
#             'celulas_actualizadas': 0,
#             'parametros': {},
#             'metadata': {'error': True}
#         }









"""
SARIMA Forecasting Model - ULTRA ROBUST version
FIXED: forecast_values indexing issue
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA as SimpleARIMA
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False


class ARIMAModel:
    """SARIMA forecasting model with ultra-robust controls"""
    
    # DEBUG: Target record
    DEBUG_TARGET_MODEL = "TC0001973"
    DEBUG_TARGET_CUSTOMER = "ATACADO ESPECIALIZADO"
    
    # CRITICAL: Maximum reasonable forecast value (safety limit)
    MAX_FORECAST_VALUE = 1e6  # 1 million
    
    def __init__(self, processor):
        """Initialize model with reference to main processor"""
        self.processor = processor
    
    def execute(self, datos: Dict[str, Any], moving_avg_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute SARIMA forecasting with ultra-robust controls
        """
        
        print(f"\n{'='*80}")
        print(f"🔬 STARTING SARIMA MODEL EXECUTION")
        print(f"{'='*80}")
        
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            # CONSERVATIVE SARIMA parameters for stability
            # (0,1,1) x (0,1,1,12) - most stable
            arima_params = self.processor.parametros.get('arima_params', (0, 1, 1))
            seasonal_params = self.processor.parametros.get('seasonal_params', (0, 1, 1, 12))
            
            print(f"📊 SARIMA Parameters: order={arima_params}, seasonal={seasonal_params}")
            
            cells_updated = 0
            processed_products = 0
            skipped_no_calc = 0
            extended_series_count = 0
            fallback_count = 0
            error_count = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                print("❌ ERROR: Could not find required columns in Main sheet")
                return self._empty_result()
            
            # Pre-allocate with NaN
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Get Moving Average forecast dataframe
            moving_avg_df = None
            if moving_avg_result and 'dataframe' in moving_avg_result:
                moving_avg_df = moving_avg_result['dataframe']
                print(f"✅ Moving Average results available for series extension")
            else:
                print(f"⚠️ No Moving Average results - limited extension capability")
            
            # Suppress warnings
            warnings.filterwarnings('ignore')
            
            total_products = len(df_main)
            print(f"📦 Processing {total_products} products...\n")
            
            # Process each product
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_model = row.get(model_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # DEBUG: Check if this is our target record
                is_debug_target = (str(product_model).strip() == self.DEBUG_TARGET_MODEL and 
                                  str(customer).strip() == self.DEBUG_TARGET_CUSTOMER)
                
                if is_debug_target:
                    print(f"\n{'='*80}")
                    print(f"🎯 [SARIMA] TARGET PRODUCT FOUND")
                    print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
                    print(f"  Row: {idx}")
                    print(f"{'='*80}\n")
                
                # Check "Não calcula"
                if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
                    skipped_no_calc += 1
                    continue
                
                # Get logic
                calc_base, p2p_model, _ = self.processor._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                # Determine historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
                else:
                    historical_series = self.processor._determine_historical_series(idx, row, datos)
                
                # Check data availability
                non_null_count = len(historical_series.dropna())
                
                if is_debug_target:
                    print(f"  📊 Historical data: {non_null_count} non-null values")
                
                # Extend if needed
                if non_null_count < 24 and moving_avg_df is not None:
                    extended_series = historical_series.copy()
                    for forecast_date in sorted(datos['forecast_dates']):
                        if forecast_date not in extended_series.index and forecast_date in moving_avg_df.columns:
                            val = moving_avg_df.iloc[idx][forecast_date]
                            if pd.notna(val):
                                extended_series[forecast_date] = val
                    
                    new_count = len(extended_series.dropna())
                    if new_count >= 24:
                        historical_series = extended_series
                        non_null_count = new_count
                        extended_series_count += 1
                        if is_debug_target:
                            print(f"  ✅ Extended to {non_null_count} values")
                
                # Skip if still insufficient
                if non_null_count < 24:
                    if is_debug_target:
                        print(f"  ❌ INSUFFICIENT DATA ({non_null_count} < 24) - SKIPPING\n")
                    continue
                
                # Clean series
                clean_series = historical_series.dropna()
                clean_series = clean_series[clean_series >= 0]
                
                if len(clean_series) < 24:
                    if is_debug_target:
                        print(f"  ❌ After cleaning: {len(clean_series)} < 24 - SKIPPING\n")
                    continue
                
                if is_debug_target:
                    print(f"  📈 Clean series: {len(clean_series)} values")
                    print(f"     Stats: min={clean_series.min():.2f}, max={clean_series.max():.2f}, mean={clean_series.mean():.2f}")
                
                # Try forecasting with multiple methods
                forecast_values = None
                method_used = None
                
                # METHOD 1: Try SARIMA
                try:
                    if SARIMAX_AVAILABLE:
                        if is_debug_target:
                            print(f"  🔬 Attempting SARIMA...")
                        
                        model = SARIMAX(
                            clean_series,
                            order=arima_params,
                            seasonal_order=seasonal_params,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        fitted = model.fit(disp=False, maxiter=100, method='lbfgs')
                        forecast_values = fitted.forecast(steps=18)
                        
                        # CRITICAL FIX: Convert pandas Series to numpy array
                        if isinstance(forecast_values, pd.Series):
                            forecast_values = forecast_values.values
                        
                        method_used = "SARIMA"
                        
                        if is_debug_target:
                            print(f"  ✅ SARIMA successful")
                            print(f"     First 3 values: {list(forecast_values[:3])}")
                
                except Exception as e:
                    if is_debug_target:
                        print(f"  ⚠️ SARIMA failed: {str(e)[:80]}")
                    
                    # METHOD 2: Try simple ARIMA (no seasonality)
                    try:
                        if is_debug_target:
                            print(f"  🔬 Attempting Simple ARIMA...")
                        
                        simple_model = SimpleARIMA(
                            clean_series,
                            order=arima_params,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        simple_fitted = simple_model.fit()
                        forecast_values = simple_fitted.forecast(steps=18)
                        
                        # CRITICAL FIX: Convert to numpy array
                        if isinstance(forecast_values, pd.Series):
                            forecast_values = forecast_values.values
                        
                        method_used = "Simple_ARIMA"
                        fallback_count += 1
                        
                        if is_debug_target:
                            print(f"  ✅ Simple ARIMA successful")
                            print(f"     First 3 values: {list(forecast_values[:3])}")
                    
                    except Exception as e2:
                        if is_debug_target:
                            print(f"  ⚠️ Simple ARIMA failed: {str(e2)[:80]}")
                        
                        # METHOD 3: Trend-based fallback
                        try:
                            if is_debug_target:
                                print(f"  🔬 Attempting Trend-based...")
                            
                            recent = clean_series.iloc[-12:] if len(clean_series) >= 12 else clean_series
                            x = np.arange(len(recent))
                            y = recent.values
                            z = np.polyfit(x, y, 1)
                            trend = z[0]
                            base = y[-1]
                            
                            forecast_values = np.array([max(0, base + trend * (i + 1)) for i in range(18)])
                            method_used = "Trend"
                            fallback_count += 1
                            
                            if is_debug_target:
                                print(f"  ✅ Trend-based successful")
                                print(f"     Trend: {trend:.2f}/month, Base: {base:.2f}")
                                print(f"     First 3 values: {list(forecast_values[:3])}")
                        
                        except Exception as e3:
                            if is_debug_target:
                                print(f"  ❌ All methods failed: {str(e3)[:80]}")
                            error_count += 1
                
                # Apply forecast if we got one
                if forecast_values is not None and len(forecast_values) > 0:
                    
                    # Ensure it's a numpy array (redundant safety check)
                    if not isinstance(forecast_values, np.ndarray):
                        forecast_values = np.array(forecast_values)
                    
                    # Cap extreme values
                    historical_max = clean_series.max()
                    cap = min(historical_max * 10, self.MAX_FORECAST_VALUE)
                    
                    for i, forecast_date in enumerate(datos['forecast_dates']):
                        if i < len(forecast_values):
                            # CRITICAL FIX: Direct array indexing (not pandas)
                            value = float(forecast_values[i])
                            value = min(value, cap)
                            value = max(0, value)
                            forecast_arrays[forecast_date][idx] = value
                            cells_updated += 1
                    
                    processed_products += 1
                    
                    if is_debug_target:
                        applied = [forecast_arrays[d][idx] for d in datos['forecast_dates'][:5]]
                        print(f"  ✅ Forecast applied ({method_used})")
                        print(f"     First 5 applied values: {applied}\n")
                
                # Progress report every 100 products
                if (idx + 1) % 100 == 0:
                    print(f"  Progress: {idx + 1}/{total_products} products processed...")
            
            # Add forecast columns
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            # Final report
            print(f"\n{'='*80}")
            print(f"✅ SARIMA EXECUTION COMPLETED")
            print(f"{'='*80}")
            print(f"📊 Results:")
            print(f"   Total products: {total_products}")
            print(f"   Processed successfully: {processed_products}")
            print(f"   Cells updated: {cells_updated}")
            print(f"   Skipped (Não calcula): {skipped_no_calc}")
            print(f"   Extended with Moving Avg: {extended_series_count}")
            print(f"   Used fallback methods: {fallback_count}")
            print(f"   Errors: {error_count}")
            print(f"{'='*80}\n")
            
            if processed_products == 0:
                print("⚠️⚠️⚠️ WARNING: NO PRODUCTS WERE PROCESSED! ⚠️⚠️⚠️")
                print("Possible reasons:")
                print("  1. All products have 'Não calcula' classification")
                print("  2. Insufficient historical data (need 24+ months)")
                print("  3. All forecasting methods failed")
                print("  4. Data quality issues (all zeros/nulls)")
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'SARIMA (ultra-robust)',
                    'order': f'{arima_params}',
                    'seasonal_order': f'{seasonal_params}',
                    'fallback_methods': 'Simple ARIMA → Trend',
                    'processed_products': processed_products,
                    'error_count': error_count,
                    'fallback_count': fallback_count
                },
                'metadata': {
                    'n_products_processed': processed_products,
                    'forecast_months': len(datos['forecast_dates']),
                    'execution_date': datetime.now()
                }
            }
            
        except Exception as e:
            print(f"\n❌❌❌ CRITICAL ERROR in SARIMA ❌❌❌")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'dataframe': pd.DataFrame(),
            'celulas_actualizadas': 0,
            'parametros': {'error': True},
            'metadata': {'error': True}
        }