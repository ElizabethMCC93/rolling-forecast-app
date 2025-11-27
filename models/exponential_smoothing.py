# """
# Exponential Smoothing (Holt-Winters) Forecasting Model
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, Any

# try:
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#     STATSMODELS_AVAILABLE = True
# except ImportError:
#     STATSMODELS_AVAILABLE = False


# class ExponentialSmoothingModel:
#     """Exponential Smoothing (Holt-Winters) forecasting model"""
    
#     def __init__(self, processor):
#         """
#         Initialize model with reference to main processor
        
#         Args:
#             processor: ForecastProcessor instance
#         """
#         self.processor = processor
    
#     def execute(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute Exponential Smoothing forecasting
        
#         Args:
#             datos: Prepared data dictionary
            
#         Returns:
#             Dictionary with results
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
            
#             if not customer_col or not class_col:
#                 print("❌ ERROR: Could not find required columns in Main sheet")
#                 return self._empty_result()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
#             # Process each product row
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
                
#                 # Get logic (use first forecast month)
#                 calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
#                     customer, product_class, datos['forecast_dates'][0]
#                 )
                
#                 # Determine historical series
#                 if pd.notna(p2p_model) and p2p_model != '':
#                     historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
#                 else:
#                     historical_series = self.processor._determine_historical_series(idx, row, datos)
                
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
#                             # Simplified fallback
#                             last_value = historical_series.dropna().iloc[-1]
#                             if pd.notna(last_value) and last_value > 0:
#                                 for forecast_date in datos['forecast_dates']:
#                                     forecast_arrays[forecast_date][idx] = max(0, last_value)
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
#             print(f"❌ ERROR in Exponential Smoothing: {str(e)}")
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
# Exponential Smoothing (Holt-Winters) Forecasting Model
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from typing import Dict, Any

# try:
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#     STATSMODELS_AVAILABLE = True
# except ImportError:
#     STATSMODELS_AVAILABLE = False


# class ExponentialSmoothingModel:
#     """Exponential Smoothing (Holt-Winters) forecasting model"""
    
#     # DEBUG: Target record - UPDATED
#     DEBUG_TARGET_MODEL = "TC0001973"
#     DEBUG_TARGET_CUSTOMER = "ATACADO ESPECIALIZADO"
    
#     def __init__(self, processor):
#         """
#         Initialize model with reference to main processor
        
#         Args:
#             processor: ForecastProcessor instance
#         """
#         self.processor = processor
    
#     def execute(self, datos: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute Exponential Smoothing forecasting
        
#         Args:
#             datos: Prepared data dictionary
            
#         Returns:
#             Dictionary with results
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
            
#             if not customer_col or not class_col:
#                 print("❌ ERROR: Could not find required columns in Main sheet")
#                 return self._empty_result()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
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
#                     print(f"🎯 [EXPONENTIAL] TARGET PRODUCT FOUND")
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
                
#                 # Check minimum data requirement
#                 non_null_count = len(historical_series.dropna())
                
#                 if is_debug_target:
#                     print(f"\n  DATA CHECK:")
#                     print(f"    Non-null values: {non_null_count}")
#                     print(f"    Required minimum: 12")
                
#                 if non_null_count >= 12:
                    
#                     if is_debug_target:
#                         print(f"    ✅ SUFFICIENT DATA - Proceeding with forecast")
                    
#                     try:
#                         if STATSMODELS_AVAILABLE:
#                             clean_series = historical_series.dropna()
#                             clean_series = clean_series[clean_series > 0]
                            
#                             if is_debug_target:
#                                 print(f"    Cleaned series: {len(clean_series)} values > 0")
                            
#                             if len(clean_series) >= 12:
#                                 if is_debug_target:
#                                     print(f"    Fitting Holt-Winters model...")
                                
#                                 model = ExponentialSmoothing(
#                                     clean_series,
#                                     seasonal_periods=12,
#                                     trend='add',
#                                     seasonal='add'
#                                 )
#                                 fitted = model.fit()
#                                 forecast_values = fitted.forecast(steps=18)
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ Model fitted successfully")
#                                     print(f"    Forecast sample (first 3): {list(forecast_values[:3])}")
                                
#                                 for i, forecast_date in enumerate(datos['forecast_dates']):
#                                     if i < len(forecast_values):
#                                         forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
#                                         cells_updated += 1
                                
#                                 processed_products += 1
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ Forecast applied to {len(forecast_values)} months")
#                             else:
#                                 if is_debug_target:
#                                     print(f"    ❌ After cleaning, only {len(clean_series)} values - INSUFFICIENT")
#                         else:
#                             if is_debug_target:
#                                 print(f"    ⚠️ statsmodels NOT available - using simplified method")
                            
#                             # Simplified fallback
#                             last_value = historical_series.dropna().iloc[-1]
#                             if pd.notna(last_value) and last_value > 0:
#                                 for forecast_date in datos['forecast_dates']:
#                                     forecast_arrays[forecast_date][idx] = max(0, last_value)
#                                     cells_updated += 1
#                                 processed_products += 1
                                
#                                 if is_debug_target:
#                                     print(f"    ✅ Simplified forecast applied: {last_value}")
                    
#                     except Exception as e:
#                         if is_debug_target:
#                             print(f"    ❌ ERROR during model fitting: {str(e)}")
                        
#                         # Fallback to average
#                         avg_value = historical_series.mean()
#                         if pd.notna(avg_value) and avg_value > 0:
#                             for forecast_date in datos['forecast_dates']:
#                                 forecast_arrays[forecast_date][idx] = max(0, avg_value)
#                                 cells_updated += 1
#                             processed_products += 1
                            
#                             if is_debug_target:
#                                 print(f"    ⚠️ Fallback to average: {avg_value}")
#                 else:
#                     if is_debug_target:
#                         print(f"    ❌ INSUFFICIENT DATA - Skipping product")
#                         print(f"       (Need at least 12 non-null values, have {non_null_count})\n")
            
#             # Add forecast columns
#             for forecast_date, values in forecast_arrays.items():
#                 df_result[forecast_date] = values
            
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
#             print(f"❌ ERROR in Exponential Smoothing: {str(e)}")
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
Exponential Smoothing (Holt-Winters) - FIXED for insufficient data
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ExponentialSmoothingModel:
    """Exponential Smoothing (Holt-Winters) forecasting model"""
    
    # DEBUG: Target record - UPDATED
    DEBUG_TARGET_MODEL = "TC0001973"
    DEBUG_TARGET_CUSTOMER = "ATACADO ESPECIALIZADO"
    
    def __init__(self, processor):
        """
        Initialize model with reference to main processor
        
        Args:
            processor: ForecastProcessor instance
        """
        self.processor = processor
    
    def execute(self, datos: Dict[str, Any], moving_avg_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute Exponential Smoothing forecasting
        
        Args:
            datos: Prepared data dictionary
            moving_avg_result: Results from Moving Average model (for series extension)
            
        Returns:
            Dictionary with results
        """
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            cells_updated = 0
            processed_products = 0
            skipped_no_calc = 0
            extended_series_count = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                print("❌ ERROR: Could not find required columns in Main sheet")
                return self._empty_result()
            
            # Pre-allocate forecast columns with NaN
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Get Moving Average forecast dataframe for series extension
            moving_avg_df = None
            if moving_avg_result and 'dataframe' in moving_avg_result:
                moving_avg_df = moving_avg_result['dataframe']
            
            # Process each product row
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
                    print(f"🎯 [EXPONENTIAL] TARGET PRODUCT FOUND")
                    print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
                    print(f"  Row: {idx}")
                    print(f"{'='*80}\n")
                
                # Check if product has "Não calcula"
                if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
                    skipped_no_calc += 1
                    if is_debug_target:
                        print(f"  ⏭️ SKIPPED: Has 'Não calcula'\n")
                    continue
                
                # Get logic (use first forecast month)
                calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                if is_debug_target:
                    print(f"  Calc Base: {calc_base}")
                    print(f"  P2P: {p2p_model if pd.notna(p2p_model) else 'None'}")
                
                # Determine historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
                    if is_debug_target:
                        print(f"  Using P2P series: {p2p_model}")
                else:
                    historical_series = self.processor._determine_historical_series(idx, row, datos)
                    if is_debug_target:
                        print(f"  Using own/reference series")
                
                # Check minimum data requirement
                non_null_count = len(historical_series.dropna())
                
                if is_debug_target:
                    print(f"\n  DATA CHECK:")
                    print(f"    Non-null values (historical): {non_null_count}")
                    print(f"    Required minimum: 24")
                
                # CRITICAL FIX: Extend series with Moving Average forecast if insufficient data
                if non_null_count < 24:
                    if moving_avg_df is not None and is_debug_target:
                        print(f"    ⚠️ INSUFFICIENT - Attempting to extend with Moving Average forecast...")
                    
                    if moving_avg_df is not None:
                        extended_series = historical_series.copy()
                        extended_count = 0
                        
                        # Add forecast values from Moving Average to extend the series
                        for forecast_date in sorted(datos['forecast_dates']):
                            if forecast_date not in extended_series.index:
                                # Get value from Moving Average result
                                if forecast_date in moving_avg_df.columns:
                                    moving_avg_val = moving_avg_df.iloc[idx][forecast_date]
                                    if pd.notna(moving_avg_val):
                                        extended_series[forecast_date] = moving_avg_val
                                        extended_count += 1
                        
                        # Check if extension was successful
                        new_non_null_count = len(extended_series.dropna())
                        
                        if is_debug_target:
                            print(f"    Extended with {extended_count} Moving Avg values")
                            print(f"    New total: {new_non_null_count} values")
                        
                        if new_non_null_count >= 24:
                            historical_series = extended_series
                            non_null_count = new_non_null_count
                            extended_series_count += 1
                            
                            if is_debug_target:
                                print(f"    ✅ EXTENSION SUCCESSFUL - Proceeding with Holt-Winters")
                        else:
                            if is_debug_target:
                                print(f"    ❌ Still insufficient ({new_non_null_count} < 24)")
                
                if non_null_count >= 24:
                    
                    if is_debug_target:
                        print(f"    ✅ SUFFICIENT DATA - Proceeding with forecast")
                    
                    try:
                        if STATSMODELS_AVAILABLE:
                            # FIXED: Accept zeros (val >= 0)
                            clean_series = historical_series.dropna()
                            clean_series = clean_series[clean_series >= 0]
                            
                            if is_debug_target:
                                print(f"    Cleaned series: {len(clean_series)} values >= 0")
                            
                            if len(clean_series) >= 24:
                                if is_debug_target:
                                    print(f"    Fitting Holt-Winters model...")
                                
                                model = ExponentialSmoothing(
                                    clean_series,
                                    seasonal_periods=12,
                                    trend='add',
                                    seasonal='add'
                                )
                                fitted = model.fit()
                                forecast_values = fitted.forecast(steps=18)
                                
                                if is_debug_target:
                                    print(f"    ✅ Model fitted successfully")
                                    print(f"    Forecast sample (first 3): {list(forecast_values[:3])}")
                                
                                for i, forecast_date in enumerate(datos['forecast_dates']):
                                    if i < len(forecast_values):
                                        forecast_arrays[forecast_date][idx] = max(0, forecast_values[i])
                                        cells_updated += 1
                                
                                processed_products += 1
                                
                                if is_debug_target:
                                    print(f"    ✅ Forecast applied to {len(forecast_values)} months\n")
                            else:
                                if is_debug_target:
                                    print(f"    ❌ After cleaning, only {len(clean_series)} values - INSUFFICIENT")
                        else:
                            if is_debug_target:
                                print(f"    ⚠️ statsmodels NOT available - using simplified method")
                            
                            # Simplified fallback (accepts zeros)
                            last_value = historical_series.dropna().iloc[-1]
                            if pd.notna(last_value) and last_value >= 0:
                                for forecast_date in datos['forecast_dates']:
                                    forecast_arrays[forecast_date][idx] = max(0, last_value)
                                    cells_updated += 1
                                processed_products += 1
                                
                                if is_debug_target:
                                    print(f"    ✅ Simplified forecast applied: {last_value}")
                    
                    except Exception as e:
                        if is_debug_target:
                            print(f"    ❌ ERROR during model fitting: {str(e)}")
                        
                        # Fallback to average (accepts zeros)
                        avg_value = historical_series[historical_series >= 0].mean()
                        if pd.notna(avg_value):
                            for forecast_date in datos['forecast_dates']:
                                forecast_arrays[forecast_date][idx] = max(0, avg_value)
                                cells_updated += 1
                            processed_products += 1
                            
                            if is_debug_target:
                                print(f"    ⚠️ Fallback to average: {avg_value}")
                else:
                    if is_debug_target:
                        print(f"    ❌ INSUFFICIENT DATA - Skipping product")
                        print(f"       (Need at least 24 non-null values, have {non_null_count})\n")
            
            # Add forecast columns
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            if extended_series_count > 0:
                print(f"ℹ️ Extended {extended_series_count} products with Moving Average data to meet Holt-Winters requirements")
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
                    'seasonal_periods': 12,
                    'reference_product_logic': True,
                    'skipped_no_calc': skipped_no_calc,
                    'extended_series_count': extended_series_count,
                    'optimized': True,
                    'accepts_zero_values': True,
                    'extends_with_moving_avg': True
                },
                'metadata': {
                    'n_products_processed': processed_products,
                    'forecast_months': len(datos['forecast_dates']),
                    'execution_date': datetime.now()
                }
            }
            
        except Exception as e:
            print(f"❌ ERROR in Exponential Smoothing: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'dataframe': pd.DataFrame(),
            'celulas_actualizadas': 0,
            'parametros': {},
            'metadata': {'error': True}
        }