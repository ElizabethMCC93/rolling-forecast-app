# """
# Moving Average (Seasonal) Forecasting Model - FIXED for zeros and future months
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# from typing import Dict, Any, List


# class MovingAverageModel:
#     """Seasonal Moving Average forecasting model"""
    
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
#         Execute Seasonal Moving Average forecasting
        
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
#             start_date_col = col_names.get('main_start_date')
            
#             if not customer_col or not class_col:
#                 print("❌ ERROR: Could not find required columns (Customer, Class) in Main sheet")
#                 return self._empty_result()
            
#             # Pre-allocate forecast columns with NaN
#             forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
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
#                     print(f"🎯 [MOVING AVG] TARGET PRODUCT FOUND")
#                     print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
#                     print(f"  Row: {idx}")
#                     print(f"{'='*80}\n")
                
#                 # Check if product has "Não calcula" in ANY month
#                 if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
#                     skipped_no_calc += 1
#                     continue
                
#                 # Get start date from column H
#                 start_date = None
#                 if start_date_col:
#                     start_date_value = row.get(start_date_col)
#                     start_date = self.processor._parse_start_date(start_date_value)
#                     if is_debug_target:
#                         print(f"  Start Date (Col H): {start_date.strftime('%Y-%m-%d') if start_date else 'None'}\n")
                
#                 # Determine historical series
#                 historical_series = self.processor._determine_historical_series(idx, row, datos)
                
#                 # Process each forecast month
#                 for i, forecast_date in enumerate(datos['forecast_dates']):
                    
#                     if is_debug_target and i < 5:  # Show first 5 months for brevity
#                         print(f"  --- Month {i+1}/{len(datos['forecast_dates'])}: {forecast_date.strftime('%Y-%m')} ---")
                    
#                     # Check start date restriction
#                     if not self.processor._should_forecast_in_month(start_date, forecast_date):
#                         if is_debug_target and i < 5:
#                             print(f"    SKIP: Before start date\n")
#                         continue
                    
#                     # Get logic from cache
#                     calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
#                         customer, product_class, forecast_date
#                     )
                    
#                     if is_debug_target and i < 5:
#                         print(f"    Calc Base: {calc_base}")
#                         print(f"    P2P: {p2p_model if pd.notna(p2p_model) else 'None'}")
                    
#                     # Handle launch dependent
#                     if calc_base == self.processor.CALC_BASE_LAUNCH_DEPENDENT:
#                         if pd.notna(launch_month):
#                             launch_date = pd.to_datetime(launch_month)
#                             if forecast_date < launch_date:
#                                 if is_debug_target and i < 5:
#                                     print(f"    SKIP: Before launch\n")
#                                 continue
                    
#                     # Handle P2P
#                     if calc_base in [self.processor.CALC_BASE_P2P, self.processor.CALC_BASE_LAUNCH_DEPENDENT]:
                        
#                         # Get series (P2P or own/reference)
#                         if pd.notna(p2p_model) and p2p_model != '':
#                             series_to_use = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
#                             if is_debug_target and i < 5:
#                                 print(f"    Using P2P: {p2p_model}")
#                         else:
#                             series_to_use = historical_series
#                             if is_debug_target and i < 5:
#                                 print(f"    Using own/ref series")
                        
#                         if len(series_to_use) > 0 and not series_to_use.isna().all():
#                             # CRITICAL FIX: Pass forecast_arrays and idx for future month lookup
#                             seasonal_values = self._get_seasonal_window(
#                                 forecast_date, 
#                                 series_to_use, 
#                                 forecast_arrays, 
#                                 idx,
#                                 is_debug_target and i < 5
#                             )
                            
#                             if len(seasonal_values) > 0:
#                                 seasonal_avg = np.mean(seasonal_values)
#                                 year = forecast_date.year
#                                 growth_factor = self.processor._get_growth_factor_cached(customer, year)
                                
#                                 forecasted_value = seasonal_avg * growth_factor
#                                 forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
#                                 cells_updated += 1
                                
#                                 if is_debug_target and i < 5:
#                                     print(f"    Result: {seasonal_avg:.2f} × {growth_factor} = {forecasted_value:.2f}\n")
#                             else:
#                                 # FIXED: Fallback accepts zeros (val >= 0 instead of val > 0)
#                                 simple_avg = series_to_use[series_to_use >= 0].mean()
#                                 if pd.notna(simple_avg):
#                                     year = forecast_date.year
#                                     growth_factor = self.processor._get_growth_factor_cached(customer, year)
#                                     forecasted_value = simple_avg * growth_factor
#                                     forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
#                                     cells_updated += 1
                                    
#                                     if is_debug_target and i < 5:
#                                         print(f"    FALLBACK: {simple_avg:.2f} × {growth_factor} = {forecasted_value:.2f}\n")
                
#                 processed_products += 1
            
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
#                     'skipped_no_calc': skipped_no_calc,
#                     'optimized': True,
#                     'accepts_zero_values': True,
#                     'uses_forecast_for_future_months': True
#                 },
#                 'metadata': {
#                     'n_products_processed': processed_products,
#                     'forecast_months': len(datos['forecast_dates']),
#                     'execution_date': datetime.now()
#                 }
#             }
            
#         except Exception as e:
#             print(f"❌ ERROR in Moving Average: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return self._empty_result()
    
#     def _get_seasonal_window(self, forecast_date: datetime, historical_series: pd.Series, 
#                             forecast_arrays: Dict, idx: int, debug: bool = False) -> List[float]:
#         """
#         Get seasonal window: same month last year + 2 following months
        
#         CRITICAL FIX 1: If data not in historical, look in current forecast (already calculated)
#         CRITICAL FIX 2: Accept zeros (val >= 0 instead of val > 0)
        
#         Args:
#             forecast_date: Month to forecast
#             historical_series: Historical data series
#             forecast_arrays: Dictionary with already calculated forecasts
#             idx: Row index in main dataframe
#             debug: Enable debug logging
            
#         Returns:
#             List of values for seasonal window
#         """
        
#         same_month_last_year = forecast_date - relativedelta(years=1)
#         month_after_1 = same_month_last_year + relativedelta(months=1)
#         month_after_2 = same_month_last_year + relativedelta(months=2)
        
#         values = []
        
#         if debug:
#             print(f"    Seasonal lookup:")
        
#         for target_date in [same_month_last_year, month_after_1, month_after_2]:
#             val = None
#             source = None
            
#             # PRIORITY 1: Try historical first
#             if target_date in historical_series.index:
#                 val = historical_series[target_date]
#                 source = "historical"
            
#             # PRIORITY 2: If not in historical, try current forecast (already calculated)
#             elif target_date in forecast_arrays:
#                 forecast_val = forecast_arrays[target_date][idx]
#                 if not np.isnan(forecast_val):
#                     val = forecast_val
#                     source = "forecast"
            
#             # CRITICAL FIX: Changed from val > 0 to val >= 0 (accept zeros)
#             if pd.notna(val) and val >= 0:
#                 values.append(float(val))
#                 if debug:
#                     print(f"      {target_date.strftime('%Y-%m')}: ✅ {val:.2f} (from {source})")
#             else:
#                 if debug:
#                     status = "N/A" if val is None else f"{val}"
#                     print(f"      {target_date.strftime('%Y-%m')}: ❌ {status}")
        
#         if debug:
#             print(f"    Found: {values}")
        
#         return values
    
#     def _empty_result(self) -> Dict[str, Any]:
#         """Return empty result structure"""
#         return {
#             'dataframe': pd.DataFrame(),
#             'celulas_actualizadas': 0,
#             'parametros': {},
#             'metadata': {'error': True}
#         }














"""
Moving Average (Seasonal) Forecasting Model - FIXED for zeros and future months
UPDATED: Removed premature "Não calcula" check - now handles month-by-month
"""
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List


class MovingAverageModel:
    """Seasonal Moving Average forecasting model"""
    
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
    
    def execute(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Seasonal Moving Average forecasting
        
        Args:
            datos: Prepared data dictionary
            
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
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            start_date_col = col_names.get('main_start_date')
            
            if not customer_col or not class_col:
                print("❌ ERROR: Could not find required columns (Customer, Class) in Main sheet")
                return self._empty_result()
            
            # Pre-allocate forecast columns with NaN
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            total_products = len(df_main)
            
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
                    print(f"🎯 [MOVING AVG] TARGET PRODUCT FOUND")
                    print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
                    print(f"  Row: {idx}")
                    print(f"{'='*80}\n")
                
                # REMOVED: Premature "Não calcula" check
                # NOTE: We now handle "Não calcula" MONTH-BY-MONTH in the forecast loop below (line ~130)
                # Skipping entire products here was causing too many to be ignored
                # The logic already checks calc_base for EACH month individually
                #
                # OLD CODE (COMMENTED OUT):
                # if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
                #     skipped_no_calc += 1
                #     continue
                
                # Get start date from column H
                start_date = None
                if start_date_col:
                    start_date_value = row.get(start_date_col)
                    start_date = self.processor._parse_start_date(start_date_value)
                    if is_debug_target:
                        print(f"  Start Date (Col H): {start_date.strftime('%Y-%m-%d') if start_date else 'None'}\n")
                
                # Determine historical series
                historical_series = self.processor._determine_historical_series(idx, row, datos)
                
                # Process each forecast month
                for i, forecast_date in enumerate(datos['forecast_dates']):
                    
                    if is_debug_target and i < 5:  # Show first 5 months for brevity
                        print(f"  --- Month {i+1}/{len(datos['forecast_dates'])}: {forecast_date.strftime('%Y-%m')} ---")
                    
                    # Check start date restriction
                    if not self.processor._should_forecast_in_month(start_date, forecast_date):
                        if is_debug_target and i < 5:
                            print(f"    SKIP: Before start date\n")
                        continue
                    
                    # Get logic from cache
                    calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
                        customer, product_class, forecast_date
                    )
                    
                    if is_debug_target and i < 5:
                        print(f"    Calc Base: {calc_base}")
                        print(f"    P2P: {p2p_model if pd.notna(p2p_model) else 'None'}")
                    
                    # MONTH-BY-MONTH LOGIC: Handle "Não calcula" for this specific month
                    if calc_base == self.processor.CALC_BASE_NO_CALC:
                        if is_debug_target and i < 5:
                            print(f"    SKIP: Não calcula\n")
                        continue
                    
                    # Handle launch dependent
                    if calc_base == self.processor.CALC_BASE_LAUNCH_DEPENDENT:
                        if pd.notna(launch_month):
                            launch_date = pd.to_datetime(launch_month)
                            if forecast_date < launch_date:
                                if is_debug_target and i < 5:
                                    print(f"    SKIP: Before launch\n")
                                continue
                    
                    # Handle P2P
                    if calc_base in [self.processor.CALC_BASE_P2P, self.processor.CALC_BASE_LAUNCH_DEPENDENT]:
                        
                        # Get series (P2P or own/reference)
                        if pd.notna(p2p_model) and p2p_model != '':
                            series_to_use = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
                            if is_debug_target and i < 5:
                                print(f"    Using P2P: {p2p_model}")
                        else:
                            series_to_use = historical_series
                            if is_debug_target and i < 5:
                                print(f"    Using own/ref series")
                        
                        if len(series_to_use) > 0 and not series_to_use.isna().all():
                            # CRITICAL FIX: Pass forecast_arrays and idx for future month lookup
                            seasonal_values = self._get_seasonal_window(
                                forecast_date, 
                                series_to_use, 
                                forecast_arrays, 
                                idx,
                                is_debug_target and i < 5
                            )
                            
                            if len(seasonal_values) > 0:
                                seasonal_avg = np.mean(seasonal_values)
                                year = forecast_date.year
                                growth_factor = self.processor._get_growth_factor_cached(customer, year)
                                
                                forecasted_value = seasonal_avg * growth_factor
                                forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
                                cells_updated += 1
                                
                                if is_debug_target and i < 5:
                                    print(f"    Result: {seasonal_avg:.2f} × {growth_factor} = {forecasted_value:.2f}\n")
                            else:
                                # FIXED: Fallback accepts zeros (val >= 0 instead of val > 0)
                                simple_avg = series_to_use[series_to_use >= 0].mean()
                                if pd.notna(simple_avg):
                                    year = forecast_date.year
                                    growth_factor = self.processor._get_growth_factor_cached(customer, year)
                                    forecasted_value = simple_avg * growth_factor
                                    forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
                                    cells_updated += 1
                                    
                                    if is_debug_target and i < 5:
                                        print(f"    FALLBACK: {simple_avg:.2f} × {growth_factor} = {forecasted_value:.2f}\n")
                
                processed_products += 1
            
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
                    'reference_product_logic': True,
                    'start_date_restriction': True,
                    'skipped_no_calc': skipped_no_calc,
                    'optimized': True,
                    'accepts_zero_values': True,
                    'uses_forecast_for_future_months': True
                },
                'metadata': {
                    'n_products_processed': processed_products,
                    'forecast_months': len(datos['forecast_dates']),
                    'execution_date': datetime.now()
                }
            }
            
        except Exception as e:
            print(f"❌ ERROR in Moving Average: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_result()
    
    def _get_seasonal_window(self, forecast_date: datetime, historical_series: pd.Series, 
                            forecast_arrays: Dict, idx: int, debug: bool = False) -> List[float]:
        """
        Get seasonal window: same month last year + 2 following months
        
        CRITICAL FIX 1: If data not in historical, look in current forecast (already calculated)
        CRITICAL FIX 2: Accept zeros (val >= 0 instead of val > 0)
        
        Args:
            forecast_date: Month to forecast
            historical_series: Historical data series
            forecast_arrays: Dictionary with already calculated forecasts
            idx: Row index in main dataframe
            debug: Enable debug logging
            
        Returns:
            List of values for seasonal window
        """
        
        same_month_last_year = forecast_date - relativedelta(years=1)
        month_after_1 = same_month_last_year + relativedelta(months=1)
        month_after_2 = same_month_last_year + relativedelta(months=2)
        
        values = []
        
        if debug:
            print(f"    Seasonal lookup:")
        
        for target_date in [same_month_last_year, month_after_1, month_after_2]:
            val = None
            source = None
            
            # PRIORITY 1: Try historical first
            if target_date in historical_series.index:
                val = historical_series[target_date]
                source = "historical"
            
            # PRIORITY 2: If not in historical, try current forecast (already calculated)
            elif target_date in forecast_arrays:
                forecast_val = forecast_arrays[target_date][idx]
                if not np.isnan(forecast_val):
                    val = forecast_val
                    source = "forecast"
            
            # CRITICAL FIX: Changed from val > 0 to val >= 0 (accept zeros)
            if pd.notna(val) and val >= 0:
                values.append(float(val))
                if debug:
                    print(f"      {target_date.strftime('%Y-%m')}: ✅ {val:.2f} (from {source})")
            else:
                if debug:
                    status = "N/A" if val is None else f"{val}"
                    print(f"      {target_date.strftime('%Y-%m')}: ❌ {status}")
        
        if debug:
            print(f"    Found: {values}")
        
        return values
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'dataframe': pd.DataFrame(),
            'celulas_actualizadas': 0,
            'parametros': {},
            'metadata': {'error': True}
        }