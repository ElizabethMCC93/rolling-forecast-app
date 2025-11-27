# """
# SARIMA Forecasting Model
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











"""
SARIMA Forecasting Model
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False


class ARIMAModel:
    """SARIMA forecasting model"""
    
    def __init__(self, processor):
        """
        Initialize model with reference to main processor
        
        Args:
            processor: ForecastProcessor instance
        """
        self.processor = processor
    
    def execute(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SARIMA forecasting
        
        Args:
            datos: Prepared data dictionary
            
        Returns:
            Dictionary with results
        """
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            arima_params = self.processor.parametros.get('arima_params', (1, 1, 1))
            cells_updated = 0
            processed_products = 0
            skipped_no_calc = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                print("❌ ERROR: Could not find required columns in Main sheet")
                return self._empty_result()
            
            # Pre-allocate with NaN
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Process each product
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_model = row.get(model_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Check if product has "Não calcula"
                if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
                    skipped_no_calc += 1
                    continue
                
                # Get logic
                calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                # Determine historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
                else:
                    historical_series = self.processor._determine_historical_series(idx, row, datos)
                
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
                    
                    except Exception:
                        # Fallback to average
                        avg_value = historical_series.mean()
                        if pd.notna(avg_value) and avg_value > 0:
                            for forecast_date in datos['forecast_dates']:
                                forecast_arrays[forecast_date][idx] = max(0, avg_value)
                                cells_updated += 1
                            processed_products += 1
            
            # Add forecast columns
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
                    'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
                    'reference_product_logic': True,
                    'skipped_no_calc': skipped_no_calc,
                    'optimized': True
                },
                'metadata': {
                    'n_products_processed': processed_products,
                    'forecast_months': len(datos['forecast_dates']),
                    'execution_date': datetime.now()
                }
            }
            
        except Exception as e:
            print(f"❌ ERROR in ARIMA: {str(e)}")
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