"""
Exponential Smoothing (Holt-Winters) Forecasting Model
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ExponentialSmoothingModel:
    """Exponential Smoothing (Holt-Winters) forecasting model"""
    
    def __init__(self, processor):
        """
        Initialize model with reference to main processor
        
        Args:
            processor: ForecastProcessor instance
        """
        self.processor = processor
    
    def execute(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Exponential Smoothing forecasting
        
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
            
            if not customer_col or not class_col:
                print("❌ ERROR: Could not find required columns in Main sheet")
                return self._empty_result()
            
            # Pre-allocate forecast columns with NaN
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Process each product row
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
                
                # Get logic (use first forecast month)
                calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                # Determine historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
                else:
                    historical_series = self.processor._determine_historical_series(idx, row, datos)
                
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
                            # Simplified fallback
                            last_value = historical_series.dropna().iloc[-1]
                            if pd.notna(last_value) and last_value > 0:
                                for forecast_date in datos['forecast_dates']:
                                    forecast_arrays[forecast_date][idx] = max(0, last_value)
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
                    'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
                    'seasonal_periods': 12,
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