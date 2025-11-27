"""
Moving Average (Seasonal) Forecasting Model
"""
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List


class MovingAverageModel:
    """Seasonal Moving Average forecasting model"""
    
    # DEBUG: Target record for detailed logging
    DEBUG_TARGET_MODEL = "4145766"
    DEBUG_TARGET_CUSTOMER = "ALPASHOP"
    
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
                    print(f"🎯 [MOVING AVG] TARGET PRODUCT")
                    print(f"  Model: {product_model}, Customer: {customer}, Class: {product_class}")
                    print(f"  Row: {idx}")
                    print(f"{'='*80}\n")
                
                # Check if product has "Não calcula" in ANY month
                if self.processor._has_no_calc_in_any_month(customer, product_model, product_class, datos['forecast_dates']):
                    skipped_no_calc += 1
                    continue
                
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
                    
                    if is_debug_target:
                        print(f"  --- Month {i+1}/{len(datos['forecast_dates'])}: {forecast_date.strftime('%Y-%m')} ---")
                    
                    # Check start date restriction
                    if not self.processor._should_forecast_in_month(start_date, forecast_date):
                        if is_debug_target:
                            print(f"    SKIP: Before start date\n")
                        continue
                    
                    # Get logic from cache
                    calc_base, p2p_model, launch_month = self.processor._get_logic_for_product_cached(
                        customer, product_class, forecast_date
                    )
                    
                    if is_debug_target:
                        print(f"    Calc Base: {calc_base}")
                        print(f"    P2P: {p2p_model if pd.notna(p2p_model) else 'None'}")
                    
                    # Handle launch dependent
                    if calc_base == self.processor.CALC_BASE_LAUNCH_DEPENDENT:
                        if pd.notna(launch_month):
                            launch_date = pd.to_datetime(launch_month)
                            if forecast_date < launch_date:
                                if is_debug_target:
                                    print(f"    SKIP: Before launch\n")
                                continue
                    
                    # Handle P2P
                    if calc_base in [self.processor.CALC_BASE_P2P, self.processor.CALC_BASE_LAUNCH_DEPENDENT]:
                        
                        # Get series (P2P or own/reference)
                        if pd.notna(p2p_model) and p2p_model != '':
                            series_to_use = self.processor._get_p2p_series_cached(customer, p2p_model, datos)
                            if is_debug_target:
                                print(f"    Using P2P: {p2p_model}")
                        else:
                            series_to_use = historical_series
                            if is_debug_target:
                                print(f"    Using own/ref series")
                        
                        if len(series_to_use) > 0 and not series_to_use.isna().all():
                            # Get seasonal window
                            seasonal_values = self._get_seasonal_window(forecast_date, series_to_use)
                            
                            # DEBUG: Show seasonal lookup
                            if is_debug_target:
                                lookup_dates = [
                                    forecast_date - relativedelta(years=1),
                                    forecast_date - relativedelta(years=1) + relativedelta(months=1),
                                    forecast_date - relativedelta(years=1) + relativedelta(months=2)
                                ]
                                print(f"    Seasonal lookup:")
                                for ld in lookup_dates:
                                    val = series_to_use.get(ld, np.nan)
                                    exists = "✅" if ld in series_to_use.index and pd.notna(val) and val > 0 else "❌"
                                    print(f"      {ld.strftime('%Y-%m')}: {exists} {val if pd.notna(val) else 'N/A'}")
                                print(f"    Found: {seasonal_values}")
                            
                            if len(seasonal_values) > 0:
                                seasonal_avg = np.mean(seasonal_values)
                                year = forecast_date.year
                                growth_factor = self.processor._get_growth_factor_cached(customer, year)
                                
                                forecasted_value = seasonal_avg * growth_factor
                                forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
                                cells_updated += 1
                                
                                if is_debug_target:
                                    print(f"    Result: {seasonal_avg:.2f} × {growth_factor} = {forecasted_value:.2f}\n")
                            else:
                                # Fallback to simple average
                                simple_avg = series_to_use.mean()
                                if pd.notna(simple_avg) and simple_avg > 0:
                                    year = forecast_date.year
                                    growth_factor = self.processor._get_growth_factor_cached(customer, year)
                                    forecasted_value = simple_avg * growth_factor
                                    forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
                                    cells_updated += 1
                                    
                                    if is_debug_target:
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
                    'optimized': True
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
    
    def _get_seasonal_window(self, forecast_date: datetime, historical_series: pd.Series) -> List[float]:
        """Get seasonal window: same month last year + 2 following months"""
        
        same_month_last_year = forecast_date - relativedelta(years=1)
        month_after_1 = same_month_last_year + relativedelta(months=1)
        month_after_2 = same_month_last_year + relativedelta(months=2)
        
        values = []
        
        for target_date in [same_month_last_year, month_after_1, month_after_2]:
            if target_date in historical_series.index:
                val = historical_series[target_date]
                if pd.notna(val) and val > 0:
                    values.append(float(val))
        
        return values
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'dataframe': pd.DataFrame(),
            'celulas_actualizadas': 0,
            'parametros': {},
            'metadata': {'error': True}
        }