"""
Forecasting processing engine - Optimized version
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Tuple
import time

# Optional imports for advanced models
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False


class ForecastProcessor:
    """Main class for forecasting processing - Optimized"""
    
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
        
        # Cache for optimization
        self._logic_cache = {}
        self._growth_factor_cache = {}
        self._p2p_series_cache = {}
    
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
        """Detect actual column names in DataFrames and create mapping"""
        
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
        
        st.info(f"🔍 Detected column mappings: {len(self.col_names)} columns identified")
        
        with st.expander("🔍 Column Detection Details"):
            st.write("**Main sheet columns:**")
            st.write(list(self.df_main.columns[:15]))
            st.write("**Logics sheet columns:**")
            st.write(list(self.df_logics.columns))
            st.write("**Relations sheet columns:**")
            st.write(list(self.df_relations.columns[:10]))
            st.write("**Detected mappings:**")
            st.json(self.col_names)
    
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
                resultados['media_movil'] = self._ejecutar_media_movil(datos_preparados)
                resultados['media_movil']['metadata']['execution_time'] = time.time() - start_time
        
        # Execute Exponential Smoothing
        if modelos_ejecutar.get('suavizacao_exponencial', False):
            with st.spinner("📊 Executing Exponential Smoothing..."):
                start_time = time.time()
                resultados['suavizacao_exponencial'] = self._ejecutar_suavizacao(datos_preparados)
                resultados['suavizacao_exponencial']['metadata']['execution_time'] = time.time() - start_time
        
        # Execute ARIMA
        if modelos_ejecutar.get('arima', False):
            with st.spinner("🔬 Executing SARIMA..."):
                start_time = time.time()
                resultados['arima'] = self._ejecutar_arima(datos_preparados)
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
        """Pre-build caches for faster execution - OPTIMIZATION"""
        
        df_logics = datos['df_logics']
        df_relations = datos['df_relations']
        col_names = datos['col_names']
        
        # Cache 1: Logic lookup (class + month -> logic)
        class_col = col_names.get('logics_class')
        month_col = col_names.get('logics_month')
        calc_base_col = col_names.get('logics_calc_base')
        p2p_col = col_names.get('logics_p2p')
        launch_col = col_names.get('logics_launch_month')
        
        if class_col and month_col and class_col in df_logics.columns and month_col in df_logics.columns:
            for idx, row in df_logics.iterrows():
                product_class = row.get(class_col)
                month_date = pd.to_datetime(row.get(month_col)).replace(day=1)
                
                cache_key = (product_class, month_date)
                
                self._logic_cache[cache_key] = (
                    row.get(calc_base_col, self.CALC_BASE_P2P) if calc_base_col else self.CALC_BASE_P2P,
                    row.get(p2p_col, None) if p2p_col else None,
                    row.get(launch_col, None) if launch_col else None
                )
        
        # Cache 2: Growth factors (customer + year -> factor)
        customer_col = col_names.get('relations_customer')
        
        if customer_col and customer_col in df_relations.columns:
            for idx, row in df_relations.iterrows():
                customer = row.get(customer_col)
                
                # Get all year columns
                for col in df_relations.columns:
                    # Check if column is a year (can be string or int)
                    is_year_column = False
                    
                    if isinstance(col, str) and col.isdigit():
                        # String year like '2024', '2025'
                        is_year_column = True
                        year = int(col)
                    elif isinstance(col, int) and 2000 <= col <= 2100:
                        # Direct integer year
                        is_year_column = True
                        year = col
                    
                    if is_year_column:
                        factor = row.get(col)
                        
                        if pd.notna(factor) and isinstance(factor, (int, float)):
                            cache_key = (customer, year)
                            self._growth_factor_cache[cache_key] = float(factor)
        
        # Cache 3: P2P series (customer + p2p_model -> series) - built on demand
        # This is done lazily to avoid loading all series upfront
    
    def _get_logic_for_product_cached(self, customer: str, product_class: str, 
                                     month_date: datetime) -> Tuple[str, str, Any]:
        """Get logic from cache - OPTIMIZED"""
        
        month_start = pd.to_datetime(month_date).replace(day=1)
        cache_key = (product_class, month_start)
        
        if cache_key in self._logic_cache:
            return self._logic_cache[cache_key]
        
        # Default if not found
        return self.CALC_BASE_P2P, None, None
    
    def _get_growth_factor_cached(self, customer: str, year: int) -> float:
        """Get growth factor from cache - OPTIMIZED"""
        
        cache_key = (customer, year)
        
        if cache_key in self._growth_factor_cache:
            return self._growth_factor_cache[cache_key]
        
        return 1.0
    
    def _get_p2p_series_cached(self, customer: str, p2p_model: str, datos: Dict) -> pd.Series:
        """Get P2P series from cache - OPTIMIZED"""
        
        cache_key = (customer, p2p_model)
        
        if cache_key in self._p2p_series_cache:
            return self._p2p_series_cache[cache_key]
        
        # Build and cache
        series = self._get_p2p_series_indexed(customer, p2p_model, datos)
        self._p2p_series_cache[cache_key] = series
        
        return series
    
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
    
    def _ejecutar_media_movil(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Seasonal Moving Average - HEAVILY OPTIMIZED"""
        
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            cells_updated = 0
            processed_products = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns (Customer, Class) in Main sheet")
                return self._resultado_vacio()
            
            # Pre-allocate forecast columns with NaN
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # OPTIMIZATION: Extract all historical series at once
            historical_data = {}
            for idx in range(len(df_main)):
                series = pd.Series(dtype=float)
                row = df_main.iloc[idx]
                
                for date_col in datos['date_columns']:
                    val = pd.to_numeric(row.get(date_col, np.nan), errors='coerce')
                    series[date_col] = val
                
                historical_data[idx] = series
            
            # Progress bar
            progress_bar = st.progress(0)
            total_products = len(df_main)
            
            # Process each product row
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Get historical series (already computed)
                historical_series = historical_data[idx]
                
                # Process each forecast month
                for forecast_date in datos['forecast_dates']:
                    
                    # Get logic from cache
                    calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
                        customer, product_class, forecast_date
                    )
                    
                    # Handle "Não calcula"
                    if calc_base == self.CALC_BASE_NO_CALC:
                        forecast_arrays[forecast_date][idx] = 0
                        cells_updated += 1
                        continue
                    
                    # Handle launch dependent
                    if calc_base == self.CALC_BASE_LAUNCH_DEPENDENT:
                        if pd.notna(launch_month):
                            launch_date = pd.to_datetime(launch_month)
                            if forecast_date < launch_date:
                                forecast_arrays[forecast_date][idx] = 0
                                cells_updated += 1
                                continue
                    
                    # Handle P2P
                    if calc_base in [self.CALC_BASE_P2P, self.CALC_BASE_LAUNCH_DEPENDENT]:
                        
                        # Get historical series (from P2P or same product)
                        if pd.notna(p2p_model) and p2p_model != '':
                            series_to_use = self._get_p2p_series_cached(customer, p2p_model, datos)
                        else:
                            series_to_use = historical_series
                        
                        if len(series_to_use) > 0:
                            # Get seasonal window
                            seasonal_values = self._get_seasonal_window(forecast_date, series_to_use)
                            
                            if len(seasonal_values) > 0:
                                seasonal_avg = np.mean(seasonal_values)
                                
                                # Get growth factor from cache
                                year = forecast_date.year
                                growth_factor = self._get_growth_factor_cached(customer, year)
                                
                                forecasted_value = seasonal_avg * growth_factor
                                forecast_arrays[forecast_date][idx] = max(0, forecasted_value)
                                cells_updated += 1
                            else:
                                # Fallback to simple average
                                simple_avg = series_to_use.mean()
                                year = forecast_date.year
                                growth_factor = self._get_growth_factor_cached(customer, year)
                                forecast_arrays[forecast_date][idx] = max(0, simple_avg * growth_factor)
                                cells_updated += 1
                        else:
                            forecast_arrays[forecast_date][idx] = 0
                            cells_updated += 1
                
                processed_products += 1
                
                # Update progress every 100 products
                if processed_products % 100 == 0:
                    progress_bar.progress(min(processed_products / total_products, 1.0))
            
            # Clear progress bar
            progress_bar.empty()
            
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
                    'optimized': True
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
        """Execute Exponential Smoothing - OPTIMIZED"""
        
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            cells_updated = 0
            processed_products = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns in Main sheet")
                return self._resultado_vacio()
            
            # Pre-allocate forecast columns
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Progress bar
            progress_bar = st.progress(0)
            total_products = len(df_main)
            
            # Process each product row
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Get logic (use first forecast month)
                calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                # Get historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
                else:
                    historical_series = self._get_historical_series_indexed(idx, datos)
                
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
                            # Simplified
                            last_value = historical_series.dropna().iloc[-1]
                            for forecast_date in datos['forecast_dates']:
                                forecast_arrays[forecast_date][idx] = max(0, last_value)
                                cells_updated += 1
                            processed_products += 1
                    
                    except Exception as e:
                        # Fallback
                        avg_value = historical_series.mean()
                        for forecast_date in datos['forecast_dates']:
                            forecast_arrays[forecast_date][idx] = max(0, avg_value)
                            cells_updated += 1
                        processed_products += 1
                
                # Update progress
                if processed_products % 100 == 0:
                    progress_bar.progress(min(processed_products / total_products, 1.0))
            
            progress_bar.empty()
            
            # Add forecast columns
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'holt_winters' if STATSMODELS_AVAILABLE else 'simple_exponential',
                    'seasonal_periods': 12,
                    'optimized': True
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
        """Execute SARIMA - OPTIMIZED"""
        
        try:
            df_main = datos['df_main'].copy()
            df_result = df_main.copy()
            col_names = datos['col_names']
            
            arima_params = self.parametros.get('arima_params', (1, 1, 1))
            cells_updated = 0
            processed_products = 0
            
            customer_col = col_names.get('main_customer')
            model_col = col_names.get('main_product_model')
            class_col = col_names.get('main_class')
            
            if not customer_col or not class_col:
                st.error("❌ Could not find required columns in Main sheet")
                return self._resultado_vacio()
            
            # Pre-allocate
            forecast_arrays = {date: np.full(len(df_main), np.nan) for date in datos['forecast_dates']}
            
            # Progress bar
            progress_bar = st.progress(0)
            total_products = len(df_main)
            
            # Process each product
            for idx in range(len(df_main)):
                row = df_main.iloc[idx]
                
                customer = row.get(customer_col, '')
                product_class = row.get(class_col, '')
                
                if pd.isna(customer) or pd.isna(product_class):
                    continue
                
                # Get logic
                calc_base, p2p_model, launch_month = self._get_logic_for_product_cached(
                    customer, product_class, datos['forecast_dates'][0]
                )
                
                # Get historical series
                if pd.notna(p2p_model) and p2p_model != '':
                    historical_series = self._get_p2p_series_cached(customer, p2p_model, datos)
                else:
                    historical_series = self._get_historical_series_indexed(idx, datos)
                
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
                    
                    except Exception as e:
                        # Fallback
                        avg_value = historical_series.mean()
                        for forecast_date in datos['forecast_dates']:
                            forecast_arrays[forecast_date][idx] = max(0, avg_value)
                            cells_updated += 1
                        processed_products += 1
                
                # Update progress
                if processed_products % 50 == 0:
                    progress_bar.progress(min(processed_products / total_products, 1.0))
            
            progress_bar.empty()
            
            # Add forecast columns
            for forecast_date, values in forecast_arrays.items():
                df_result[forecast_date] = values
            
            return {
                'dataframe': df_result,
                'celulas_actualizadas': cells_updated,
                'parametros': {
                    'method': 'SARIMA' if SARIMAX_AVAILABLE else 'trend_based',
                    'order': f'({arima_params[0]},{arima_params[1]},{arima_params[2]})',
                    'optimized': True
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
        except Exception as e:
            pass
        
        return pd.Series(dtype=float)
    
    def _resultado_vacio(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'dataframe': pd.DataFrame(),
            'celulas_actualizadas': 0,
            'parametros': {},
            'metadata': {'error': True}
        }