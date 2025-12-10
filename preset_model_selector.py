"""
Preset Model Selector - Maps products to their best forecasting model
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List


class PresetModelSelector:
    """
    Determines which forecasting model to use for each product
    based on the Models sheet configuration
    """
    
    # Model name mapping (from Models sheet to internal keys)
    MODEL_MAPPING = {
        'Media movil': 'media_movil',
        'HWAS': 'suavizacao_exponencial',
        'SARIMA': 'arima',
        'Sem histórico': 'sem_historico'  # Special case: no forecast
    }
    
    def __init__(self, df_models: pd.DataFrame, df_main: pd.DataFrame):
        """
        Initialize preset model selector
        
        Args:
            df_models: DataFrame from Models sheet (ID Customer, Product Model, Best Model)
            df_main: DataFrame from Main sheet (with all product data)
        """
        self.df_models = df_models
        self.df_main = df_main
        
        # Build lookup dictionary for fast access
        self._build_model_lookup()
    
    def _build_model_lookup(self):
        """Build dictionary for O(1) model lookups"""
        
        self.model_lookup = {}
        
        for idx, row in self.df_models.iterrows():
            customer = str(row.get('ID Customer', '')).strip()
            product_model = str(row.get('Product Model', '')).strip()
            best_model = str(row.get('Best Model', '')).strip()
            
            # Skip invalid rows
            if not customer or not product_model or not best_model:
                continue
            
            # Create lookup key: (customer, product_model)
            key = (customer, product_model)
            
            # Map to internal model name
            internal_model = self.MODEL_MAPPING.get(best_model)
            
            if internal_model:
                self.model_lookup[key] = internal_model
            else:
                # Unknown model name - log warning but continue
                print(f"⚠️ Unknown model '{best_model}' for {customer} - {product_model}")
    
    def get_model_for_product(self, customer: str, product_model: str, 
                              reference_model: Optional[str] = None) -> str:
        """
        Get the best model to use for a specific product
        
        Args:
            customer: ID Customer
            product_model: Product Model (from column B)
            reference_model: Reference Model (from column G) - optional
        
        Returns:
            Internal model name ('media_movil', 'suavizacao_exponencial', 'arima', 'sem_historico')
            or None if not found
        """
        
        customer = str(customer).strip()
        product_model = str(product_model).strip()
        
        # PRIORITY 1: If has reference, use reference model
        if reference_model and pd.notna(reference_model):
            reference_str = str(reference_model).strip()
            
            # Check if reference is different from product model
            if reference_str and reference_str != '-' and reference_str != product_model:
                key = (customer, reference_str)
                
                if key in self.model_lookup:
                    return self.model_lookup[key]
        
        # PRIORITY 2: Use own product model
        key = (customer, product_model)
        
        if key in self.model_lookup:
            return self.model_lookup[key]
        
        # NOT FOUND: Return None (will be handled as "no forecast")
        return None
    
    def get_model_distribution(self) -> Dict[str, int]:
        """
        Get distribution of models in the Models sheet
        
        Returns:
            Dictionary with model names and counts
        """
        
        distribution = {}
        
        for model in self.model_lookup.values():
            distribution[model] = distribution.get(model, 0) + 1
        
        return distribution
    
    def validate_models_sheet(self) -> Tuple[bool, List[str]]:
        """
        Validate Models sheet structure and content
        
        Returns:
            Tuple: (is_valid, list_of_warnings)
        """
        
        warnings = []
        
        # Check if empty
        if self.df_models.empty:
            warnings.append("Models sheet is empty")
            return False, warnings
        
        # Check required columns
        required_cols = ['ID Customer', 'Product Model', 'Best Model']
        missing_cols = [col for col in required_cols if col not in self.df_models.columns]
        
        if missing_cols:
            warnings.append(f"Missing columns: {', '.join(missing_cols)}")
            return False, warnings
        
        # Check for unknown model names
        unique_models = self.df_models['Best Model'].dropna().unique()
        unknown_models = [m for m in unique_models if m not in self.MODEL_MAPPING]
        
        if unknown_models:
            warnings.append(f"Unknown model names: {', '.join(unknown_models)}")
        
        # Check for duplicate entries
        duplicates = self.df_models.duplicated(subset=['ID Customer', 'Product Model'], keep=False)
        
        if duplicates.any():
            dup_count = duplicates.sum()
            warnings.append(f"Found {dup_count} duplicate product configurations")
        
        is_valid = len(warnings) == 0 or all('Unknown model' not in w for w in warnings)
        
        return is_valid, warnings