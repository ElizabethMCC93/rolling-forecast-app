# """
# Data loading and validation handler - English version
# """
# import pandas as pd
# import streamlit as st
# from typing import Dict, List
# from datetime import datetime


# class DataHandler:
#     """Class to load and validate Excel files"""
    
#     # Required sheet names (now in English)
#     REQUIRED_SHEETS = {
#         'main': 'Main',
#         'logics': 'LogicsxMonth',
#         'relations': 'Relations'
#     }
    
#     def __init__(self, uploaded_file):
#         """
#         Initialize data handler
        
#         Args:
#             uploaded_file: File uploaded from Streamlit
#         """
#         self.uploaded_file = uploaded_file
#         self.dataframes = {}
#         self.errores = []
#         self.forecast_start_date = None
    
#     def cargar_archivo(self) -> bool:
#         """
#         Load Excel file and validate its structure
        
#         Returns:
#             bool: True if loading was successful
#         """
#         try:
#             # Read Excel file
#             excel_file = pd.ExcelFile(self.uploaded_file)
#             available_sheets = excel_file.sheet_names
            
#             # st.info(f"📄 Sheets found: {', '.join(available_sheets)}")
            
#             # Validate required sheets
#             required_sheets = list(self.REQUIRED_SHEETS.values())
#             missing_sheets = [s for s in required_sheets if s not in available_sheets]
            
#             if missing_sheets:
#                 self.errores.append(f"⚠️ Missing sheets: {', '.join(missing_sheets)}")
#                 return False
            
#             # Load each sheet
#             return self._cargar_pestanas()
            
#         except Exception as e:
#             self.errores.append(f"❌ Error reading file: {str(e)}")
#             return False
    
#     def _cargar_pestanas(self) -> bool:
#         """
#         Load all required sheets
        
#         Returns:
#             bool: True if all sheets loaded correctly
#         """
#         try:
#             # Load Main sheet
#             with st.spinner("Loading 'Main' sheet..."):
#                 df_main = pd.read_excel(
#                     self.uploaded_file, 
#                     sheet_name=self.REQUIRED_SHEETS['main'],
#                     header=1  # Row 2 is the header (index 1)
#                 )
                
#                 # Extract forecast start date from B2 (index [0,1])
#                 df_temp = pd.read_excel(
#                     self.uploaded_file, 
#                     sheet_name=self.REQUIRED_SHEETS['main'],
#                     header=None,
#                     nrows=2
#                 )
#                 self.forecast_start_date = df_temp.iloc[0, 1]  # B2 cell
                
#                 if df_main.empty:
#                     self.errores.append("❌ 'Main' sheet is empty")
#                     return False
                    
#                 self.dataframes['main'] = df_main
                
#                 # st.success(f"✅ Forecast start date: {self.forecast_start_date}")
            
#             # Load LogicsxMonth sheet
#             with st.spinner("Loading 'LogicsxMonth' sheet..."):
#                 df_logics = pd.read_excel(
#                     self.uploaded_file, 
#                     sheet_name=self.REQUIRED_SHEETS['logics'],
#                     header=1  # Row 2 is the header
#                 )
#                 if df_logics.empty:
#                     self.errores.append("❌ 'LogicsxMonth' sheet is empty")
#                     return False
#                 self.dataframes['logics'] = df_logics
            
#             # Load Relations sheet
#             with st.spinner("Loading 'Relations' sheet..."):
#                 df_relations = pd.read_excel(
#                     self.uploaded_file, 
#                     sheet_name=self.REQUIRED_SHEETS['relations'],
#                     header=7  # Row 8 is the header (index 7)
#                 )
#                 if df_relations.empty:
#                     self.errores.append("❌ 'Relations' sheet is empty")
#                     return False
#                 self.dataframes['relations'] = df_relations
            
#             return True
            
#         except Exception as e:
#             self.errores.append(f"❌ Error loading sheets: {str(e)}")
#             st.exception(e)
#             return False
    
#     def obtener_dataframes(self) -> Dict[str, pd.DataFrame]:
#         """
#         Get loaded DataFrames
        
#         Returns:
#             Dict with keys 'main', 'logics', 'relations'
#         """
#         return self.dataframes
    
#     def obtener_fecha_inicio(self) -> datetime:
#         """
#         Get forecast start date from B2
        
#         Returns:
#             Forecast start date
#         """
#         return self.forecast_start_date
    
#     def obtener_errores(self) -> List[str]:
#         """
#         Get list of found errors
        
#         Returns:
#             List of error messages
#         """
#         return self.errores




"""
Data loading and validation handler - English version
UPDATED: Added Models sheet support for preset model selection
"""
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime


class DataHandler:
    """Class to load and validate Excel files"""
    
    # Required sheet names (now in English)
    REQUIRED_SHEETS = {
        'main': 'Main',
        'logics': 'LogicsxMonth',
        'relations': 'Relations'
    }
    
    # Optional sheet for preset models
    OPTIONAL_SHEETS = {
        'models': 'Models'
    }
    
    def __init__(self, uploaded_file):
        """
        Initialize data handler
        
        Args:
            uploaded_file: File uploaded from Streamlit
        """
        self.uploaded_file = uploaded_file
        self.dataframes = {}
        self.errores = []
        self.forecast_start_date = None
        self.has_models_sheet = False
    
    def cargar_archivo(self) -> bool:
        """
        Load Excel file and validate its structure
        
        Returns:
            bool: True if loading was successful
        """
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(self.uploaded_file)
            available_sheets = excel_file.sheet_names
            
            # Validate required sheets
            required_sheets = list(self.REQUIRED_SHEETS.values())
            missing_sheets = [s for s in required_sheets if s not in available_sheets]
            
            if missing_sheets:
                self.errores.append(f"⚠️ Missing required sheets: {', '.join(missing_sheets)}")
                return False
            
            # Check for optional Models sheet
            models_sheet = self.OPTIONAL_SHEETS['models']
            self.has_models_sheet = models_sheet in available_sheets
            
            if self.has_models_sheet:
                st.info(f"✅ Found optional sheet '{models_sheet}' - Preset models feature enabled")
            else:
                st.info(f"ℹ️ Optional sheet '{models_sheet}' not found - Preset models feature disabled")
            
            # Load each sheet
            return self._cargar_pestanas()
            
        except Exception as e:
            self.errores.append(f"❌ Error reading file: {str(e)}")
            return False
    
    def _cargar_pestanas(self) -> bool:
        """
        Load all required sheets and optional Models sheet
        
        Returns:
            bool: True if all sheets loaded correctly
        """
        try:
            # Load Main sheet
            with st.spinner("Loading 'Main' sheet..."):
                df_main = pd.read_excel(
                    self.uploaded_file, 
                    sheet_name=self.REQUIRED_SHEETS['main'],
                    header=1  # Row 2 is the header (index 1)
                )
                
                # Extract forecast start date from B2 (index [0,1])
                df_temp = pd.read_excel(
                    self.uploaded_file, 
                    sheet_name=self.REQUIRED_SHEETS['main'],
                    header=None,
                    nrows=2
                )
                self.forecast_start_date = df_temp.iloc[0, 1]  # B2 cell
                
                if df_main.empty:
                    self.errores.append("❌ 'Main' sheet is empty")
                    return False
                    
                self.dataframes['main'] = df_main
            
            # Load LogicsxMonth sheet
            with st.spinner("Loading 'LogicsxMonth' sheet..."):
                df_logics = pd.read_excel(
                    self.uploaded_file, 
                    sheet_name=self.REQUIRED_SHEETS['logics'],
                    header=1  # Row 2 is the header
                )
                if df_logics.empty:
                    self.errores.append("❌ 'LogicsxMonth' sheet is empty")
                    return False
                self.dataframes['logics'] = df_logics
            
            # Load Relations sheet
            with st.spinner("Loading 'Relations' sheet..."):
                df_relations = pd.read_excel(
                    self.uploaded_file, 
                    sheet_name=self.REQUIRED_SHEETS['relations'],
                    header=7  # Row 8 is the header (index 7)
                )
                if df_relations.empty:
                    self.errores.append("❌ 'Relations' sheet is empty")
                    return False
                self.dataframes['relations'] = df_relations
            
            # Load Models sheet (optional)
            if self.has_models_sheet:
                with st.spinner("Loading 'Models' sheet..."):
                    try:
                        df_models = pd.read_excel(
                            self.uploaded_file,
                            sheet_name=self.OPTIONAL_SHEETS['models'],
                            header=0  # Row 1 is the header (A1: ID Customer, B1: Product Model, C1: Best Model)
                        )
                        
                        if df_models.empty:
                            st.warning("⚠️ 'Models' sheet is empty - Preset feature will be disabled")
                            self.has_models_sheet = False
                        else:
                            # Validate required columns
                            expected_cols = ['ID Customer', 'Product Model', 'Best Model']
                            missing_cols = [col for col in expected_cols if col not in df_models.columns]
                            
                            if missing_cols:
                                st.warning(f"⚠️ 'Models' sheet missing columns: {', '.join(missing_cols)} - Preset feature disabled")
                                self.has_models_sheet = False
                            else:
                                self.dataframes['models'] = df_models
                                st.success(f"✅ Loaded {len(df_models)} preset model configurations")
                    
                    except Exception as e:
                        st.warning(f"⚠️ Could not load 'Models' sheet: {str(e)} - Preset feature disabled")
                        self.has_models_sheet = False
            
            return True
            
        except Exception as e:
            self.errores.append(f"❌ Error loading sheets: {str(e)}")
            st.exception(e)
            return False
    
    def obtener_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get loaded DataFrames
        
        Returns:
            Dict with keys 'main', 'logics', 'relations', and optionally 'models'
        """
        return self.dataframes
    
    def obtener_fecha_inicio(self) -> datetime:
        """
        Get forecast start date from B2
        
        Returns:
            Forecast start date
        """
        return self.forecast_start_date
    
    def tiene_hoja_modelos(self) -> bool:
        """
        Check if Models sheet was loaded successfully
        
        Returns:
            bool: True if Models sheet is available
        """
        return self.has_models_sheet and 'models' in self.dataframes
    
    def obtener_errores(self) -> List[str]:
        """
        Get list of found errors
        
        Returns:
            List of error messages
        """
        return self.errores