# """
# Manejo de carga y validación de datos
# """
# import pandas as pd
# import streamlit as st
# from typing import Dict, List
# from datetime import datetime


# class DataHandler:
#     """Clase para cargar y validar archivos Excel"""
    
#     # Nombres de pestañas requeridas
#     REQUIRED_SHEETS = {
#         'resumo': 'Resumo',
#         'logicas': 'LogicasxMes',
#         'relaciones': 'Relaciones'
#     }
    
#     def __init__(self, uploaded_file):
#         """
#         Inicializa el manejador de datos
        
#         Args:
#             uploaded_file: Archivo subido desde Streamlit
#         """
#         self.uploaded_file = uploaded_file
#         self.dataframes = {}
#         self.errores = []
    
#     def cargar_archivo(self) -> bool:
#         """
#         Carga el archivo Excel y valida su estructura
        
#         Returns:
#             bool: True si la carga fue exitosa
#         """
#         try:
#             # Leer archivo Excel
#             excel_file = pd.ExcelFile(self.uploaded_file)
#             pestanas_disponibles = excel_file.sheet_names
            
#             st.info(f"📄 Pestañas encontradas: {', '.join(pestanas_disponibles)}")
            
#             # Validar pestañas requeridas
#             pestanas_requeridas = list(self.REQUIRED_SHEETS.values())
#             pestanas_faltantes = [p for p in pestanas_requeridas if p not in pestanas_disponibles]
            
#             if pestanas_faltantes:
#                 self.errores.append(f"⚠️ Pestañas faltantes: {', '.join(pestanas_faltantes)}")
#                 return False
            
#             # Cargar cada pestaña
#             return self._cargar_pestanas()
            
#         except Exception as e:
#             self.errores.append(f"❌ Error al leer el archivo: {str(e)}")
#             return False
    
#     def _cargar_pestanas(self) -> bool:
#         """
#         Carga todas las pestañas requeridas
        
#         Returns:
#             bool: True si todas las pestañas se cargaron correctamente
#         """
#         try:
#             # Cargar Resumo
#             with st.spinner("Cargando pestaña 'Resumo'..."):
#                 df_resumo = pd.read_excel(
#                     self.uploaded_file, 
#                     sheet_name=self.REQUIRED_SHEETS['resumo']
#                 )
#                 if df_resumo.empty:
#                     self.errores.append("❌ La pestaña 'Resumo' está vacía")
#                     return False
#                 self.dataframes['resumo'] = df_resumo
            
#             # Cargar LogicasxMes
#             with st.spinner("Cargando pestaña 'LogicasxMes'..."):
#                 df_logicas = pd.read_excel(
#                     self.uploaded_file, 
#                     sheet_name=self.REQUIRED_SHEETS['logicas']
#                 )
#                 if df_logicas.empty:
#                     self.errores.append("❌ La pestaña 'LogicasxMes' está vacía")
#                     return False
#                 self.dataframes['logicas'] = df_logicas
            
#             # Cargar Relaciones
#             with st.spinner("Cargando pestaña 'Relaciones'..."):
#                 df_relaciones = pd.read_excel(
#                     self.uploaded_file, 
#                     sheet_name=self.REQUIRED_SHEETS['relaciones']
#                 )
#                 if df_relaciones.empty:
#                     self.errores.append("❌ La pestaña 'Relaciones' está vacía")
#                     return False
#                 self.dataframes['relaciones'] = df_relaciones
            
#             return True
            
#         except Exception as e:
#             self.errores.append(f"❌ Error al cargar las pestañas: {str(e)}")
#             return False
    
#     def obtener_dataframes(self) -> Dict[str, pd.DataFrame]:
#         """
#         Obtiene los DataFrames cargados
        
#         Returns:
#             Dict con claves 'resumo', 'logicas', 'relaciones'
#         """
#         return self.dataframes
    
#     def obtener_errores(self) -> List[str]:
#         """
#         Obtiene la lista de errores encontrados
        
#         Returns:
#             Lista de mensajes de error
#         """
#         return self.errores
    
#     @staticmethod
#     def formatear_columnas(df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Formatea las columnas del DataFrame (convierte fechas, etc.)
        
#         Args:
#             df: DataFrame a formatear
            
#         Returns:
#             DataFrame con columnas formateadas
#         """
#         # Crear copia
#         df_formatted = df.copy()
        
#         # Convertir columnas de tipo datetime a string
#         new_columns = []
#         for col in df_formatted.columns:
#             if isinstance(col, datetime):
#                 new_columns.append(col.strftime('%Y-%m-%d'))
#             else:
#                 new_columns.append(str(col))
        
#         df_formatted.columns = new_columns
        
#         return df_formatted








"""
Data loading and validation handler - English version
"""
import pandas as pd
import streamlit as st
from typing import Dict, List
from datetime import datetime


class DataHandler:
    """Class to load and validate Excel files"""
    
    # Required sheet names (now in English)
    REQUIRED_SHEETS = {
        'main': 'Main',
        'logics': 'LogicsxMonth',
        'relations': 'Relations'
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
            
            # st.info(f"📄 Sheets found: {', '.join(available_sheets)}")
            
            # Validate required sheets
            required_sheets = list(self.REQUIRED_SHEETS.values())
            missing_sheets = [s for s in required_sheets if s not in available_sheets]
            
            if missing_sheets:
                self.errores.append(f"⚠️ Missing sheets: {', '.join(missing_sheets)}")
                return False
            
            # Load each sheet
            return self._cargar_pestanas()
            
        except Exception as e:
            self.errores.append(f"❌ Error reading file: {str(e)}")
            return False
    
    def _cargar_pestanas(self) -> bool:
        """
        Load all required sheets
        
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
                
                # st.success(f"✅ Forecast start date: {self.forecast_start_date}")
            
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
            
            return True
            
        except Exception as e:
            self.errores.append(f"❌ Error loading sheets: {str(e)}")
            st.exception(e)
            return False
    
    def obtener_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get loaded DataFrames
        
        Returns:
            Dict with keys 'main', 'logics', 'relations'
        """
        return self.dataframes
    
    def obtener_fecha_inicio(self) -> datetime:
        """
        Get forecast start date from B2
        
        Returns:
            Forecast start date
        """
        return self.forecast_start_date
    
    def obtener_errores(self) -> List[str]:
        """
        Get list of found errors
        
        Returns:
            List of error messages
        """
        return self.errores