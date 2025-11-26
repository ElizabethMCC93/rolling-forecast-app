# # models.py

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # Importaciones opcionales para modelos avanzados
# try:
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#     STATSMODELS_AVAILABLE = True
# except ImportError:
#     STATSMODELS_AVAILABLE = False

# try:
#     from statsmodels.tsa.arima.model import ARIMA
#     ARIMA_AVAILABLE = True
# except ImportError:
#     ARIMA_AVAILABLE = False

# def preparar_datos(df_resumo, df_logicas, df_relaciones, fecha_base):
#     """Prepara y limpia los datos para los modelos"""
    
#     # Convertir fecha_base a datetime si es necesario
#     if isinstance(fecha_base, str):
#         fecha_base = datetime.strptime(fecha_base, '%Y-%m-%d')
#     elif hasattr(fecha_base, 'date'):
#         fecha_base = fecha_base.date()
    
#     return {
#         'resumo': df_resumo.copy(),
#         'logicas': df_logicas.copy(),
#         'relaciones': df_relaciones.copy(),
#         'fecha_base': fecha_base
#     }

# class LogicaLancamento:
#     """Lógica complementaria para determinar datos históricos basados en mes de lanzamiento"""
    
#     def __init__(self, datos):
#         self.datos = datos
    
#     def es_logica_lancamento(self, classe):
#         """Verifica si una clase usa lógica de lanzamiento"""
#         try:
#             logicas_classe = self.datos['logicas'][
#                 self.datos['logicas'].iloc[:, 2] == classe  # Asumiendo columna 2 es clase
#             ]
            
#             if not logicas_classe.empty:
#                 logica_text = str(logicas_classe.iloc[0, 6]).lower()  # Asumiendo columna 6 es lógica
#                 return "depende do mês de lançamento" in logica_text or "depende do mes de lancamento" in logica_text
            
#             return False
#         except Exception:
#             return False
    
#     def obtener_serie_temporal(self, row):
#         """Determina qué datos históricos usar basado en lógica de lanzamiento"""
        
#         try:
#             # Buscar columnas de mes de lançamento y percentual
#             # Asumiendo que están en las últimas columnas como en tu macro
#             df_resumo = self.datos['resumo']
            
#             # Buscar mes de lançamento (columna AY = 51 en tu macro)
#             mes_lancamento = None
#             percentual_crescimento = None
#             produto_referencia = None
            
#             # Intentar encontrar las columnas correctas
#             for col_idx, col_name in enumerate(df_resumo.columns):
#                 if col_idx >= len(row):
#                     break
                    
#                 # Buscar fecha de lanzamiento
#                 if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (datetime, pd.Timestamp)):
#                     mes_lancamento = row.iloc[col_idx]
                
#                 # Buscar percentual (número entre 0 y 10 típicamente)
#                 if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (int, float)):
#                     if 0 < row.iloc[col_idx] <= 10:
#                         percentual_crescimento = row.iloc[col_idx]
            
#             # Buscar produto de referencia (DEPARA SEGUINTE)
#             if len(row) > 7:  # Columna H = índice 7
#                 produto_referencia = row.iloc[7]
            
#             if not all([mes_lancamento, percentual_crescimento, produto_referencia]):
#                 return np.array([])
            
#             # Buscar datos del producto de referencia
#             cliente = row.iloc[0] if len(row) > 0 else None
            
#             if cliente is None:
#                 return np.array([])
            
#             # Buscar fila del producto de referencia
#             produto_ref_mask = (
#                 (df_resumo.iloc[:, 0] == cliente) &  # Cliente
#                 (df_resumo.iloc[:, 1] == produto_referencia)  # Produto
#             )
            
#             produto_ref_rows = df_resumo[produto_ref_mask]
            
#             if produto_ref_rows.empty:
#                 return np.array([])
            
#             # Extraer serie temporal del producto de referencia
#             produto_ref_row = produto_ref_rows.iloc[0]
#             serie_referencia = self.extraer_serie_desde_lancamento(produto_ref_row, mes_lancamento)
            
#             # Aplicar factor de crecimiento
#             return serie_referencia * percentual_crescimento
            
#         except Exception as e:
#             return np.array([])
    
#     def extraer_serie_desde_lancamento(self, row_referencia, mes_lancamento):
#         """Extrae datos históricos desde el mes de lanzamiento hacia atrás"""
        
#         try:
#             valores = []
            
#             # Buscar columnas con fechas
#             for col_idx, col_val in enumerate(row_referencia):
#                 if col_idx < 2:  # Saltar cliente y producto
#                     continue
                
#                 # Intentar encontrar columnas de fechas y valores
#                 if pd.notna(col_val) and isinstance(col_val, (int, float)) and col_val > 0:
#                     valores.append(col_val)
            
#             # Tomar últimos 12 valores como máximo
#             if len(valores) > 12:
#                 valores = valores[-12:]
            
#             return np.array(valores)
            
#         except Exception:
#             return np.array([])

# class ModeloBase:
#     def __init__(self, datos):
#         self.datos = datos
#         self.logica_lancamento = LogicaLancamento(datos)
    
#     def obtener_datos_historicos(self, row):
#         """Determina qué datos históricos usar según la lógica"""
        
#         try:
#             # Obtener clase (asumiendo columna 5 = índice 5)
#             classe = row.iloc[5] if len(row) > 5 else None
            
#             if classe and self.logica_lancamento.es_logica_lancamento(classe):
#                 # Usar lógica de lanzamiento
#                 return self.logica_lancamento.obtener_serie_temporal(row)
#             else:
#                 # Usar datos históricos normales
#                 return self.extraer_serie_temporal_normal(row)
                
#         except Exception:
#             return self.extraer_serie_temporal_normal(row)
    
#     def extraer_serie_temporal_normal(self, row):
#         """Extrae serie temporal normal de la fila"""
        
#         try:
#             valores = []
            
#             # Buscar valores numéricos en la fila (saltando primeras columnas de metadata)
#             for col_idx in range(2, len(row)):  # Empezar desde columna 2
#                 val = row.iloc[col_idx]
#                 if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
#                     valores.append(val)
            
#             # Tomar últimos 12 valores como máximo
#             if len(valores) > 12:
#                 valores = valores[-12:]
            
#             return np.array(valores)
            
#         except Exception:
#             return np.array([])
    
#     def obtener_factor(self, cliente, año):
#         """Obtiene factor desde tabla de relaciones"""
        
#         try:
#             relaciones = self.datos['relaciones']
            
#             # Buscar cliente en primera columna
#             cliente_mask = relaciones.iloc[:, 0] == cliente
#             cliente_rows = relaciones[cliente_mask]
            
#             if cliente_rows.empty:
#                 return 1.0
            
#             # Buscar columna del año
#             if año == 2026 and len(cliente_rows.columns) > 1:
#                 factor = cliente_rows.iloc[0, 1]  # Columna B
#             elif año == 2027 and len(cliente_rows.columns) > 2:
#                 factor = cliente_rows.iloc[0, 2]  # Columna C
#             else:
#                 return 1.0
            
#             return float(factor) if pd.notna(factor) else 1.0
            
#         except Exception:
#             return 1.0
    
#     def calcular_precisao(self, df_resultado):
#         """Calcula precisión básica"""
#         try:
#             if 'forecast' in df_resultado.columns:
#                 forecasts = df_resultado['forecast'].dropna()
#                 return len(forecasts) / len(df_resultado) if len(df_resultado) > 0 else 0
#             return 0
#         except Exception:
#             return 0

# class ModeloMediaMovil(ModeloBase):
#     """Modelo de Media Móvil (lógica actual)"""
    
#     def calcular(self):
#         contador = 0
#         df_resultado = self.datos['resumo'].copy()
#         df_resultado['forecast'] = np.nan
        
#         try:
#             for index, row in df_resultado.iterrows():
#                 # Obtener datos históricos
#                 serie_historica = self.obtener_datos_historicos(row)
                
#                 if len(serie_historica) >= 3:
#                     # Calcular media móvil de últimos 3 períodos
#                     media_movil = np.mean(serie_historica[-3:])
                    
#                     # Obtener factor
#                     cliente = row.iloc[0] if len(row) > 0 else None
#                     año_objetivo = 2026  # Por defecto, podrías calcularlo dinámicamente
                    
#                     if cliente:
#                         factor = self.obtener_factor(cliente, año_objetivo)
#                         forecast_value = media_movil * factor
                        
#                         df_resultado.loc[index, 'forecast'] = forecast_value
#                         contador += 1
            
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_atualizadas': contador,
#                 'modelo': 'Media Móvil',
#                 'precisao': self.calcular_precisao(df_resultado)
#             }
            
#         except Exception as e:
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_atualizadas': 0,
#                 'modelo': 'Media Móvil',
#                 'error': str(e),
#                 'precisao': 0
#             }

# class SuavizacaoExponencial(ModeloBase):
#     """Suavización Exponencial Simple"""
    
#     def __init__(self, datos, alpha=0.3):
#         super().__init__(datos)
#         self.alpha = alpha
    
#     def calcular(self):
#         if not STATSMODELS_AVAILABLE:
#             return {
#                 'dataframe': self.datos['resumo'].copy(),
#                 'celulas_atualizadas': 0,
#                 'modelo': 'Suavización Exponencial',
#                 'error': 'statsmodels no está disponible',
#                 'precisao': 0
#             }
        
#         contador = 0
#         df_resultado = self.datos['resumo'].copy()
#         df_resultado['forecast'] = np.nan
        
#         try:
#             for index, row in df_resultado.iterrows():
#                 serie_historica = self.obtener_datos_historicos(row)
                
#                 if len(serie_historica) >= 6:  # Mínimo para suavización
#                     try:
#                         # Aplicar suavización exponencial simple
#                         modelo = ExponentialSmoothing(
#                             serie_historica, 
#                             trend=None, 
#                             seasonal=None
#                         )
#                         fitted_model = modelo.fit(smoothing_level=self.alpha)
                        
#                         # Forecast próximo período
#                         forecast = fitted_model.forecast(steps=1)[0]
                        
#                         if forecast > 0:
#                             df_resultado.loc[index, 'forecast'] = forecast
#                             contador += 1
                    
#                     except Exception:
#                         continue
            
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_atualizadas': contador,
#                 'modelo': 'Suavización Exponencial',
#                 'parametros': {'alpha': self.alpha},
#                 'precisao': self.calcular_precisao(df_resultado)
#             }
            
#         except Exception as e:
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_atualizadas': 0,
#                 'modelo': 'Suavización Exponencial',
#                 'error': str(e),
#                 'precisao': 0
#             }

# class ModeloARIMA(ModeloBase):
#     """Modelo ARIMA"""
    
#     def __init__(self, datos, params=(1,1,1)):
#         super().__init__(datos)
#         self.p, self.d, self.q = params
    
#     def calcular(self):
#         if not ARIMA_AVAILABLE:
#             return {
#                 'dataframe': self.datos['resumo'].copy(),
#                 'celulas_atualizadas': 0,
#                 'modelo': 'ARIMA',
#                 'error': 'statsmodels.tsa.arima no está disponible',
#                 'precisao': 0
#             }
        
#         contador = 0
#         df_resultado = self.datos['resumo'].copy()
#         df_resultado['forecast'] = np.nan
        
#         try:
#             for index, row in df_resultado.iterrows():
#                 serie_historica = self.obtener_datos_historicos(row)
                
#                 if len(serie_historica) >= 12:  # ARIMA necesita más datos
#                     try:
#                         # Ajustar modelo ARIMA
#                         modelo = ARIMA(serie_historica, order=(self.p, self.d, self.q))
#                         fitted_model = modelo.fit()
                        
#                         # Forecast próximo período
#                         forecast = fitted_model.forecast(steps=1)[0]
                        
#                         if forecast > 0:
#                             df_resultado.loc[index, 'forecast'] = forecast
#                             contador += 1
                    
#                     except Exception:
#                         continue
            
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_atualizadas': contador,
#                 'modelo': 'ARIMA',
#                 'parametros': {'p': self.p, 'd': self.d, 'q': self.q},
#                 'precisao': self.calcular_precisao(df_resultado)
#             }
            
#         except Exception as e:
#             return {
#                 'dataframe': df_resultado,
#                 'celulas_atualizadas': 0,
#                 'modelo': 'ARIMA',
#                 'error': str(e),
#                 'precisao': 0
#             }









# models.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importaciones opcionales para modelos avanzados
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

def preparar_datos(df_resumo, df_logicas, df_relaciones, fecha_base):
    """Prepara y limpia los datos para los modelos"""
    
    # Convertir fecha_base a datetime si es necesario
    if isinstance(fecha_base, str):
        fecha_base = datetime.strptime(fecha_base, '%Y-%m-%d')
    elif hasattr(fecha_base, 'date'):
        fecha_base = fecha_base.date()
    
    return {
        'resumo': df_resumo.copy(),
        'logicas': df_logicas.copy(),
        'relaciones': df_relaciones.copy(),
        'fecha_base': fecha_base
    }

class LogicaLancamento:
    """Lógica complementaria para determinar datos históricos basados en mes de lanzamiento"""
    
    def __init__(self, datos):
        self.datos = datos
    
    def es_logica_lancamento(self, classe):
        """Verifica si una clase usa lógica de lanzamiento"""
        try:
            logicas_classe = self.datos['logicas'][
                self.datos['logicas'].iloc[:, 2] == classe  # Asumiendo columna 2 es clase
            ]
            
            if not logicas_classe.empty:
                logica_text = str(logicas_classe.iloc[0, 6]).lower()  # Asumiendo columna 6 es lógica
                return "depende do mês de lançamento" in logica_text or "depende do mes de lancamento" in logica_text
            
            return False
        except Exception:
            return False
    
    def obtener_serie_temporal(self, row):
        """Determina qué datos históricos usar basado en lógica de lanzamiento"""
        
        try:
            # Buscar columnas de mes de lançamento y percentual
            # Asumiendo que están en las últimas columnas como en tu macro
            df_resumo = self.datos['resumo']
            
            # Buscar mes de lançamento (columna AY = 51 en tu macro)
            mes_lancamento = None
            percentual_crescimento = None
            produto_referencia = None
            
            # Intentar encontrar las columnas correctas
            for col_idx, col_name in enumerate(df_resumo.columns):
                if col_idx >= len(row):
                    break
                    
                # Buscar fecha de lanzamiento
                if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (datetime, pd.Timestamp)):
                    mes_lancamento = row.iloc[col_idx]
                
                # Buscar percentual (número entre 0 y 10 típicamente)
                if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (int, float)):
                    if 0 < row.iloc[col_idx] <= 10:
                        percentual_crescimento = row.iloc[col_idx]
            
            # Buscar produto de referencia (DEPARA SEGUINTE)
            if len(row) > 7:  # Columna H = índice 7
                produto_referencia = row.iloc[7]
            
            if not all([mes_lancamento, percentual_crescimento, produto_referencia]):
                return np.array([])
            
            # Buscar datos del producto de referencia
            cliente = row.iloc[0] if len(row) > 0 else None
            
            if cliente is None:
                return np.array([])
            
            # Buscar fila del producto de referencia
            produto_ref_mask = (
                (df_resumo.iloc[:, 0] == cliente) &  # Cliente
                (df_resumo.iloc[:, 1] == produto_referencia)  # Produto
            )
            
            produto_ref_rows = df_resumo[produto_ref_mask]
            
            if produto_ref_rows.empty:
                return np.array([])
            
            # Extraer serie temporal del producto de referencia
            produto_ref_row = produto_ref_rows.iloc[0]
            serie_referencia = self.extraer_serie_desde_lancamento(produto_ref_row, mes_lancamento)
            
            # Aplicar factor de crecimiento
            return serie_referencia * percentual_crescimento
            
        except Exception as e:
            return np.array([])
    
    def extraer_serie_desde_lancamento(self, row_referencia, mes_lancamento):
        """Extrae datos históricos desde el mes de lanzamiento hacia atrás"""
        
        try:
            valores = []
            
            # Buscar columnas con fechas
            for col_idx, col_val in enumerate(row_referencia):
                if col_idx < 2:  # Saltar cliente y producto
                    continue
                
                # Intentar encontrar columnas de fechas y valores
                if pd.notna(col_val) and isinstance(col_val, (int, float)) and col_val > 0:
                    valores.append(col_val)
            
            # Tomar últimos 12 valores como máximo
            if len(valores) > 12:
                valores = valores[-12:]
            
            return np.array(valores)
            
        except Exception:
            return np.array([])

class ModeloBase:
    def __init__(self, datos):
        self.datos = datos
        self.logica_lancamento = LogicaLancamento(datos)
    
    def parsear_fecha(self, valor):
        """
        Parsea una fecha en formato 01/mm/aaaa u otros formatos comunes
        """
        if pd.isna(valor):
            return None
        
        try:
            # Si ya es datetime, retornar
            if isinstance(valor, (datetime, pd.Timestamp)):
                return pd.to_datetime(valor)
            
            # Si es string, intentar parsear
            if isinstance(valor, str):
                valor = valor.strip()
                
                # Intentar formato dd/mm/yyyy
                try:
                    return pd.to_datetime(valor, format='%d/%m/%Y')
                except:
                    pass
                
                # Intentar formato dd/mm/yy
                try:
                    return pd.to_datetime(valor, format='%d/%m/%y')
                except:
                    pass
                
                # Intentar otros formatos comunes
                try:
                    return pd.to_datetime(valor, dayfirst=True)
                except:
                    pass
            
            return None
            
        except Exception:
            return None
    
    def obtener_fecha_inicio_proyeccion(self, row):
        """
        Verifica si existe una fecha en la columna H (índice 7 - encabezado XF).
        Si existe, retorna esa fecha. Si no, retorna None (proyectar todo el periodo).
        Formato esperado: 01/mm/aaaa
        """
        try:
            # Columna H = índice 7 (A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7)
            if len(row) > 7:
                valor_columna_h = row.iloc[7]
                fecha_parseada = self.parsear_fecha(valor_columna_h)
                
                if fecha_parseada is not None:
                    print(f"[DEBUG] Fecha encontrada en columna H para fila {row.name}: {fecha_parseada}")
                
                return fecha_parseada
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Error al obtener fecha de columna H: {e}")
            return None
    
    def debe_proyectar_en_mes(self, fecha_inicio_proyeccion, año_mes):
        """
        Determina si se debe proyectar en un mes específico basado en la fecha de inicio.
        
        Args:
            fecha_inicio_proyeccion: Fecha desde la cual se debe empezar a proyectar (de columna H)
            año_mes: Tupla (año, mes) del periodo a evaluar, ej: (2026, 1) para enero 2026
        
        Returns:
            True si se debe proyectar, False si debe quedar vacío
        """
        try:
            # Si no hay restricción de fecha, proyectar siempre
            if fecha_inicio_proyeccion is None:
                return True
            
            # Asegurar que fecha_inicio_proyeccion sea datetime
            if not isinstance(fecha_inicio_proyeccion, (datetime, pd.Timestamp)):
                fecha_inicio_proyeccion = pd.to_datetime(fecha_inicio_proyeccion)
            
            # Crear fecha del periodo a evaluar (primer día del mes)
            año_eval, mes_eval = año_mes
            fecha_periodo = datetime(año_eval, mes_eval, 1)
            
            # Normalizar fecha_inicio a primer día del mes para comparación justa
            fecha_inicio_normalizada = datetime(
                fecha_inicio_proyeccion.year,
                fecha_inicio_proyeccion.month,
                1
            )
            
            # Solo proyectar si el periodo es igual o posterior a la fecha de inicio
            resultado = fecha_periodo >= fecha_inicio_normalizada
            
            print(f"[DEBUG] Comparando: Periodo {año_eval}/{mes_eval} vs Inicio {fecha_inicio_normalizada.year}/{fecha_inicio_normalizada.month} = {resultado}")
            
            return resultado
            
        except Exception as e:
            print(f"[DEBUG] Error en debe_proyectar_en_mes: {e}")
            # En caso de error, proyectar por defecto
            return True
    
    def usar_producto_referencia(self, row):
        """
        Verifica si columna G es diferente de columna B.
        Si es diferente, retorna el producto de referencia (columna G).
        Si es igual o G está vacío, retorna None.
        
        Columna A (índice 0): ID Customer
        Columna B (índice 1): Modelo actual
        Columna G (índice 6): Modelo de referencia
        
        Args:
            row: Fila del DataFrame
        
        Returns:
            Producto de referencia (string) o None
        """
        try:
            # Columna B = índice 1 (modelo actual)
            # Columna G = índice 6 (modelo de referencia)
            
            if len(row) <= 6:
                return None
            
            producto_actual = row.iloc[1]  # Columna B (Modelo actual)
            producto_referencia = row.iloc[6]  # Columna G (Modelo de referencia)
            
            # Limpiar valores
            producto_actual_str = str(producto_actual).strip() if pd.notna(producto_actual) else ''
            producto_referencia_str = str(producto_referencia).strip() if pd.notna(producto_referencia) else ''
            
            # Verificar si hay producto de referencia y es diferente del actual
            if producto_referencia_str != '' and producto_actual_str != producto_referencia_str:
                print(f"[DEBUG] Producto actual (B): '{producto_actual_str}' != Producto referencia (G): '{producto_referencia_str}'")
                return producto_referencia_str
            
            return None
            
        except Exception as e:
            print(f"[DEBUG] Error en usar_producto_referencia: {e}")
            return None
    
    def obtener_datos_producto_referencia(self, row, producto_referencia):
        """
        Busca y extrae los datos históricos del producto de referencia.
        CRÍTICO: Debe buscar por ID Customer (columna A) + Modelo (columna B) = producto_referencia
        
        Args:
            row: Fila actual
            producto_referencia: Nombre del producto de referencia (valor de columna G)
        
        Returns:
            Array con serie temporal del producto de referencia
        """
        try:
            df_resumo = self.datos['resumo']
            
            # Columna A (índice 0): ID Customer
            id_customer = row.iloc[0] if len(row) > 0 else None
            
            if id_customer is None or pd.isna(id_customer):
                print(f"[DEBUG] ID Customer no encontrado")
                return np.array([])
            
            # Limpiar valores para comparación
            id_customer_str = str(id_customer).strip()
            producto_referencia_str = str(producto_referencia).strip()
            
            print(f"[DEBUG] Buscando: ID Customer = '{id_customer_str}', Modelo = '{producto_referencia_str}'")
            
            # Buscar fila del producto de referencia con:
            # - Mismo ID Customer (columna A, índice 0)
            # - Modelo = producto_referencia (columna B, índice 1)
            produto_ref_mask = (
                df_resumo.iloc[:, 0].astype(str).str.strip() == id_customer_str
            ) & (
                df_resumo.iloc[:, 1].astype(str).str.strip() == produto_referencia_str
            )
            
            produto_ref_rows = df_resumo[produto_ref_mask]
            
            print(f"[DEBUG] Filas encontradas: {len(produto_ref_rows)}")
            
            if produto_ref_rows.empty:
                print(f"[DEBUG] No se encontró producto de referencia para ID Customer '{id_customer_str}' y Modelo '{produto_referencia_str}'")
                return np.array([])
            
            # Extraer serie temporal del producto de referencia
            produto_ref_row = produto_ref_rows.iloc[0]
            serie = self.extraer_serie_temporal_normal(produto_ref_row)
            
            print(f"[DEBUG] Serie temporal extraída: {len(serie)} valores")
            
            return serie
            
        except Exception as e:
            print(f"[DEBUG] Error en obtener_datos_producto_referencia: {e}")
            return np.array([])
    
    def obtener_datos_historicos(self, row):
        """
        Determina qué datos históricos usar según la lógica:
        PRIORIDAD 1: Si columna G ≠ columna B: usar datos del producto en G (mismo ID Customer)
        PRIORIDAD 2: Si clase tiene lógica de lanzamiento: usar lógica de lanzamiento
        PRIORIDAD 3: Caso contrario: usar datos históricos normales
        """
        
        try:
            # PRIORIDAD 1: Verificar si hay producto de referencia (G ≠ B)
            produto_referencia = self.usar_producto_referencia(row)
            
            if produto_referencia is not None:
                print(f"[DEBUG] Usando producto de referencia: {produto_referencia}")
                # Usar datos del producto de referencia (mismo ID Customer)
                serie_ref = self.obtener_datos_producto_referencia(row, produto_referencia)
                if len(serie_ref) > 0:
                    print(f"[DEBUG] Serie de referencia obtenida: {len(serie_ref)} valores")
                    return serie_ref
                else:
                    print(f"[DEBUG] No se encontraron datos para producto de referencia, usando datos propios")
            
            # PRIORIDAD 2: Verificar lógica de lanzamiento
            classe = row.iloc[5] if len(row) > 5 else None
            
            if classe and self.logica_lancamento.es_logica_lancamento(classe):
                # Usar lógica de lanzamiento
                serie_lanc = self.logica_lancamento.obtener_serie_temporal(row)
                if len(serie_lanc) > 0:
                    return serie_lanc
            
            # PRIORIDAD 3: Usar datos históricos normales
            return self.extraer_serie_temporal_normal(row)
                
        except Exception as e:
            print(f"[DEBUG] Error en obtener_datos_historicos: {e}")
            return self.extraer_serie_temporal_normal(row)
    
    def extraer_serie_temporal_normal(self, row):
        """Extrae serie temporal normal de la fila"""
        
        try:
            valores = []
            
            # Buscar valores numéricos en la fila (saltando primeras columnas de metadata)
            # Asumiendo que las columnas 0-7 son metadata y desde 8 en adelante son datos históricos
            for col_idx in range(8, len(row)):  # Empezar desde columna después de H
                val = row.iloc[col_idx]
                if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                    valores.append(val)
            
            # Si no encontró nada, intentar desde columna 2
            if len(valores) == 0:
                for col_idx in range(2, len(row)):
                    val = row.iloc[col_idx]
                    if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                        valores.append(val)
            
            # Tomar últimos 12 valores como máximo
            if len(valores) > 12:
                valores = valores[-12:]
            
            return np.array(valores)
            
        except Exception:
            return np.array([])
    
    def obtener_factor(self, cliente, año):
        """Obtiene factor desde tabla de relaciones"""
        
        try:
            relaciones = self.datos['relaciones']
            
            # Buscar cliente en primera columna
            cliente_mask = relaciones.iloc[:, 0] == cliente
            cliente_rows = relaciones[cliente_mask]
            
            if cliente_rows.empty:
                return 1.0
            
            # Buscar columna del año
            if año == 2026 and len(cliente_rows.columns) > 1:
                factor = cliente_rows.iloc[0, 1]  # Columna B
            elif año == 2027 and len(cliente_rows.columns) > 2:
                factor = cliente_rows.iloc[0, 2]  # Columna C
            else:
                return 1.0
            
            return float(factor) if pd.notna(factor) else 1.0
            
        except Exception:
            return 1.0
    
    def calcular_precisao(self, df_resultado):
        """Calcula precisión básica"""
        try:
            forecast_cols = [col for col in df_resultado.columns if col.startswith('forecast')]
            if len(forecast_cols) > 0:
                total_forecasts = 0
                for col in forecast_cols:
                    total_forecasts += df_resultado[col].notna().sum()
                return total_forecasts / (len(df_resultado) * len(forecast_cols)) if len(df_resultado) > 0 else 0
            return 0
        except Exception:
            return 0

class ModeloMediaMovil(ModeloBase):
    """Modelo de Media Móvil (lógica actual)"""
    
    def calcular(self, periodo_proyeccion=None):
        """
        Calcula forecast usando media móvil
        
        Args:
            periodo_proyeccion: Lista de tuplas (año, mes) para proyectar, 
                              ej: [(2026, 1), (2026, 2), ...] 
                              Si None, proyecta un solo periodo
        """
        contador = 0
        df_resultado = self.datos['resumo'].copy()
        
        # Si hay periodo de proyección múltiple, crear columnas para cada mes
        if periodo_proyeccion is not None and len(periodo_proyeccion) > 0:
            for año, mes in periodo_proyeccion:
                col_name = f'forecast_{año}_{mes:02d}'
                df_resultado[col_name] = np.nan
        else:
            df_resultado['forecast'] = np.nan
        
        try:
            for index, row in df_resultado.iterrows():
                print(f"\n[DEBUG] ========== Procesando fila {index} ==========")
                
                # Verificar si hay fecha de inicio de proyección en columna H
                fecha_inicio_proyeccion = self.obtener_fecha_inicio_proyeccion(row)
                
                # Obtener datos históricos (considerando G ≠ B)
                serie_historica = self.obtener_datos_historicos(row)
                
                print(f"[DEBUG] Serie histórica: {len(serie_historica)} valores")
                
                if len(serie_historica) >= 3:
                    # Calcular media móvil de últimos 3 períodos
                    media_movil = np.mean(serie_historica[-3:])
                    
                    print(f"[DEBUG] Media móvil calculada: {media_movil}")
                    
                    # Obtener cliente
                    cliente = row.iloc[0] if len(row) > 0 else None
                    
                    if cliente:
                        # Proyectar para cada periodo
                        if periodo_proyeccion is not None and len(periodo_proyeccion) > 0:
                            for año, mes in periodo_proyeccion:
                                # Verificar si debe proyectar en este mes
                                if self.debe_proyectar_en_mes(fecha_inicio_proyeccion, (año, mes)):
                                    factor = self.obtener_factor(cliente, año)
                                    forecast_value = media_movil * factor
                                    
                                    col_name = f'forecast_{año}_{mes:02d}'
                                    df_resultado.loc[index, col_name] = forecast_value
                                    contador += 1
                                    
                                    print(f"[DEBUG] Proyectado {col_name}: {forecast_value}")
                                else:
                                    print(f"[DEBUG] Mes {año}/{mes} omitido por restricción de fecha")
                        else:
                            # Proyección simple (un solo periodo)
                            año_objetivo = 2026  # Por defecto
                            factor = self.obtener_factor(cliente, año_objetivo)
                            forecast_value = media_movil * factor
                            
                            df_resultado.loc[index, 'forecast'] = forecast_value
                            contador += 1
                else:
                    print(f"[DEBUG] Datos insuficientes para calcular (necesita al menos 3 valores)")
            
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': contador,
                'modelo': 'Media Móvil',
                'precisao': self.calcular_precisao(df_resultado)
            }
            
        except Exception as e:
            print(f"[ERROR] Error en calcular: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': 0,
                'modelo': 'Media Móvil',
                'error': str(e),
                'precisao': 0
            }

class SuavizacaoExponencial(ModeloBase):
    """Suavización Exponencial Simple"""
    
    def __init__(self, datos, alpha=0.3):
        super().__init__(datos)
        self.alpha = alpha
    
    def calcular(self, periodo_proyeccion=None):
        if not STATSMODELS_AVAILABLE:
            return {
                'dataframe': self.datos['resumo'].copy(),
                'celulas_atualizadas': 0,
                'modelo': 'Suavización Exponencial',
                'error': 'statsmodels no está disponible',
                'precisao': 0
            }
        
        contador = 0
        df_resultado = self.datos['resumo'].copy()
        
        # Si hay periodo de proyección múltiple, crear columnas para cada mes
        if periodo_proyeccion is not None and len(periodo_proyeccion) > 0:
            for año, mes in periodo_proyeccion:
                col_name = f'forecast_{año}_{mes:02d}'
                df_resultado[col_name] = np.nan
        else:
            df_resultado['forecast'] = np.nan
        
        try:
            for index, row in df_resultado.iterrows():
                # Verificar si hay fecha de inicio de proyección en columna H
                fecha_inicio_proyeccion = self.obtener_fecha_inicio_proyeccion(row)
                
                # Obtener datos históricos (considerando G ≠ B)
                serie_historica = self.obtener_datos_historicos(row)
                
                if len(serie_historica) >= 6:  # Mínimo para suavización
                    try:
                        # Aplicar suavización exponencial simple
                        modelo = ExponentialSmoothing(
                            serie_historica, 
                            trend=None, 
                            seasonal=None
                        )
                        fitted_model = modelo.fit(smoothing_level=self.alpha)
                        
                        # Obtener cliente
                        cliente = row.iloc[0] if len(row) > 0 else None
                        
                        if cliente:
                            # Proyectar para cada periodo
                            if periodo_proyeccion is not None and len(periodo_proyeccion) > 0:
                                for año, mes in periodo_proyeccion:
                                    # Verificar si debe proyectar en este mes
                                    if self.debe_proyectar_en_mes(fecha_inicio_proyeccion, (año, mes)):
                                        # Forecast próximo período
                                        forecast = fitted_model.forecast(steps=1)[0]
                                        
                                        if forecast > 0:
                                            col_name = f'forecast_{año}_{mes:02d}'
                                            df_resultado.loc[index, col_name] = forecast
                                            contador += 1
                            else:
                                # Proyección simple
                                forecast = fitted_model.forecast(steps=1)[0]
                                
                                if forecast > 0:
                                    df_resultado.loc[index, 'forecast'] = forecast
                                    contador += 1
                    
                    except Exception:
                        continue
            
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': contador,
                'modelo': 'Suavización Exponencial',
                'parametros': {'alpha': self.alpha},
                'precisao': self.calcular_precisao(df_resultado)
            }
            
        except Exception as e:
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': 0,
                'modelo': 'Suavización Exponencial',
                'error': str(e),
                'precisao': 0
            }

class ModeloARIMA(ModeloBase):
    """Modelo ARIMA"""
    
    def __init__(self, datos, params=(1,1,1)):
        super().__init__(datos)
        self.p, self.d, self.q = params
    
    def calcular(self, periodo_proyeccion=None):
        if not ARIMA_AVAILABLE:
            return {
                'dataframe': self.datos['resumo'].copy(),
                'celulas_atualizadas': 0,
                'modelo': 'ARIMA',
                'error': 'statsmodels.tsa.arima no está disponible',
                'precisao': 0
            }
        
        contador = 0
        df_resultado = self.datos['resumo'].copy()
        
        # Si hay periodo de proyección múltiple, crear columnas para cada mes
        if periodo_proyeccion is not None and len(periodo_proyeccion) > 0:
            for año, mes in periodo_proyeccion:
                col_name = f'forecast_{año}_{mes:02d}'
                df_resultado[col_name] = np.nan
        else:
            df_resultado['forecast'] = np.nan
        
        try:
            for index, row in df_resultado.iterrows():
                # Verificar si hay fecha de inicio de proyección en columna H
                fecha_inicio_proyeccion = self.obtener_fecha_inicio_proyeccion(row)
                
                # Obtener datos históricos (considerando G ≠ B)
                serie_historica = self.obtener_datos_historicos(row)
                
                if len(serie_historica) >= 12:  # ARIMA necesita más datos
                    try:
                        # Ajustar modelo ARIMA
                        modelo = ARIMA(serie_historica, order=(self.p, self.d, self.q))
                        fitted_model = modelo.fit()
                        
                        # Obtener cliente
                        cliente = row.iloc[0] if len(row) > 0 else None
                        
                        if cliente:
                            # Proyectar para cada periodo
                            if periodo_proyeccion is not None and len(periodo_proyeccion) > 0:
                                for año, mes in periodo_proyeccion:
                                    # Verificar si debe proyectar en este mes
                                    if self.debe_proyectar_en_mes(fecha_inicio_proyeccion, (año, mes)):
                                        # Forecast próximo período
                                        forecast = fitted_model.forecast(steps=1)[0]
                                        
                                        if forecast > 0:
                                            col_name = f'forecast_{año}_{mes:02d}'
                                            df_resultado.loc[index, col_name] = forecast
                                            contador += 1
                            else:
                                # Proyección simple
                                forecast = fitted_model.forecast(steps=1)[0]
                                
                                if forecast > 0:
                                    df_resultado.loc[index, 'forecast'] = forecast
                                    contador += 1
                    
                    except Exception:
                        continue
            
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': contador,
                'modelo': 'ARIMA',
                'parametros': {'p': self.p, 'd': self.d, 'q': self.q},
                'precisao': self.calcular_precisao(df_resultado)
            }
            
        except Exception as e:
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': 0,
                'modelo': 'ARIMA',
                'error': str(e),
                'precisao': 0
            }