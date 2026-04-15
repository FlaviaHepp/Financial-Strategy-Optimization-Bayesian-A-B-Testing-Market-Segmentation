"""Explicación completa del conjunto de datos
Título: Conjunto de datos para pruebas A/B en el mercado financiero
Subtítulo: Analizar y optimizar estrategias de mercado mediante experimentos

Este conjunto de datos está diseñado para facilitar las pruebas A/B en el contexto del mercado financiero. Permite a los usuarios simular, analizar y optimizar estrategias de mercado comparando el rendimiento de los grupos de control y de tratamiento en diversos escenarios de negociación.

Características principales:
Grupos de control y de tratamiento:

El grupo de control representa el comportamiento de mercado de referencia.
El grupo de tratamiento refleja el impacto de una estrategia comercial específica o una intervención en el mercado.
Variables de mercado:

Incluye características como fluctuaciones de precios, volumen de negociación, sentimiento del mercado e indicadores externos.
Métricas de resultados:

Las métricas incluyen la rentabilidad, las tasas de éxito y otros indicadores de éxito para evaluar las estrategias.
Versatilidad:

Ideal para experimentos como la prueba de estrategias de negociación algorítmica, el análisis del sentimiento del mercado y los estudios de finanzas conductuales.
Casos de uso:
Evaluar la eficacia de los nuevos algoritmos de negociación.
Comprender el impacto de la volatilidad del mercado en los resultados de las operaciones bursátiles.
Realización de simulaciones con fines educativos y de investigación en finanzas.
Preguntas sobre pruebas A/B
Nivel principiante:
¿Cuáles son las principales diferencias entre el grupo de control y el grupo de tratamiento?
¿Cuál es el beneficio promedio del grupo de control en comparación con el grupo de tratamiento?
Crea un gráfico de barras que compare el rendimiento de los dos grupos.
Nivel intermedio:
Realizar una prueba A/B básica para determinar si la estrategia del grupo de tratamiento genera beneficios significativamente mayores que la del grupo de control.
Calcula la tasa de conversión (por ejemplo, operaciones rentables frente al total de operaciones) para ambos grupos.
Utilice la visualización para explorar cómo difiere una métrica clave (por ejemplo, el volumen) entre los dos grupos a lo largo del tiempo.
Nivel avanzado:
Realice una prueba de hipótesis estadística (por ejemplo, una prueba t) para evaluar si el rendimiento del grupo de tratamiento es significativamente diferente al del grupo de control.
Analizar el impacto de las variables externas (por ejemplo, el sentimiento del mercado o la volatilidad) en los resultados de la prueba A/B.
Desarrollar un modelo de regresión logística para predecir la probabilidad de éxito de una estrategia basándose en las características del conjunto de datos.
Nivel experto:
Diseñar una prueba A/B multivariable para evaluar los efectos combinados de dos o más intervenciones sobre el desempeño del mercado.
Aplique pruebas A/B bayesianas para evaluar la efectividad de la estrategia del grupo de tratamiento bajo diferentes niveles de confianza.
Utilice técnicas de aprendizaje automático (por ejemplo, agrupamiento o aprendizaje profundo) para identificar subgrupos dentro del conjunto de datos donde el tratamiento tenga el impacto más significativo.
"""

# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import seaborn as sns
sns.set_theme(style="darkgrid")
import logging
import pymc as pm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score

# Configuración de Logging para trazabilidad profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialStrategyAnalyzer:
    """
    Clase de nivel Senior para el análisis y validación de estrategias financieras.
    Integra Segmentación, Predicción de Rentabilidad y Pruebas A/B Bayesianas.
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.scaler = StandardScaler()
        logging.info("Analyzer inicializado con un dataset de %d registros.", len(self.df))

    def segment_market(self, features, n_clusters=3):
        """
        Aplica KMeans para identificar regímenes de mercado.
        """
        logging.info("Iniciando segmentación de mercado (K-Means)...")
        try:
            x_scaled = self.scaler.fit_transform(self.df[features])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df['market_segment'] = kmeans.fit_predict(x_scaled)
            logging.info("Segmentación completada exitosamente.")
            return self.df
        except Exception as e:
            logging.error(f"Error en la segmentación: {e}")
            return None

    def predict_profitability(self, features, target='is_profit'):
        """
        Entrena un modelo de Regresión Logística para predecir el éxito de una operación.
        """
        logging.info("Entrenando modelo predictivo de rentabilidad...")
        x = self.df[features]
        y = self.df[target]
        
        x_scaled = self.scaler.fit_transform(x)
        model = LogisticRegression(class_weight='balanced')
        model.fit(x_scaled, y)
        
        y_pred = model.predict(x_scaled)
        auc = roc_auc_score(y, model.predict_proba(x_scaled)[:, 1])
        
        logging.info(f"Modelo entrenado. ROC AUC Score: {auc:.4f}")
        print("\n--- Reporte de Clasificación ---")
        print(classification_report(y, y_pred))
        return model

    def run_bayesian_ab_test(self, control_col, treatment_col):
        """
        Ejecuta un Test A/B Bayesiano usando PyMC para comparar rendimientos.
        """
        logging.info("Iniciando Muestreo Bayesiano (MCMC)...")
        
        # Extracción de datos para los grupos
        group_a = self.df[self.df['group'] == 'control'][control_col].values
        group_b = self.df[self.df['group'] == 'treatment'][treatment_col].values

        with pm.Model() as ab_model:
            # Priors no informativos para las medias
            mu_a = pm.Normal('mu_A', mu=group_a.mean(), sigma=group_a.std() * 2)
            mu_b = pm.Normal('mu_B', mu=group_b.mean(), sigma=group_b.std() * 2)
            
            # Verosimilitud (Likelihood)
            pm.Normal('obs_A', mu=mu_a, sigma=group_a.std(), observed=group_a)
            pm.Normal('obs_B', mu=mu_b, sigma=group_b.std(), observed=group_b)
            
            # Determinístico: Diferencia de medias (Lift)
            diff = pm.Deterministic('delta', mu_b - mu_a)
            rel_lift = pm.Deterministic('rel_lift', (mu_b - mu_a) / mu_a)
            
            # Muestreo (Protegido para multiprocessing)
            trace = pm.sample(draws=2000, tune=1000, chains=2, cores=2, random_seed=42)
            
        logging.info("Muestreo Bayesiano completado.")
        return trace

    plt.rcParams["figure.facecolor"] = "black"
    plt.rcParams["axes.facecolor"] = "black"
    plt.rcParams["savefig.facecolor"] = "black"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["axes.edgecolor"] = "white"


    def plot_insights(self, trace):
        """Genera visualizaciones clave para la toma de decisiones."""
        pm.plot_posterior(trace, var_names=['delta', 'rel_lift'], ref_val=0)
        plt.title("Distribución Posterior de la Diferencia de Rendimiento\n", fontweight='bold', fontsize=16)
        plt.show()

# ==============================================================================
# PUNTO DE ENTRADA (ESTO CORRIGE EL RUNTIME ERROR EN WINDOWS)
# ==============================================================================
if __name__ == '__main__':
    # 1. Carga de datos (Simulada para el ejemplo, reemplaza con tu CSV)
    # df = pd.read_csv('tu_archivo.csv')
    
    # Simulación de datos para que el código sea ejecutable de inmediato
    np.random.seed(42)
    data = {
        'price_change': np.random.normal(0, 1, 1000),
        'volume_change': np.random.normal(0, 1, 1000),
        'return_after': np.random.normal(0.02, 0.05, 1000),
        'is_profit': np.random.choice([0, 1], size=1000),
        'group': np.random.choice(['control', 'treatment'], size=1000),
        'ticker_num': np.random.randint(1, 10, 1000),
        'market_num': np.random.randint(1, 5, 1000)
    }
    df_market = pd.DataFrame(data)

    # 2. Instanciar la herramienta
    analyzer = FinancialStrategyAnalyzer(df_market)

    # 3. Fase de Segmentación
    features_segment = ['price_change', 'volume_change', 'return_after']
    df_segmented = analyzer.segment_market(features_segment)

    # 4. Fase de Predicción
    features_ml = ['price_change', 'volume_change', 'ticker_num', 'market_num']
    model_lr = analyzer.predict_profitability(features_ml)

    # 5. Fase de Validación Bayesiana (A/B Testing)
    # Evaluamos si el grupo treatment tiene un 'return_after' superior al control
    trace_results = analyzer.run_bayesian_ab_test('return_after', 'return_after')

    # 6. Visualización de resultados
    analyzer.plot_insights(trace_results)
    
    print("\n✅ El proceso ha finalizado correctamente.")
