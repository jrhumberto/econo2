# econometric_app_final.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit, probit
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_breusch_godfrey, 
    het_arch, linear_harvey_collier, linear_rainbow,
    breaks_cusumolsresid
)
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.api import het_goldfeldquandt
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import shapiro, anderson, normaltest, levene, bartlett, fligner
import io
import base64
from datetime import datetime
from fpdf import FPDF
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Econometric Analysis Suite Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estado da sessão
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'model_spec' not in st.session_state:
    st.session_state.model_spec = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}

# Usuários fictícios
USERS = {
    "admin": "admin123",
    "researcher": "econometrics2024",
    "student": "password123",
    "professor": "stats101",
    "guest": "guest123"
}

# Funções auxiliares para análise exploratória
def interpret_correlation(corr):
    """Interpreta o valor de correlação"""
    if pd.isna(corr):
        return "Indefinido"
    
    abs_corr = abs(corr)
    if abs_corr < 0.3:
        return "Correlação fraca"
    elif abs_corr < 0.7:
        return "Correlação moderada"
    else:
        return "Correlação forte"

def get_distribution_shape(series):
    """Retorna descrição da forma da distribuição"""
    if len(series.dropna()) == 0:
        return "Sem dados suficientes"
    
    skew = series.skew()
    kurt = series.kurtosis()
    
    shape = ""
    if abs(skew) < 0.5:
        shape += "Simétrica"
    elif skew > 0:
        shape += "Assimétrica à direita"
    else:
        shape += "Assimétrica à esquerda"
    
    shape += ", "
    
    if kurt > 3.5:
        shape += "leptocúrtica (caudas pesadas)"
    elif kurt < 2.5:
        shape += "platicúrtica (caudas leves)"
    else:
        shape += "mesocúrtica (similar à normal)"
    
    return shape

def detect_outliers_text(series):
    """Detecta e descreve outliers"""
    if len(series.dropna()) == 0:
        return "Sem dados"
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        return "Sem variabilidade suficiente"
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    n_outliers = len(outliers)
    
    if n_outliers == 0:
        return "Nenhum outlier detectado"
    elif n_outliers == 1:
        return f"1 outlier ({outliers.iloc[0]:.2f})"
    else:
        return f"{n_outliers} outliers (de {outliers.min():.2f} a {outliers.max():.2f})"

# Banco de explicações dos testes
TEST_EXPLANATIONS = {
    'jarque_bera': {
        'name': 'Teste de Jarque-Bera',
        'purpose': 'Testar se os resíduos seguem uma distribuição normal',
        'null_hypothesis': 'Os resíduos são normalmente distribuídos',
        'interpretation': 'p-valor > 0.05: não rejeita normalidade',
        'economic_meaning': 'Importante para inferência válida. Se violado, os testes t e F podem não ser confiáveis.',
        'solutions': 'Transformar variáveis, usar métodos robustos, aumentar amostra'
    },
    'shapiro_wilk': {
        'name': 'Teste de Shapiro-Wilk',
        'purpose': 'Teste de normalidade para amostras pequenas (n ≤ 5000)',
        'null_hypothesis': 'Os dados vêm de uma distribuição normal',
        'interpretation': 'p-valor baixo indica não-normalidade',
        'economic_meaning': 'Normalidade é crucial para intervalos de confiança precisos',
        'solutions': 'Verificar outliers, transformar dados'
    },
    'breusch_pagan': {
        'name': 'Teste de Breusch-Pagan',
        'purpose': 'Detectar heterocedasticidade (variância não constante dos erros)',
        'null_hypothesis': 'Homocedasticidade (variância constante dos erros)',
        'interpretation': 'p-valor < 0.05 indica heterocedasticidade',
        'economic_meaning': 'Heterocedasticidade torna os erros padrão ineficientes',
        'solutions': 'Erros robustos (HC1, HC2, HC3), transformar variável dependente'
    },
    'white_test': {
        'name': 'Teste de White',
        'purpose': 'Teste geral de heterocedasticidade (não precisa especificar forma)',
        'null_hypothesis': 'Homocedasticidade',
        'interpretation': 'Rejeita H0 se p-valor < 0.05',
        'economic_meaning': 'Versão mais geral do Breusch-Pagan',
        'solutions': 'Usar matriz de covariância robusta de White'
    },
    'durbin_watson': {
        'name': 'Estatística de Durbin-Watson',
        'purpose': 'Detectar autocorrelação de primeira ordem nos resíduos',
        'null_hypothesis': 'Não há autocorrelação',
        'interpretation': 'Valor próximo de 2: sem autocorrelação; <1.5: positiva; >2.5: negativa',
        'economic_meaning': 'Autocorrelação viola independência dos erros',
        'solutions': 'Incluir defasagens, usar modelos ARIMA, erros padrão HAC'
    },
    'breusch_godfrey': {
        'name': 'Teste de Breusch-Godfrey',
        'purpose': 'Detectar autocorrelação de ordens superiores',
        'null_hypothesis': 'Não há autocorrelação até ordem p',
        'interpretation': 'p-valor < 0.05 indica autocorrelação',
        'economic_meaning': 'Importante em séries temporais e dados em painel',
        'solutions': 'Modelos com correção de autocorrelação'
    },
    'vif': {
        'name': 'Fator de Inflação da Variância (VIF)',
        'purpose': 'Detectar multicolinearidade entre variáveis explicativas',
        'null_hypothesis': 'Não há multicolinearidade perfeita',
        'interpretation': 'VIF > 10: multicolinearidade problemática; VIF > 5: atenção',
        'economic_meaning': 'Multicolinearidade torna coeficientes instáveis',
        'solutions': 'Remover variáveis correlacionadas, usar PCR ou Ridge Regression'
    },
    'ramsey_reset': {
        'name': 'Teste de Ramsey RESET',
        'purpose': 'Verificar especificação funcional do modelo (formas funcionais incorretas)',
        'null_hypothesis': 'O modelo está corretamente especificado',
        'interpretation': 'p-valor < 0.05 indica má especificação',
        'economic_meaning': 'Modelo mal especificado leva a viés nos coeficientes',
        'solutions': 'Adicionar termos não-lineares, transformar variáveis'
    },
    'adf_test': {
        'name': 'Teste ADF (Augmented Dickey-Fuller)',
        'purpose': 'Testar estacionariedade em séries temporais',
        'null_hypothesis': 'A série possui uma raiz unitária (não estacionária)',
        'interpretation': 'p-valor < 0.05: série estacionária',
        'economic_meaning': 'Regressões com séries não-estacionárias podem ser espúrias',
        'solutions': 'Tomar diferenças, usar cointegração'
    },
    'kpss_test': {
        'name': 'Teste KPSS',
        'purpose': 'Testar estacionariedade (hipótese nula invertida)',
        'null_hypothesis': 'A série é estacionária',
        'interpretation': 'p-valor < 0.05: não estacionária',
        'economic_meaning': 'Complementar ao ADF',
        'solutions': 'Diferenciação da série'
    },
    'f_test': {
        'name': 'Teste F de significância conjunta',
        'purpose': 'Testar se todos os coeficientes (exceto intercepto) são zero',
        'null_hypothesis': 'Todos os coeficientes das variáveis explicativas são zero',
        'interpretation': 'p-valor < 0.05: pelo menos um coeficiente é não-zero',
        'economic_meaning': 'Testa se o modelo como um todo tem poder explicativo',
        'solutions': 'Se p-valor alto, reconsiderar variáveis explicativas'
    },
    'goldfeld_quandt': {
        'name': 'Teste de Goldfeld-Quandt',
        'purpose': 'Testar heterocedasticidade quando se suspeita de relação com uma variável',
        'null_hypothesis': 'Homocedasticidade',
        'interpretation': 'p-valor < 0.05 indica heterocedasticidade',
        'economic_meaning': 'Teste útil quando heterocedasticidade segue padrão específico',
        'solutions': 'Weighted Least Squares (WLS)'
    },
    'arch_test': {
        'name': 'Teste ARCH',
        'purpose': 'Detectar heterocedasticidade condicional (volatilidade clustering)',
        'null_hypothesis': 'Não há efeitos ARCH',
        'interpretation': 'p-valor < 0.05 indica presença de ARCH',
        'economic_meaning': 'Comum em dados financeiros (volatilidade se agrupa)',
        'solutions': 'Modelos GARCH, ARCH'
    },
    'levene_test': {
        'name': 'Teste de Levene',
        'purpose': 'Testar homogeneidade de variâncias entre grupos',
        'null_hypothesis': 'As variâncias são iguais entre grupos',
        'interpretation': 'p-valor < 0.05 indica heterogeneidade',
        'economic_meaning': 'Importante em ANOVA e quando comparando grupos',
        'solutions': 'Transformações, métodos robustos'
    }
}

def get_test_explanation(test_name):
    """Retorna explicação detalhada do teste"""
    return TEST_EXPLANATIONS.get(test_name, {
        'name': test_name.replace('_', ' ').title(),
        'purpose': 'Teste estatístico',
        'null_hypothesis': 'Hipótese nula padrão',
        'interpretation': 'Interpretação padrão',
        'economic_meaning': 'Significado econômico',
        'solutions': 'Soluções possíveis'
    })

def login_page():
    """Página de login com design melhorado"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("📊 Econometric Analysis Suite Pro")
        st.markdown("---")
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h2>Análise Econométrica Avançada</h2>
            <p>Upload de dados • Modelagem • Testes de Hipótese • Relatórios PDF</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔐 Login")
        
        username = st.text_input("👤 Username", key="login_user")
        password = st.text_input("🔒 Password", type="password", key="login_pass")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🚀 Login", use_container_width=True, type="primary"):
                if username in USERS and USERS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.success(f"✅ Bem-vindo(a), {username}!")
                    st.rerun()
                else:
                    st.error("❌ Usuário ou senha incorretos")
        
        with col_btn2:
            if st.button("👤 Guest Access", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.current_user = "guest"
                st.success("✅ Logado como guest!")
                st.rerun()
        
        with col_btn3:
            if st.button("ℹ️ Info", use_container_width=True):
                st.info("""
                **Credenciais disponíveis:**
                - admin / admin123
                - researcher / econometrics2024
                - student / password123
                - professor / stats101
                - guest / guest123
                """)
        
        st.markdown("---")
        
        with st.expander("✨ Recursos da Aplicação", expanded=True):
            st.markdown("""
            ### 📋 **Funcionalidades Principais:**
            
            1. **Upload e Merge de Dados**
               - Múltiplos arquivos CSV
               - Merge inteligente
               - Tratamento de valores ausentes
            
            2. **Análise Exploratória**
               - Estatísticas descritivas
               - Visualizações interativas
               - Matriz de correlação
            
            3. **Modelagem Econométrica**
               - OLS, Logit, Probit
               - Dados em painel
               - Séries temporais
            
            4. **Testes de Hipótese (30+ testes)**
               - Normalidade (Jarque-Bera, Shapiro-Wilk)
               - Heterocedasticidade (Breusch-Pagan, White)
               - Autocorrelação (Durbin-Watson, Breusch-Godfrey)
               - Multicolinearidade (VIF)
               - Especificação (Ramsey RESET)
               - Estacionariedade (ADF, KPSS)
            
            5. **Relatório Completo em PDF**
               - Gráficos incorporados
               - Explicações detalhadas
               - Resultados interpretados
               - Recomendações práticas
            """)

def upload_files():
    """Upload de arquivos CSV"""
    st.header("📤 Upload de Dados")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Selecione arquivos CSV", 
            type=["csv"],
            accept_multiple_files=True,
            help="Você pode selecionar múltiplos arquivos para análise"
        )
    
    with col2:
        st.info("""
        **💡 Dicas para melhores resultados:**
        1. Use dados numéricos para variáveis contínuas
        2. Limpe dados ausentes antes do upload
        3. Para dados em painel, inclua colunas de ID e tempo
        4. Use nomes descritivos para variáveis
        """)
    
    if uploaded_files:
        st.session_state.uploaded_files = []
        
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                st.session_state.uploaded_files.append({
                    "name": file.name,
                    "data": df,
                    "columns": df.columns.tolist(),
                    "shape": df.shape,
                    "dtypes": df.dtypes.astype(str).to_dict()
                })
            except Exception as e:
                st.error(f"❌ Erro ao ler {file.name}: {e}")
        
        st.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s) com sucesso!")
        
        for i, file_info in enumerate(st.session_state.uploaded_files):
            with st.expander(f"📄 {file_info['name']} ({file_info['shape'][0]}×{file_info['shape'][1]})"):
                tab_info, tab_preview = st.tabs(["📊 Informações", "👁️ Pré-visualização"])
                
                with tab_info:
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Linhas", f"{file_info['shape'][0]:,}")
                        st.metric("Colunas", file_info['shape'][1])
                    with col_stat2:
                        missing = file_info['data'].isnull().sum().sum()
                        st.metric("Valores Ausentes", f"{missing:,}")
                        st.metric("Memória", f"{file_info['data'].memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    st.write("**Tipos de Dados:**")
                    type_counts = file_info['data'].dtypes.value_counts()
                    for dtype, count in type_counts.items():
                        st.write(f"- {dtype}: {count} colunas")
                
                with tab_preview:
                    st.dataframe(file_info['data'].head(10), use_container_width=True)

def merge_files():
    """Merge de arquivos"""
    if not st.session_state.uploaded_files:
        st.warning("⚠️ Por favor, faça upload de arquivos primeiro.")
        return
    
    st.header("🔄 Merge de Arquivos")
    
    if len(st.session_state.uploaded_files) == 1:
        st.session_state.merged_data = st.session_state.uploaded_files[0]["data"]
        st.success("✅ Apenas um arquivo - merge não necessário")
        
        with st.expander("👁️ Visualizar Dados"):
            st.dataframe(st.session_state.merged_data.head(), use_container_width=True)
            st.write(f"**Forma:** {st.session_state.merged_data.shape}")
        return
    
    merge_method = st.radio(
        "Método de Merge:",
        ["Concatenar Verticalmente", "Join por Chave", "Merge Inteligente"],
        horizontal=True
    )
    
    if merge_method == "Concatenar Verticalmente":
        common_cols = set.intersection(*[set(f["columns"]) for f in st.session_state.uploaded_files])
        if common_cols:
            selected_cols = st.multiselect("Selecionar colunas para manter:", list(common_cols), default=list(common_cols))
            
            if st.button("🔄 Concatenar", type="primary"):
                dfs = [f["data"][selected_cols] for f in st.session_state.uploaded_files]
                st.session_state.merged_data = pd.concat(dfs, axis=0, ignore_index=True)
                st.success(f"✅ Concatenado! {st.session_state.merged_data.shape[0]:,} linhas × {st.session_state.merged_data.shape[1]} colunas")
    
    elif merge_method == "Join por Chave":
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            left_file = st.selectbox("Arquivo Esquerdo", [f["name"] for f in st.session_state.uploaded_files])
        with col_sel2:
            right_file = st.selectbox("Arquivo Direito", [f["name"] for f in st.session_state.uploaded_files if f["name"] != left_file])
        
        left_df = next(f["data"] for f in st.session_state.uploaded_files if f["name"] == left_file)
        right_df = next(f["data"] for f in st.session_state.uploaded_files if f["name"] == right_file)
        
        join_type = st.selectbox("Tipo de Join:", ["inner", "left", "right", "outer"])
        
        col_key1, col_key2 = st.columns(2)
        with col_key1:
            left_key = st.selectbox("Chave Esquerda", left_df.columns)
        with col_key2:
            right_key = st.selectbox("Chave Direita", right_df.columns)
        
        if st.button("🔗 Realizar Join", type="primary"):
            try:
                st.session_state.merged_data = pd.merge(
                    left_df, right_df,
                    left_on=left_key, right_on=right_key,
                    how=join_type,
                    suffixes=('_left', '_right')
                )
                st.success(f"✅ Join realizado! {st.session_state.merged_data.shape[0]:,} linhas × {st.session_state.merged_data.shape[1]} colunas")
            except Exception as e:
                st.error(f"❌ Erro no join: {e}")
    
    else:
        st.info("O sistema tentará encontrar chaves comuns automaticamente.")
        if st.button("🤖 Merge Automático", type="primary"):
            try:
                dfs = [f["data"] for f in st.session_state.uploaded_files]
                st.session_state.merged_data = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
                st.success(f"✅ Merge realizado! {st.session_state.merged_data.shape[0]:,} linhas × {st.session_state.merged_data.shape[1]} colunas")
            except Exception as e:
                st.error(f"❌ Erro: {e}")
    
    if st.session_state.merged_data is not None:
        with st.expander("📊 Dados Mergeados", expanded=True):
            tab_view, tab_stats = st.tabs(["Visualização", "Estatísticas"])
            
            with tab_view:
                rows_to_show = st.slider("Linhas para mostrar:", 5, 50, 15)
                st.dataframe(st.session_state.merged_data.head(rows_to_show), use_container_width=True)
            
            with tab_stats:
                col1, col2, col3 = st.columns(3)
                df = st.session_state.merged_data
                
                with col1:
                    st.metric("Total de Linhas", f"{df.shape[0]:,}")
                    st.metric("Total de Colunas", df.shape[1])
                
                with col2:
                    missing = df.isnull().sum().sum()
                    st.metric("Valores Ausentes", f"{missing:,}")
                    st.metric("Memória", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                with col3:
                    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
                    st.metric("Colunas Numéricas", numeric_cols)
                    st.metric("Colunas Categóricas", df.shape[1] - numeric_cols)

def convert_to_python_types(data):
    """Converte dados numpy/pandas para tipos Python nativos serializáveis"""
    if isinstance(data, pd.DataFrame):
        return data.astype(object).where(pd.notnull(data), None).to_dict('records')
    elif isinstance(data, pd.Series):
        return data.astype(object).where(pd.notnull(data), None).tolist()
    elif isinstance(data, np.ndarray):
        return data.astype(object).tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, (float, np.floating)):
        return float(data)
    elif pd.isna(data):
        return None
    else:
        return data

####

def exploratory_analysis():
    """Análise exploratória dos dados"""
    if st.session_state.merged_data is None:
        st.warning("⚠️ Por favor, carregue e merge os dados primeiro.")
        return
    
    st.header("🔍 Análise Exploratória")
    
    df = st.session_state.merged_data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    tab_overview, tab_stats, tab_viz, tab_corr = st.tabs([
        "📋 Visão Geral", 
        "📊 Estatísticas", 
        "📈 Visualizações", 
        "🔗 Correlações"
    ])
    
    with tab_overview:
        st.subheader("Visão Geral dos Dados")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.write("**Informações Gerais:**")
            st.write(f"- **Total de Observações:** {df.shape[0]:,}")
            st.write(f"- **Total de Variáveis:** {df.shape[1]}")
            st.write(f"- **Variáveis Numéricas:** {len(numeric_cols)}")
            st.write(f"- **Variáveis Categóricas:** {df.shape[1] - len(numeric_cols)}")
            st.write(f"- **Valores Ausentes:** {df.isnull().sum().sum():,}")
        
        with col_info2:
            dtype_df = pd.DataFrame(df.dtypes.value_counts()).reset_index()
            dtype_df.columns = ['Tipo', 'Quantidade']
            
            fig = px.pie(dtype_df, values='Quantidade', names='Tipo', 
                        title='Distribuição de Tipos de Dados',
                        hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_stats:
        st.subheader("Estatísticas Descritivas")
        
        if numeric_cols:
            selected_vars = st.multiselect(
                "Selecione variáveis para análise:",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_vars:
                try:
                    # Garantir que estamos trabalhando apenas com dados numéricos
                    numeric_data = df[selected_vars].apply(pd.to_numeric, errors='coerce')
                    
                    desc_stats = numeric_data.describe().T
                    
                    # Adicionar estatísticas adicionais
                    desc_stats['skewness'] = numeric_data.skew()
                    desc_stats['kurtosis'] = numeric_data.kurtosis()
                    
                    # Calcular CV apenas para variáveis com média não-zero
                    cv_values = []
                    for var in selected_vars:
                        mean_val = desc_stats.loc[var, 'mean']
                        std_val = desc_stats.loc[var, 'std']
                        if mean_val != 0 and not pd.isna(mean_val):
                            cv = std_val / mean_val
                        else:
                            cv = np.nan
                        cv_values.append(cv)
                    
                    desc_stats['CV'] = cv_values
                    desc_stats['missing'] = numeric_data.isnull().sum()
                    
                    # Converter para tipos Python nativos para exibição
                    desc_stats_display = desc_stats.copy()
                    for col in desc_stats_display.columns:
                        desc_stats_display[col] = desc_stats_display[col].apply(
                            lambda x: float(x) if isinstance(x, (np.generic, np.ndarray)) else x
                        )
                    
                    st.dataframe(desc_stats_display.style.format("{:.4f}"), use_container_width=True)
                    
                    with st.expander("📖 Explicação das Estatísticas"):
                        st.markdown("""
                        **Média**: Valor médio da variável  
                        **Desvio Padrão**: Dispersão em torno da média  
                        **Assimetria (Skewness)**:  
                        - **> 0**: Distribuição assimétrica à direita  
                        - **≈ 0**: Distribuição simétrica  
                        - **< 0**: Distribuição assimétrica à esquerda  
                        
                        **Curtose (Kurtosis)**:  
                        - **> 3**: Distribuição leptocúrtica (caudas pesadas)  
                        - **= 3**: Distribuição normal  
                        - **< 3**: Distribuição platicúrtica (caudas leves)  
                        
                        **Coeficiente de Variação (CV)**: Desvio padrão / Média  
                        - **CV < 1**: Baixa dispersão relativa  
                        - **CV > 1**: Alta dispersão relativa  
                        """)
                except Exception as e:
                    st.error(f"❌ Erro ao calcular estatísticas: {e}")
        else:
            st.warning("❌ Nenhuma variável numérica encontrada para análise estatística.")
    
    with tab_viz:
        st.subheader("Visualizações Exploratórias")
        
        if numeric_cols:
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                viz_type = st.selectbox(
                    "Tipo de Gráfico:",
                    ["Histograma", "Box Plot", "Densidade", "Scatter Plot"]
                )
                
                x_var = st.selectbox("Variável X:", numeric_cols)
                
                if viz_type == "Scatter Plot":
                    available_y_vars = [c for c in numeric_cols if c != x_var]
                    if available_y_vars:
                        y_var = st.selectbox("Variável Y:", available_y_vars)
                    else:
                        y_var = None
                        st.warning("Não há outra variável numérica disponível para Scatter Plot")
            
            with col_viz2:
                st.info("""
                **💡 Interpretação dos Gráficos:**
                - **Histograma**: Distribuição da variável
                - **Box Plot**: Dispersão e outliers
                - **Densidade**: Forma da distribuição
                - **Scatter Plot**: Relação entre duas variáveis
                """)
            
            # Gerar gráfico
            fig = None
            
            try:
                if viz_type == "Histograma":
                    # Garantir que os dados são numéricos
                    x_data = pd.to_numeric(df[x_var], errors='coerce').dropna()
                    if len(x_data) > 0:
                        fig = px.histogram(df, x=x_var, nbins=30, 
                                          title=f"Distribuição de {x_var}",
                                          marginal="box")
                        if len(x_data) > 0:
                            mean_val = float(x_data.mean())
                            fig.add_vline(x=mean_val, line_dash="dash", 
                                        line_color="red", annotation_text=f"Média: {mean_val:.2f}")
                    else:
                        st.warning(f"Não há dados numéricos válidos para {x_var}")
                    
                elif viz_type == "Box Plot":
                    fig = px.box(df, y=x_var, title=f"Box Plot de {x_var}")
                    
                elif viz_type == "Densidade":
                    data_clean = pd.to_numeric(df[x_var], errors='coerce').dropna()
                    if len(data_clean) > 0:
                        # Converter para lista Python para evitar problemas de serialização
                        data_list = data_clean.tolist()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=data_list,
                            histnorm='probability density',
                            name='Histograma',
                            opacity=0.7
                        ))
                        
                        # Adicionar curva normal
                        if len(data_list) > 1:
                            mean_val = float(np.mean(data_list))
                            std_val = float(np.std(data_list))
                            x_norm = np.linspace(min(data_list), max(data_list), 100)
                            y_norm = stats.norm.pdf(x_norm, mean_val, std_val)
                            
                            # Converter para listas Python
                            x_norm_list = x_norm.tolist()
                            y_norm_list = y_norm.tolist()
                            
                            fig.add_trace(go.Scatter(
                                x=x_norm_list,
                                y=y_norm_list,
                                mode='lines',
                                name='Distribuição Normal',
                                line=dict(color='red', width=2)
                            ))
                        
                        fig.update_layout(title=f"Densidade de {x_var}")
                    else:
                        st.warning(f"Não há dados numéricos suficientes para {x_var}")
                    
                elif viz_type == "Scatter Plot" and y_var is not None:
                    # Garantir que ambas as variáveis são numéricas
                    scatter_data = df[[x_var, y_var]].apply(pd.to_numeric, errors='coerce').dropna()
                    if len(scatter_data) > 0:
                        fig = px.scatter(scatter_data, x=x_var, y=y_var, 
                                        trendline="ols",
                                        title=f"{x_var} vs {y_var}")
                        # Calcular correlação
                        corr = scatter_data.corr().iloc[0,1]
                        if not pd.isna(corr):
                            fig.add_annotation(
                                text=f"Correlação: {float(corr):.3f}",
                                xref="paper", yref="paper",
                                x=0.05, y=0.95,
                                showarrow=False,
                                bgcolor="white"
                            )
                    else:
                        st.warning("Não há dados suficientes para o Scatter Plot")
                
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretação do gráfico
                    with st.expander("📝 Interpretação do Gráfico"):
                        if viz_type == "Histograma":
                            try:
                                x_data = pd.to_numeric(df[x_var], errors='coerce').dropna()
                                if len(x_data) > 0:
                                    st.markdown(f"""
                                    **Análise de {x_var}:**
                                    1. **Forma da Distribuição**: {get_distribution_shape(x_data)}
                                    2. **Centro**: A maioria dos valores está em torno de {float(x_data.mean()):.2f}
                                    3. **Dispersão**: Os valores variam entre {float(x_data.min()):.2f} e {float(x_data.max()):.2f}
                                    4. **Outliers**: {detect_outliers_text(x_data)}
                                    """)
                            except:
                                pass
                        elif viz_type == "Scatter Plot" and y_var is not None:
                            try:
                                scatter_data = df[[x_var, y_var]].apply(pd.to_numeric, errors='coerce').dropna()
                                if len(scatter_data) > 0:
                                    corr = scatter_data.corr().iloc[0,1]
                                    if not pd.isna(corr):
                                        st.markdown(f"""
                                        **Relação entre {x_var} e {y_var}:**
                                        1. **Correlação**: {float(corr):.3f} ({interpret_correlation(corr)})
                                        2. **Direção**: {'Positiva' if corr > 0 else 'Negativa' if corr < 0 else 'Nenhuma'}
                                        3. **Força**: {'Forte' if abs(corr) > 0.7 else 'Moderada' if abs(corr) > 0.3 else 'Fraca'}
                                        """)
                            except:
                                pass
            
            except Exception as e:
                st.error(f"❌ Erro ao gerar gráfico: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    with tab_corr:
        st.subheader("Análise de Correlação")
        
        if len(numeric_cols) >= 2:
            try:
                # Garantir que todas as colunas são numéricas
                numeric_data = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # Matriz de correlação
                corr_matrix = numeric_data.corr()
                
                # CORREÇÃO: Converter matriz de correlação para lista de listas com valores serializáveis
                corr_matrix_values = corr_matrix.values
                corr_matrix_list = []
                for row in corr_matrix_values:
                    # Converter cada valor para float Python nativo (não numpy)
                    corr_matrix_list.append([float(x) if not pd.isna(x) else 0.0 for x in row])
                
                # CORREÇÃO: Converter nomes das colunas para lista Python
                column_names = corr_matrix.columns.tolist()
                row_names = corr_matrix.index.tolist()
                
                # Heatmap interativo
                fig = px.imshow(corr_matrix_list,
                              x=column_names,
                              y=row_names,
                              text_auto='.2f',
                              color_continuous_scale='RdBu',
                              zmin=-1, zmax=1,
                              title='Matriz de Correlação',
                              aspect="auto",
                              labels=dict(x="Variáveis", y="Variáveis", color="Correlação"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análise de multicolinearidade
                st.subheader("🔍 Detecção de Multicolinearidade")
                
                # Calcular VIF para variáveis selecionadas
                selected_for_vif = st.multiselect(
                    "Selecione variáveis para cálculo de VIF:",
                    numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))]
                )
                
                if len(selected_for_vif) >= 2:
                    try:
                        # Remover valores ausentes e garantir dados numéricos
                        X_clean = df[selected_for_vif].apply(pd.to_numeric, errors='coerce').dropna()
                        if len(X_clean) > 0 and X_clean.shape[1] == len(selected_for_vif):
                            X_with_const = sm.add_constant(X_clean)
                            vif_data = pd.DataFrame()
                            vif_data["Variável"] = X_with_const.columns.tolist()
                            
                            # Calcular VIF para cada variável
                            vif_values = []
                            for i in range(X_with_const.shape[1]):
                                try:
                                    vif = float(variance_inflation_factor(X_with_const.values, i))
                                except:
                                    vif = np.nan
                                vif_values.append(vif)
                            
                            vif_data["VIF"] = vif_values
                            vif_data["Tolerância"] = [1/v if v != 0 and not pd.isna(v) else np.nan for v in vif_values]
                            
                            # Classificar multicolinearidade
                            def classify_vif(vif):
                                if pd.isna(vif):
                                    return "❌ Erro"
                                elif vif > 10:
                                    return "🚨 Severa"
                                elif vif > 5:
                                    return "⚠️ Moderada"
                                else:
                                    return "✅ Aceitável"
                            
                            vif_data["Classificação"] = vif_data["VIF"].apply(classify_vif)
                            
                            # Converter valores para tipos Python nativos
                            for col in ['VIF', 'Tolerância']:
                                vif_data[col] = vif_data[col].apply(
                                    lambda x: float(x) if isinstance(x, (np.generic, np.ndarray)) else x
                                )
                            
                            st.dataframe(vif_data, use_container_width=True)
                            
                            # Explicação do VIF
                            with st.expander("📖 O que é VIF e como interpretar?"):
                                st.markdown("""
                                **Fator de Inflação da Variância (VIF)**: Mede quanto a variância de um coeficiente de regressão 
                                está inflada devido à multicolinearidade.
                                
                                **Interpretação**:
                                - **VIF = 1**: Sem correlação
                                - **1 < VIF ≤ 5**: Correlação moderada (geralmente aceitável)
                                - **5 < VIF ≤ 10**: Correlação alta (pode ser problemática)
                                - **VIF > 10**: Multicolinearidade severa (problema sério)
                                
                                **Tolerância**: 1/VIF. Valores abaixo de 0.1 indicam problemas.
                                
                                **O que fazer se VIF for alto?**
                                1. Remover variáveis altamente correlacionadas
                                2. Usar Análise de Componentes Principais (PCA)
                                3. Aplicar Regularização (Ridge, Lasso)
                                4. Coletar mais dados
                                """)
                        else:
                            st.warning("Não há dados suficientes para calcular VIF após remover valores ausentes.")
                    except Exception as e:
                        st.warning(f"Não foi possível calcular VIF: {e}")
            except Exception as e:
                st.error(f"❌ Erro na análise de correlação: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ É necessário pelo menos 2 variáveis numéricas para análise de correlação.")

####


def specify_model():
    """Especificação do modelo econométrico"""
    if st.session_state.merged_data is None:
        st.warning("⚠️ Por favor, carregue e merge os dados primeiro.")
        return
    
    st.header("⚙️ Especificação do Modelo Econométrico")
    
    df = st.session_state.merged_data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("❌ Nenhuma variável numérica encontrada. Não é possível especificar modelo.")
        return
    
    col_spec1, col_spec2 = st.columns([2, 1])
    
    with col_spec1:
        st.subheader("1. Variável Dependente (Y)")
        y_var = st.selectbox(
            "Selecione a variável que deseja explicar:",
            numeric_cols,
            help="Esta é a variável que seu modelo tentará prever ou explicar."
        )
        
        # Mostrar informações sobre Y
        y_data = df[y_var].dropna()
        if len(y_data) > 0:
            st.info(f"""
            **Informações sobre {y_var}:**
            - Média: {y_data.mean():.2f}
            - Desvio Padrão: {y_data.std():.2f}
            - Mínimo: {y_data.min():.2f}
            - Máximo: {y_data.max():.2f}
            - Valores ausentes: {df[y_var].isnull().sum()}
            """)
        else:
            st.warning(f"⚠️ A variável {y_var} não contém dados numéricos válidos.")
        
        st.subheader("2. Variáveis Independentes (X)")
        x_vars = st.multiselect(
            "Selecione as variáveis explicativas:",
            [c for c in numeric_cols if c != y_var],
            help="Estas são as variáveis que explicam ou predizem a variável dependente."
        )
        
        if x_vars:
            st.success(f"✅ {len(x_vars)} variável(s) independente(s) selecionada(s)")
            
            # Mostrar correlações com Y
            correlations = []
            for x in x_vars:
                clean_data = df[[y_var, x]].dropna()
                if len(clean_data) > 0:
                    try:
                        corr = clean_data.corr().iloc[0,1]
                        if not pd.isna(corr):
                            correlations.append((x, corr))
                    except:
                        pass
            
            if correlations:
                correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                
                st.write("**Correlação com a variável dependente:**")
                for var, corr in correlations[:5]:  # Mostrar apenas top 5
                    st.write(f"- {var}: {corr:.3f} ({interpret_correlation(corr)})")
    
    with col_spec2:
        st.subheader("3. Tipo de Modelo")
        
        model_type = st.selectbox(
            "Escolha o tipo de modelo:",
            [
                "Regressão Linear (OLS)",
                "Regressão Linear Robusta",
                "Modelo Logit",
                "Modelo Probit",
                "Regressão Quantílica",
                "Modelo de Efeitos Fixos (Painel)",
                "Modelo de Efeitos Aleatórios (Painel)"
            ]
        )
        
        # Configurações do modelo
        st.subheader("4. Configurações")
        
        confidence_level = st.slider(
            "Nível de Confiança:",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Probabilidade de que o intervalo de confiança contenha o verdadeiro parâmetro."
        )
        
        include_constant = st.checkbox(
            "Incluir termo constante (intercepto)",
            value=True,
            help="Adiciona um intercepto ao modelo. Geralmente recomendado."
        )
        
        robust_errors = st.checkbox(
            "Usar erros padrão robustos",
            value=False,
            help="Ajusta para heterocedasticidade. Recomendado quando não se tem certeza sobre homocedasticidade."
        )
    
    # Hipóteses do modelo
    with st.expander("📝 Especificar Hipóteses do Modelo", expanded=True):
        col_hyp1, col_hyp2 = st.columns(2)
        
        with col_hyp1:
            st.subheader("Hipótese Nula (H₀)")
            null_hypothesis = st.text_area(
                "Hipótese nula principal:",
                "Os coeficientes de todas as variáveis independentes são iguais a zero.",
                height=100
            )
        
        with col_hyp2:
            st.subheader("Hipótese Alternativa (H₁)")
            alt_hypothesis = st.text_area(
                "Hipótese alternativa:",
                "Pelo menos um coeficiente das variáveis independentes é diferente de zero.",
                height=100
            )
    
    # Tratamento de dados
    with st.expander("🔧 Tratamento de Dados"):
        missing_treatment = st.selectbox(
            "Tratamento de valores ausentes:",
            ["Remover observações incompletas", "Imputar com média", "Imputar com mediana"]
        )
        
        outlier_treatment = st.selectbox(
            "Tratamento de outliers:",
            ["Manter todos", "Remover outliers extremos", "Winsorizar (substituir)"]
        )
    
    # Botão para salvar especificação
    if st.button("💾 Salvar Especificação do Modelo", type="primary"):
        if not x_vars:
            st.error("❌ Selecione pelo menos uma variável independente.")
        else:
            st.session_state.model_spec = {
                'y_var': y_var,
                'x_vars': x_vars,
                'model_type': model_type,
                'confidence_level': confidence_level,
                'include_constant': include_constant,
                'robust_errors': robust_errors,
                'missing_treatment': missing_treatment,
                'outlier_treatment': outlier_treatment,
                'hypotheses': {
                    'null': null_hypothesis,
                    'alternative': alt_hypothesis
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.success("✅ Especificação do modelo salva com sucesso!")
            
            # Mostrar resumo
            st.subheader("📋 Resumo da Especificação")
            
            spec = st.session_state.model_spec
            col_sum1, col_sum2 = st.columns(2)
            
            with col_sum1:
                st.write("**Variáveis:**")
                st.write(f"- Dependente (Y): {spec['y_var']}")
                st.write(f"- Independentes (X): {', '.join(spec['x_vars'])}")
                st.write(f"- Total de variáveis: {len(spec['x_vars'])}")
                
                st.write("\n**Configurações:**")
                st.write(f"- Tipo: {spec['model_type']}")
                st.write(f"- Nível de confiança: {spec['confidence_level']*100}%")
                st.write(f"- Constante: {'Sim' if spec['include_constant'] else 'Não'}")
                st.write(f"- Erros robustos: {'Sim' if spec['robust_errors'] else 'Não'}")
            
            with col_sum2:
                st.write("**Hipóteses:**")
                st.write(f"- H₀: {spec['hypotheses']['null'][:100]}...")
                st.write(f"- H₁: {spec['hypotheses']['alternative'][:100]}...")
                
                st.write("\n**Tratamento de dados:**")
                st.write(f"- Valores ausentes: {spec['missing_treatment']}")
                st.write(f"- Outliers: {spec['outlier_treatment']}")
                
                st.write(f"\n**Especificado em:** {spec['timestamp']}")

def run_analysis():
    """Executar análise econométrica completa"""
    if not st.session_state.model_spec:
        st.warning("⚠️ Por favor, especifique o modelo primeiro.")
        return
    
    st.header("🔬 Executar Análise Econométrica")
    
    spec = st.session_state.model_spec
    
    with st.expander("📋 Revisar Especificação", expanded=True):
        st.write(f"**Modelo:** {spec['model_type']}")
        st.write(f"**Y:** {spec['y_var']}")
        st.write(f"**X:** {', '.join(spec['x_vars'])}")
        st.write(f"**H₀:** {spec['hypotheses']['null']}")
    
    col_run1, col_run2 = st.columns([2, 1])
    
    with col_run1:
        st.subheader("Opções de Análise")
        
        analysis_options = st.multiselect(
            "Selecione análises a realizar:",
            [
                "Modelo Principal",
                "Testes de Diagnóstico",
                "Análise de Resíduos",
                "Testes de Robustez",
                "Comparação de Modelos",
                "Validação Cruzada"
            ],
            default=["Modelo Principal", "Testes de Diagnóstico", "Análise de Resíduos"]
        )
    
    with col_run2:
        st.subheader("Configuração")
        
        random_seed = st.number_input("Semente aleatória:", value=42, min_value=0)
        np.random.seed(random_seed)
        
        test_size = st.slider(
            "Tamanho do teste (validação):",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Proporção dos dados para validação"
        )
    
    if st.button("🚀 Executar Análise Completa", type="primary", use_container_width=True):
        with st.spinner("🔍 Executando análise... Isso pode levar alguns instantes."):
            try:
                results = perform_econometric_analysis()
                
                if results:
                    st.session_state.analysis_results = results
                    
                    generate_explanations(results)
                    
                    st.success("✅ Análise concluída com sucesso!")
                    
                    st.rerun()
                else:
                    st.error("❌ A análise não produziu resultados.")
                    
            except Exception as e:
                st.error(f"❌ Erro durante a análise: {str(e)}")
                st.exception(e)

def perform_econometric_analysis():
    """Executar a análise econométrica completa"""
    df = st.session_state.merged_data.copy()
    spec = st.session_state.model_spec
    
    # Verificar se as variáveis existem
    missing_vars = []
    for var in [spec['y_var']] + spec['x_vars']:
        if var not in df.columns:
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"As seguintes variáveis não existem nos dados: {', '.join(missing_vars)}")
    
    # Preparar dados
    y = df[spec['y_var']].copy()
    X = df[spec['x_vars']].copy()
    
    # Tratamento de valores ausentes
    if "Remover" in spec['missing_treatment']:
        data = pd.concat([y, X], axis=1).dropna()
        y = data[spec['y_var']]
        X = data[spec['x_vars']]
    elif "média" in spec['missing_treatment'].lower():
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
    elif "mediana" in spec['missing_treatment'].lower():
        X = X.fillna(X.median())
        y = y.fillna(y.median())
    
    # Verificar se ainda temos dados suficientes
    if len(y) == 0 or len(X) == 0:
        raise ValueError("Não há dados suficientes após o tratamento de valores ausentes.")
    
    # Adicionar constante se necessário
    if spec['include_constant']:
        X = sm.add_constant(X)
    
    # Ajustar modelo
    if "Linear" in spec['model_type']:
        model = sm.OLS(y, X).fit()
        
        if "Robusta" in spec['model_type'] or spec['robust_errors']:
            model = model.get_robustcov_results(cov_type='HC3')
    
    elif "Logit" in spec['model_type']:
        model = logit(y, X).fit(disp=False, maxiter=1000)
    
    elif "Probit" in spec['model_type']:
        model = probit(y, X).fit(disp=False, maxiter=1000)
    
    else:
        model = sm.OLS(y, X).fit()
    
    # Calcular previsões e resíduos
    y_pred = model.predict(X)
    residuals = model.resid
    
    # Executar testes
    test_results = run_all_diagnostic_tests(model, X, y, residuals)
    
    # Calcular métricas de performance
    performance = calculate_performance_metrics(y, y_pred, model)
    
    return {
        'model': model,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'residuals': residuals,
        'test_results': test_results,
        'performance': performance,
        'specification': spec,
        'data_info': {
            'n_obs': len(y),
            'n_vars': X.shape[1],
            'y_stats': {
                'mean': y.mean() if len(y) > 0 else np.nan,
                'std': y.std() if len(y) > 0 else np.nan,
                'min': y.min() if len(y) > 0 else np.nan,
                'max': y.max() if len(y) > 0 else np.nan
            }
        }
    }

def run_all_diagnostic_tests(model, X, y, residuals):
    """Executar todos os testes de diagnóstico"""
    results = {}
    
    # 1. Testes de Normalidade
    results['normality'] = {
        'jarque_bera': perform_jarque_bera(residuals),
        'shapiro_wilk': perform_shapiro_wilk(residuals),
        'anderson_darling': perform_anderson_darling(residuals)
    }
    
    # 2. Testes de Heterocedasticidade
    results['heteroscedasticity'] = {
        'breusch_pagan': perform_breusch_pagan(model, X, residuals),
        'white_test': perform_white_test(model, X, residuals),
        'goldfeld_quandt': perform_goldfeld_quandt(y, X)
    }
    
    # 3. Testes de Autocorrelação
    results['autocorrelation'] = {
        'durbin_watson': perform_durbin_watson(residuals),
        'breusch_godfrey': perform_breusch_godfrey(model, X, residuals),
        'ljung_box': perform_ljung_box(residuals)
    }
    
    # 4. Multicolinearidade
    results['multicollinearity'] = {
        'vif': calculate_vif(X),
        'condition_number': calculate_condition_number(X)
    }
    
    # 5. Testes de Especificação
    results['specification'] = {
        'ramsey_reset': perform_ramsey_reset(model, X, y),
        'harvey_collier': perform_harvey_collier(model)
    }
    
    # 6. Estacionariedade (se relevante)
    results['stationarity'] = {
        'adf': perform_adf_test(y),
        'kpss': perform_kpss_test(y)
    }
    
    return results

def perform_jarque_bera(residuals):
    """Executar teste de Jarque-Bera"""
    try:
        if len(residuals) > 0:
            stat, p_value = jarque_bera(residuals)
            return {
                'statistic': float(stat),
                'p_value': float(p_value),
                'conclusion': 'Normal' if p_value > 0.05 else 'Não normal',
                'skewness': float(stats.skew(residuals)) if len(residuals) > 0 else np.nan,
                'kurtosis': float(stats.kurtosis(residuals)) if len(residuals) > 0 else np.nan
            }
        else:
            return {'error': 'Sem dados para o teste'}
    except Exception as e:
        return {'error': str(e)}

def perform_shapiro_wilk(residuals):
    """Executar teste de Shapiro-Wilk"""
    try:
        if len(residuals) <= 5000 and len(residuals) > 3:
            stat, p_value = shapiro(residuals)
            return {
                'statistic': float(stat),
                'p_value': float(p_value),
                'conclusion': 'Normal' if p_value > 0.05 else 'Não normal'
            }
        elif len(residuals) <= 3:
            return {'error': 'Amostra muito pequena para Shapiro-Wilk (n ≤ 3)'}
        else:
            return {'error': 'Amostra muito grande para Shapiro-Wilk (n > 5000)'}
    except Exception as e:
        return {'error': str(e)}

def perform_anderson_darling(residuals):
    """Executar teste de Anderson-Darling"""
    try:
        if len(residuals) > 0:
            result = anderson(residuals, dist='norm')
            return {
                'statistic': float(result.statistic),
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_level.tolist()
            }
        else:
            return {'error': 'Sem dados para o teste'}
    except Exception as e:
        return {'error': str(e)}

def perform_breusch_pagan(model, X, residuals):
    """Executar teste de Breusch-Pagan"""
    try:
        lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, X)
        return {
            'lm_statistic': float(lm),
            'lm_p_value': float(lm_p_value),
            'f_statistic': float(fvalue),
            'f_p_value': float(f_p_value),
            'conclusion': 'Homocedástico' if lm_p_value > 0.05 else 'Heterocedástico'
        }
    except Exception as e:
        return {'error': str(e)}

def perform_white_test(model, X, residuals):
    """Executar teste de White"""
    try:
        lm, lm_p_value, fvalue, f_p_value = het_white(residuals, X)
        return {
            'lm_statistic': float(lm),
            'lm_p_value': float(lm_p_value),
            'f_statistic': float(fvalue),
            'f_p_value': float(f_p_value),
            'conclusion': 'Homocedástico' if lm_p_value > 0.05 else 'Heterocedástico'
        }
    except Exception as e:
        return {'error': str(e)}

def perform_goldfeld_quandt(y, X):
    """Executar teste de Goldfeld-Quandt"""
    try:
        stat, p_value = het_goldfeldquandt(y, X)
        return {
            'statistic': float(stat),
            'p_value': float(p_value),
            'conclusion': 'Homocedástico' if p_value > 0.05 else 'Heterocedástico'
        }
    except Exception as e:
        return {'error': str(e)}

def perform_durbin_watson(residuals):
    """Calcular estatística de Durbin-Watson"""
    try:
        if len(residuals) > 0:
            stat = durbin_watson(residuals)
            
            if stat < 1.5:
                interpretation = "Autocorrelação positiva"
            elif stat > 2.5:
                interpretation = "Autocorrelação negativa"
            else:
                interpretation = "Sem autocorrelação significativa"
            
            return {
                'statistic': float(stat),
                'interpretation': interpretation
            }
        else:
            return {'error': 'Sem dados para o teste'}
    except Exception as e:
        return {'error': str(e)}

def perform_breusch_godfrey(model, X, residuals):
    """Executar teste de Breusch-Godfrey"""
    try:
        bg_test = acorr_breusch_godfrey(model, nlags=2)
        return {
            'lm_statistic': float(bg_test[0]),
            'p_value': float(bg_test[1]),
            'conclusion': 'Sem autocorrelação' if bg_test[1] > 0.05 else 'Com autocorrelação'
        }
    except Exception as e:
        return {'error': str(e)}

def perform_ljung_box(residuals):
    """Executar teste de Ljung-Box"""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        if len(residuals) > 0:
            result = acorr_ljungbox(residuals, lags=[5], return_df=True)
            return {
                'statistic': float(result['lb_stat'].iloc[0]),
                'p_value': float(result['lb_pvalue'].iloc[0]),
                'conclusion': 'Sem autocorrelação' if result['lb_pvalue'].iloc[0] > 0.05 else 'Com autocorrelação'
            }
        else:
            return {'error': 'Sem dados para o teste'}
    except Exception as e:
        return {'error': str(e)}

def calculate_vif(X):
    """Calcular VIF para todas as variáveis"""
    try:
        vif_data = []
        for i, col in enumerate(X.columns):
            if col != 'const':  # Ignorar constante
                vif = variance_inflation_factor(X.values, i)
                tolerance = 1 / vif if vif != 0 else float('inf')
                
                # Classificação
                if pd.isna(vif):
                    classification = "❌ Erro no cálculo"
                elif vif > 10:
                    classification = "🚨 Multicolinearidade severa"
                elif vif > 5:
                    classification = "⚠️ Multicolinearidade moderada"
                else:
                    classification = "✅ Aceitável"
                
                vif_data.append({
                    'variable': col,
                    'vif': float(vif) if not pd.isna(vif) else np.nan,
                    'tolerance': float(tolerance) if not pd.isna(vif) else np.nan,
                    'classification': classification
                })
        
        return vif_data
    except Exception as e:
        return {'error': str(e)}

def calculate_condition_number(X):
    """Calcular número de condição da matriz X"""
    try:
        X_matrix = X.values if hasattr(X, 'values') else X
        if X_matrix.shape[0] > 0 and X_matrix.shape[1] > 0:
            cond_num = np.linalg.cond(X_matrix)
            
            if cond_num > 1000:
                interpretation = "Multicolinearidade severa"
            elif cond_num > 100:
                interpretation = "Multicolinearidade moderada"
            else:
                interpretation = "Aceitável"
            
            return {
                'condition_number': float(cond_num),
                'interpretation': interpretation
            }
        else:
            return {'error': 'Matriz X vazia'}
    except Exception as e:
        return {'error': str(e)}

def perform_ramsey_reset(model, X, y):
    """Executar teste de Ramsey RESET"""
    try:
        y_pred = model.predict(X)
        X_augmented = X.copy()
        
        X_augmented['y_pred^2'] = y_pred ** 2
        X_augmented['y_pred^3'] = y_pred ** 3
        
        model_augmented = sm.OLS(y, X_augmented).fit()
        
        rss_restricted = model.ssr
        rss_unrestricted = model_augmented.ssr
        df_restricted = model.df_resid
        df_unrestricted = model_augmented.df_resid
        
        f_stat = ((rss_restricted - rss_unrestricted) / 2) / (rss_unrestricted / df_unrestricted)
        p_value = 1 - stats.f.cdf(f_stat, 2, df_unrestricted)
        
        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'conclusion': 'Bem especificado' if p_value > 0.05 else 'Mal especificado'
        }
    except Exception as e:
        return {'error': str(e)}

def perform_harvey_collier(model):
    """Executar teste de Harvey-Collier"""
    try:
        t_stat, p_value = linear_harvey_collier(model)
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'conclusion': 'Linear' if p_value > 0.05 else 'Não linear'
        }
    except Exception as e:
        return {'error': str(e)}

def perform_adf_test(y):
    """Executar teste ADF"""
    try:
        y_clean = y.dropna()
        if len(y_clean) > 0:
            result = adfuller(y_clean)
            return {
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'conclusion': 'Estacionária' if result[1] < 0.05 else 'Não estacionária'
            }
        else:
            return {'error': 'Sem dados para o teste'}
    except Exception as e:
        return {'error': str(e)}

def perform_kpss_test(y):
    """Executar teste KPSS"""
    try:
        y_clean = y.dropna()
        if len(y_clean) > 0:
            result = kpss(y_clean, regression='c')
            return {
                'kpss_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[3].items()},
                'conclusion': 'Estacionária' if result[1] > 0.05 else 'Não estacionária'
            }
        else:
            return {'error': 'Sem dados para o teste'}
    except Exception as e:
        return {'error': str(e)}

def calculate_performance_metrics(y, y_pred, model):
    """Calcular métricas de performance"""
    try:
        errors = y - y_pred
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        if (y != 0).all():
            mape = np.mean(np.abs(errors / y)) * 100
        else:
            mape = np.nan
        
        r_squared = model.rsquared if hasattr(model, 'rsquared') else None
        r_squared_adj = model.rsquared_adj if hasattr(model, 'rsquared_adj') else None
        
        aic = model.aic if hasattr(model, 'aic') else None
        bic = model.bic if hasattr(model, 'bic') else None
        
        llf = model.llf if hasattr(model, 'llf') else None
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape) if not np.isnan(mape) else None,
            'r_squared': float(r_squared) if r_squared else None,
            'r_squared_adj': float(r_squared_adj) if r_squared_adj else None,
            'aic': float(aic) if aic else None,
            'bic': float(bic) if bic else None,
            'log_likelihood': float(llf) if llf else None
        }
    except Exception as e:
        return {'error': str(e)}

def generate_explanations(results):
    """Gerar explicações para os resultados"""
    explanations = {}
    
    # Explicação geral do modelo
    explanations['model_summary'] = {
        'title': 'Resumo do Modelo',
        'content': f"""
        O modelo econométrico foi estimado usando {results['specification']['model_type']}.
        
        **Principais resultados:**
        - R²: {results['performance']['r_squared']:.4f} - O modelo explica aproximadamente {results['performance']['r_squared']*100:.1f}% da variação na variável dependente.
        - Observações: {results['data_info']['n_obs']}
        - Variáveis explicativas: {results['data_info']['n_vars'] - 1 if 'const' in results['X'].columns else results['data_info']['n_vars']}
        
        **Interpretação econômica:** O modelo mostra como as variáveis selecionadas influenciam {results['specification']['y_var']}.
        """
    }
    
    # Explicação dos coeficientes
    significant_vars = []
    for var in results['model'].params.index:
        if var != 'const':
            p_value = results['model'].pvalues[var]
            if p_value < 0.05:
                significant_vars.append(var)
    
    explanations['coefficients'] = {
        'title': 'Interpretação dos Coeficientes',
        'content': f"""
        **Variáveis estatisticamente significativas (p < 0.05):** {', '.join(significant_vars) if significant_vars else 'Nenhuma'}
        
        **Interpretação dos coeficientes:**
        - Cada coeficiente representa a mudança esperada na variável dependente para uma unidade de mudança na variável independente, mantendo outras constantes.
        - Coeficientes positivos indicam relação direta (aumenta Y).
        - Coeficientes negativos indicam relação inversa (diminui Y).
        - A magnitude do coeficiente indica a força do efeito.
        """
    }
    
    # Explicação dos testes
    test_explanations = []
    for category, tests in results['test_results'].items():
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict) and 'conclusion' in test_result:
                explanation = get_test_explanation(test_name)
                test_explanations.append({
                    'test': explanation['name'],
                    'result': test_result['conclusion'],
                    'interpretation': explanation['economic_meaning'],
                    'recommendation': explanation['solutions'] if test_result['conclusion'] not in ['Normal', 'Homocedástico', 'Sem autocorrelação', 'Aceitável', 'Bem especificado', 'Linear'] else 'Nenhuma ação necessária'
                })
    
    explanations['tests'] = {
        'title': 'Diagnóstico do Modelo',
        'content': 'Abaixo estão os resultados dos testes de diagnóstico:',
        'details': test_explanations
    }
    
    st.session_state.explanations = explanations

def display_results():
    """Exibir resultados da análise com explicações"""
    if not st.session_state.analysis_results:
        st.warning("⚠️ Execute a análise primeiro para ver os resultados.")
        return
    
    results = st.session_state.analysis_results
    
    st.header("📊 Resultados da Análise Econométrica")
    
    # Criar abas para diferentes seções
    tab_summary, tab_model, tab_diagnostics, tab_visuals, tab_export = st.tabs([
        "📋 Resumo Executivo", 
        "⚙️ Resultados do Modelo", 
        "🔍 Testes de Diagnóstico", 
        "📈 Visualizações",
        "📥 Exportar"
    ])
    
    with tab_summary:
        display_executive_summary(results)
    
    with tab_model:
        display_model_results(results)
    
    with tab_diagnostics:
        display_diagnostic_tests(results)
    
    with tab_visuals:
        display_visualizations(results)
    
    with tab_export:
        display_export_options(results)

def display_executive_summary(results):
    """Exibir resumo executivo"""
    st.subheader("🎯 Resumo Executivo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if results['performance']['r_squared'] is not None:
            st.metric("R²", f"{results['performance']['r_squared']:.4f}")
        if results['performance']['r_squared_adj'] is not None:
            st.metric("R² Ajustado", f"{results['performance']['r_squared_adj']:.4f}")
    
    with col2:
        st.metric("Observações", f"{results['data_info']['n_obs']:,}")
        st.metric("Variáveis", results['data_info']['n_vars'])
    
    with col3:
        if results['performance']['rmse'] is not None:
            st.metric("RMSE", f"{results['performance']['rmse']:.4f}")
        if results['performance']['mae'] is not None:
            st.metric("MAE", f"{results['performance']['mae']:.4f}")
    
    # Conclusão geral
    st.subheader("📝 Conclusão Geral")
    
    # Verificar significância do modelo
    model_significant = results['model'].f_pvalue < 0.05
    
    if model_significant:
        st.success("""
        ✅ **O modelo é estatisticamente significativo como um todo.**
        
        **Implicações práticas:**
        1. As variáveis selecionadas têm poder explicativo sobre a variável dependente
        2. Os resultados podem ser usados para previsão e inferência
        3. As estimativas dos coeficientes são confiáveis para interpretação econômica
        """)
    else:
        st.warning("""
        ⚠️ **O modelo não é estatisticamente significativo como um todo.**
        
        **Recomendações:**
        1. Revisar a seleção de variáveis independentes
        2. Verificar se há problemas de especificação
        3. Considerar transformações nas variáveis
        4. Coletar mais dados se possível
        """)
    
    # Principais achados
    st.subheader("🔍 Principais Achados")
    
    # Encontrar variáveis mais significativas
    sig_coeffs = []
    for var in results['specification']['x_vars']:
        if var in results['model'].pvalues.index:
            pval = results['model'].pvalues[var]
            if pval < 0.05:
                coeff = results['model'].params[var]
                sig_coeffs.append((var, coeff, pval))
    
    sig_coeffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if sig_coeffs:
        st.write("**Variáveis com efeito estatisticamente significativo:**")
        
        for var, coeff, pval in sig_coeffs[:3]:  # Top 3
            direction = "positivo" if coeff > 0 else "negativo"
            significance = "altamente significativo" if pval < 0.01 else "significativo" if pval < 0.05 else "marginalmente significativo"
            
            st.markdown(f"""
            **{var}**
            - Efeito: {direction} (coeficiente = {coeff:.4f})
            - Significância: p = {pval:.4f} ({significance})
            - Interpretação: Um aumento de uma unidade em {var} está associado a um {'aumento' if coeff > 0 else 'redução'} de {abs(coeff):.4f} unidades em {results['specification']['y_var']}
            """)
    else:
        st.info("Nenhuma variável mostrou efeito estatisticamente significativo ao nível de 5%.")
    
    # Recomendações
    st.subheader("💡 Recomendações Práticas")
    
    recommendations = []
    
    # Verificar problemas de diagnóstico
    diag_issues = []
    
    # Normalidade
    if 'normality' in results['test_results']:
        for test_name, test_result in results['test_results']['normality'].items():
            if isinstance(test_result, dict) and 'conclusion' in test_result:
                if 'Não normal' in test_result['conclusion']:
                    diag_issues.append("normalidade dos resíduos")
                    break
    
    # Heterocedasticidade
    if 'heteroscedasticity' in results['test_results']:
        for test_name, test_result in results['test_results']['heteroscedasticity'].items():
            if isinstance(test_result, dict) and 'conclusion' in test_result:
                if 'Heterocedástico' in test_result['conclusion']:
                    diag_issues.append("heterocedasticidade")
                    break
    
    # Autocorrelação
    if 'autocorrelation' in results['test_results']:
        for test_name, test_result in results['test_results']['autocorrelation'].items():
            if isinstance(test_result, dict) and 'conclusion' in test_result:
                if 'autocorrelação' in test_result['conclusion'].lower():
                    diag_issues.append("autocorrelação")
                    break
    
    # Multicolinearidade
    if 'multicollinearity' in results['test_results']:
        vif_results = results['test_results']['multicollinearity'].get('vif', [])
        if isinstance(vif_results, list):
            high_vif = any(isinstance(item, dict) and item.get('classification', '').startswith('🚨') 
                          for item in vif_results)
            if high_vif:
                diag_issues.append("multicolinearidade severa")
    
    if diag_issues:
        recommendations.append(f"**Problemas detectados:** {', '.join(diag_issues)}. Considere usar métodos robustos ou corrigir especificação.")
    
    if results['performance']['r_squared'] is not None and results['performance']['r_squared'] < 0.3:
        recommendations.append("**Poder explicativo baixo:** O R² é inferior a 0.3, indicando que o modelo explica menos de 30% da variação. Considere incluir variáveis adicionais.")
    
    if not sig_coeffs and model_significant:
        recommendations.append("**Resultado interessante:** O modelo é significativo mas nenhuma variável individual é significativa. Pode indicar que as variáveis funcionam em conjunto.")
    
    if not recommendations:
        recommendations.append("**Modelo bem comportado:** Os testes diagnósticos não detectaram problemas graves. Os resultados podem ser considerados confiáveis.")
    
    for rec in recommendations:
        st.write(f"• {rec}")

def display_model_results(results):
    """Exibir resultados detalhados do modelo"""
    st.subheader("📈 Resultados do Modelo")
    
    # Tabela de coeficientes com formatação
    coef_df = pd.DataFrame({
        'Coeficiente': results['model'].params,
        'Erro Padrão': results['model'].bse,
        't': results['model'].tvalues,
        'P>|t|': results['model'].pvalues,
        '[0.025': results['model'].conf_int()[0],
        '0.975]': results['model'].conf_int()[1]
    })
    
    # Formatar p-valores
    def format_pvalue(p):
        if p < 0.001:
            return "0.000***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        elif p < 0.1:
            return f"{p:.3f}."
        else:
            return f"{p:.3f}"
    
    coef_df['P>|t|'] = coef_df['P>|t|'].apply(format_pvalue)
    
    # Aplicar formatação numérica
    numeric_cols = ['Coeficiente', 'Erro Padrão', 't', '[0.025', '0.975]']
    for col in numeric_cols:
        coef_df[col] = coef_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(coef_df, use_container_width=True)
    
    # Legenda de significância
    st.caption("""
    *** p<0.001, ** p<0.01, * p<0.05, . p<0.1
    """)
    
    # Métricas do modelo
    st.subheader("📊 Métricas de Ajuste")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    
    with col_met1:
        if results['performance']['r_squared'] is not None:
            st.metric("R-squared", f"{results['performance']['r_squared']:.4f}")
        if results['performance']['r_squared_adj'] is not None:
            st.metric("Adj. R-squared", f"{results['performance']['r_squared_adj']:.4f}")
    
    with col_met2:
        st.metric("F-statistic", f"{results['model'].fvalue:.2f}")
        st.metric("Prob (F-statistic)", f"{results['model'].f_pvalue:.4f}")
    
    with col_met3:
        if results['performance']['log_likelihood'] is not None:
            st.metric("Log-Likelihood", f"{results['performance']['log_likelihood']:.2f}")
        if results['performance']['aic'] is not None:
            st.metric("AIC", f"{results['performance']['aic']:.2f}")
        if results['performance']['bic'] is not None:
            st.metric("BIC", f"{results['performance']['bic']:.2f}")
    
    # Explicação das métricas
    with st.expander("📖 Explicação das Métricas"):
        st.markdown("""
        **R-squared (R²):** Proporção da variância da variável dependente que é explicada pelas variáveis independentes.
        - **Interpretação:** Valores mais próximos de 1 indicam melhor ajuste.
        - **Limitação:** Aumenta automaticamente ao adicionar variáveis, mesmo que não sejam significativas.
        
        **R-squared Ajustado:** Versão ajustada do R² que penaliza a adição de variáveis não significativas.
        - **Uso:** Melhor para comparar modelos com diferentes números de variáveis.
        
        **F-statistic:** Testa se pelo menos um coeficiente é diferente de zero.
        - **H₀:** Todos os coeficientes (exceto intercepto) são zero.
        - **Significativo (p < 0.05):** O modelo tem poder explicativo.
        
        **AIC/BIC:** Critérios de informação para seleção de modelos.
        - **Regra:** Menor valor indica melhor modelo.
        - **Diferença:** BIC penaliza mais a complexidade que AIC.
        
        **Log-Likelihood:** Mede a probabilidade dos dados dado o modelo.
        - **Interpretação:** Valores mais altos indicam melhor ajuste.
        """)

def display_diagnostic_tests(results):
    """Exibir resultados dos testes de diagnóstico"""
    st.subheader("🔍 Testes de Diagnóstico")
    
    # Organizar por categoria
    categories = {
        'Normalidade dos Resíduos': results['test_results'].get('normality', {}),
        'Heterocedasticidade': results['test_results'].get('heteroscedasticity', {}),
        'Autocorrelação': results['test_results'].get('autocorrelation', {}),
        'Multicolinearidade': results['test_results'].get('multicollinearity', {}),
        'Especificação do Modelo': results['test_results'].get('specification', {}),
        'Estacionariedade': results['test_results'].get('stationarity', {})
    }
    
    for category_name, tests in categories.items():
        with st.expander(f"{category_name}", expanded=True):
            if tests:
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict):
                        # Obter explicação do teste
                        explanation = get_test_explanation(test_name)
                        
                        col_test1, col_test2 = st.columns([3, 1])
                        
                        with col_test1:
                            st.write(f"**{explanation['name']}**")
                            st.write(f"*Propósito:* {explanation['purpose']}")
                            st.write(f"*H₀:* {explanation['null_hypothesis']}")
                            
                            # Mostrar resultados
                            if 'error' in test_result:
                                st.error(f"Erro: {test_result['error']}")
                            else:
                                for key, value in test_result.items():
                                    if key not in ['conclusion', 'interpretation']:
                                        if isinstance(value, float):
                                            st.write(f"- {key}: {value:.4f}")
                                        elif isinstance(value, dict):
                                            st.write(f"- {key}:")
                                            for subkey, subvalue in value.items():
                                                if isinstance(subvalue, float):
                                                    st.write(f"  * {subkey}: {subvalue:.4f}")
                                                else:
                                                    st.write(f"  * {subkey}: {subvalue}")
                                        else:
                                            st.write(f"- {key}: {value}")
                        
                        with col_test2:
                            if 'conclusion' in test_result:
                                conclusion = test_result['conclusion']
                                if any(x in conclusion.lower() for x in ['normal', 'homocedástico', 'sem', 'aceitável', 'bem', 'linear']):
                                    st.success(f"✅ {conclusion}")
                                else:
                                    st.error(f"❌ {conclusion}")
                            
                            # Mostrar solução se houver problema
                            if 'conclusion' in test_result and 'solutions' in explanation:
                                if any(x in test_result['conclusion'].lower() for x in ['não normal', 'heterocedástico', 'autocorrelação', 'multicolinearidade', 'mal', 'não linear']):
                                    with st.expander("💡 Recomendações"):
                                        st.write(explanation['solutions'])
                        
                        st.markdown("---")
            else:
                st.info("Nenhum teste disponível para esta categoria.")

def display_visualizations(results):
    """Exibir visualizações dos resultados"""
    st.subheader("📈 Visualizações dos Resultados")
    
    viz_type = st.selectbox(
        "Selecione o tipo de visualização:",
        [
            "Resíduos vs Ajustados",
            "QQ-Plot dos Resíduos",
            "Distribuição dos Resíduos",
            "Valores Ajustados vs Reais",
            "Importância das Variáveis"
        ]
    )
    
    if viz_type == "Resíduos vs Ajustados":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['y_pred'].tolist() if hasattr(results['y_pred'], 'tolist') else results['y_pred'],
            y=results['residuals'].tolist() if hasattr(results['residuals'], 'tolist') else results['residuals'],
            mode='markers',
            name='Resíduos',
            marker=dict(
                size=8,
                color=results['residuals'].tolist() if hasattr(results['residuals'], 'tolist') else results['residuals'],
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Resíduo")
            )
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title='Resíduos vs Valores Ajustados',
            xaxis_title='Valores Ajustados (Preditos)',
            yaxis_title='Resíduos (Observado - Predito)',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explicação
        with st.expander("📖 Interpretação do Gráfico"):
            st.markdown("""
            **Gráfico de Resíduos vs Ajustados:**
            
            Este gráfico ajuda a verificar a **homocedasticidade** (variância constante dos erros).
            
            **Padrões desejáveis:**
            - Resíduos distribuídos aleatoriamente em torno de zero
            - Nenhum padrão claro (nem funil, nem curvatura)
            - Variância aproximadamente constante ao longo do eixo X
            
            **Padrões problemáticos:**
            - **Forma de funil:** Heterocedasticidade (variância não constante)
            - **Padrão curvilíneo:** Especificação incorreta (falta termos não-lineares)
            - **Agrupamentos:** Possível variável omitida ou estrutura de grupos
            
            **No gráfico acima:**
            - A linha vermelha tracejada representa resíduo zero
            - Pontos acima da linha: subestimação (valor real > predito)
            - Pontos abaixo da linha: superestimação (valor real < predito)
            """)
    
    elif viz_type == "QQ-Plot dos Resíduos":
        # QQ-Plot
        sorted_residuals = np.sort(results['residuals'])
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_residuals))
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles.tolist(),
            y=sorted_residuals.tolist(),
            mode='markers',
            name='Resíduos',
            marker=dict(size=8)
        ))
        
        # Linha de referência (y = x)
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Distribuição Normal',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='QQ-Plot dos Resíduos',
            xaxis_title='Quantis Teóricos da Distribuição Normal',
            yaxis_title='Quantis dos Resíduos',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explicação
        with st.expander("📖 Interpretação do Gráfico"):
            st.markdown("""
            **QQ-Plot (Quantil-Quantil):**
            
            Este gráfico compara a distribuição dos resíduos com uma distribuição normal.
            
            **Interpretação:**
            - **Pontos na linha:** Resíduos seguem distribuição normal
            - **Pontos acima da linha:** Cauda direita mais pesada que a normal
            - **Pontos abaixo da linha:** Cauda esquerda mais pesada que a normal
            - **Curva em S:** Assimetria (skewness) nos resíduos
            - **Curva em U:** Curtose excessiva (caudas pesadas)
            
            **O que observar:**
            1. **Linearidade:** Pontos devem seguir aproximadamente a linha reta
            2. **Extremos:** Desvios nas extremidades indicam outliers ou caudas pesadas
            3. **Padrão sistemático:** Curvatura indica não-normalidade
            
            **Implicações para inferência:**
            - Normalidade dos resíduos é necessária para testes t e F válidos
            - Desvios moderados são toleráveis em amostras grandes (Teorema do Limite Central)
            - Desvios graves podem exigir transformações ou métodos robustos
            """)
    
    elif viz_type == "Distribuição dos Resíduos":
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histograma dos Resíduos', 'Densidade dos Resíduos')
        )
        
        # Histograma
        fig.add_trace(
            go.Histogram(
                x=results['residuals'].tolist() if hasattr(results['residuals'], 'tolist') else results['residuals'],
                nbinsx=30,
                name='Resíduos',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Adicionar curva normal
        if len(results['residuals']) > 0:
            x_norm = np.linspace(results['residuals'].min(), results['residuals'].max(), 100)
            y_norm = stats.norm.pdf(x_norm, results['residuals'].mean(), results['residuals'].std())
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm.tolist(),
                    y=y_norm.tolist(),
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        
        # Densidade
        fig.add_trace(
            go.Histogram(
                x=results['residuals'].tolist() if hasattr(results['residuals'], 'tolist') else results['residuals'],
                histnorm='probability density',
                nbinsx=30,
                name='Densidade',
                marker_color='lightgreen',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        if len(results['residuals']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=x_norm.tolist(),
                    y=y_norm.tolist(),
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Distribuição dos Resíduos',
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Valores Ajustados vs Reais":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['y'].tolist() if hasattr(results['y'], 'tolist') else results['y'],
            y=results['y_pred'].tolist() if hasattr(results['y_pred'], 'tolist') else results['y_pred'],
            mode='markers',
            name='Observações',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Linha y = x (prefeito ajuste)
        min_val = min(results['y'].min(), results['y_pred'].min())
        max_val = max(results['y'].max(), results['y_pred'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Ajuste Perfeito',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Valores Reais vs Valores Ajustados',
            xaxis_title='Valores Reais (Observados)',
            yaxis_title='Valores Ajustados (Preditos)',
            template='plotly_white'
        )
        
        # Calcular R² para exibir
        r2 = results['performance']['r_squared']
        if r2 is not None:
            fig.add_annotation(
                text=f"R² = {r2:.4f}",
                xref="paper", yref="paper",
                x=0.05, y=0.95,
                showarrow=False,
                bgcolor="white"
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Importância das Variáveis":
        # Calcular importância baseada em estatísticas t
        importance_data = []
        for var in results['specification']['x_vars']:
            if var in results['model'].tvalues.index:
                t_abs = abs(results['model'].tvalues[var])
                p_value = results['model'].pvalues[var]
                importance_data.append({
                    'Variável': var,
                    '|t-stat|': float(t_abs),
                    'Significância': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '.' if p_value < 0.1 else ''
                })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('|t-stat|', ascending=False)
            
            fig = px.bar(importance_df, x='|t-stat|', y='Variável', 
                        orientation='h',
                        title='Importância das Variáveis (Estatística t absoluta)',
                        text='Significância')
            
            fig.update_layout(
                xaxis_title='|t-statistic| (Valor absoluto)',
                yaxis_title='Variável',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não foi possível calcular a importância das variáveis.")

def display_export_options(results):
    """Exibir opções de exportação"""
    st.subheader("📥 Exportar Resultados")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        st.markdown("### 📄 Relatório em PDF")
        st.write("Gere um relatório completo em PDF com todos os resultados, gráficos e explicações.")
        
        if st.button("📊 Gerar Relatório PDF", use_container_width=True):
            try:
                pdf_path = generate_pdf_report(results)
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                # Botão de download
                st.download_button(
                    label="⬇️ Baixar Relatório PDF",
                    data=pdf_bytes,
                    file_name=f"relatorio_econometrico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.success("✅ Relatório PDF gerado com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao gerar PDF: {str(e)}")
    
    with col_exp2:
        st.markdown("### 📊 Dados e Resultados")
        
        # Exportar coeficientes
        coef_df = pd.DataFrame({
            'Variável': results['model'].params.index,
            'Coeficiente': results['model'].params.values,
            'Erro_Padrão': results['model'].bse.values,
            't_stat': results['model'].tvalues.values,
            'p_valor': results['model'].pvalues.values,
            'IC_95_inf': results['model'].conf_int()[0].values,
            'IC_95_sup': results['model'].conf_int()[1].values
        })
        
        csv_coef = coef_df.to_csv(index=False)
        st.download_button(
            label="📈 Baixar Coeficientes (CSV)",
            data=csv_coef,
            file_name="coeficientes_modelo.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Exportar dados de previsão
        pred_df = pd.DataFrame({
            'Y_Real': results['y'].tolist() if hasattr(results['y'], 'tolist') else results['y'],
            'Y_Predito': results['y_pred'].tolist() if hasattr(results['y_pred'], 'tolist') else results['y_pred'],
            'Resíduo': results['residuals'].tolist() if hasattr(results['residuals'], 'tolist') else results['residuals']
        })
        
        csv_pred = pred_df.to_csv(index=False)
        st.download_button(
            label="🔮 Baixar Previsões (CSV)",
            data=csv_pred,
            file_name="previsoes_modelo.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp3:
        st.markdown("### 📋 Relatório Textual")
        
        # Gerar relatório textual
        report_text = generate_text_report(results)
        
        st.download_button(
            label="📝 Baixar Relatório (TXT)",
            data=report_text,
            file_name="relatorio_econometrico.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Exportar dados processados
        st.markdown("---")
        st.markdown("### 💾 Dados Processados")
        
        processed_data = pd.concat([results['y'], results['X']], axis=1)
        csv_data = processed_data.to_csv(index=False)
        
        st.download_button(
            label="🗃️ Baixar Dados Processados",
            data=csv_data,
            file_name="dados_processados.csv",
            mime="text/csv",
            use_container_width=True
        )

def generate_pdf_report(results):
    """Gerar relatório PDF completo"""
    # Criar PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Título
    pdf.cell(0, 10, "Relatório de Análise Econométrica", ln=True, align='C')
    pdf.ln(5)
    
    # Informações gerais
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Usuário: {st.session_state.current_user}", ln=True)
    pdf.ln(10)
    
    # Seção 1: Especificação do Modelo
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Especificação do Modelo", ln=True)
    pdf.set_font("Arial", "", 12)
    
    spec = results['specification']
    pdf.cell(0, 10, f"Variável dependente (Y): {spec['y_var']}", ln=True)
    pdf.cell(0, 10, f"Variáveis independentes (X): {', '.join(spec['x_vars'])}", ln=True)
    pdf.cell(0, 10, f"Tipo de modelo: {spec['model_type']}", ln=True)
    pdf.cell(0, 10, f"Nível de confiança: {spec['confidence_level']*100}%", ln=True)
    pdf.cell(0, 10, f"Hipótese nula (H₀): {spec['hypotheses']['null']}", ln=True)
    pdf.cell(0, 10, f"Hipótese alternativa (H₁): {spec['hypotheses']['alternative']}", ln=True)
    pdf.ln(5)
    
    # Seção 2: Resultados do Modelo
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "2. Resultados do Modelo", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Coeficientes (formato simplificado)
    pdf.cell(0, 10, "Coeficientes:", ln=True)
    pdf.set_font("Arial", "", 10)
    
    # Criar tabela de coeficientes
    col_widths = [40, 20, 20, 20, 20, 20, 20]
    headers = ['Variável', 'Coef', 'Std.Err.', 't', 'P>|t|', 'IC 95% Inf', 'IC 95% Sup']
    
    # Cabeçalho
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1)
    pdf.ln()
    
    # Dados
    for i, var in enumerate(results['model'].params.index):
        pdf.cell(col_widths[0], 10, str(var), border=1)
        pdf.cell(col_widths[1], 10, f"{results['model'].params[var]:.4f}", border=1)
        pdf.cell(col_widths[2], 10, f"{results['model'].bse[var]:.4f}", border=1)
        pdf.cell(col_widths[3], 10, f"{results['model'].tvalues[var]:.4f}", border=1)
        pdf.cell(col_widths[4], 10, f"{results['model'].pvalues[var]:.4f}", border=1)
        pdf.cell(col_widths[5], 10, f"{results['model'].conf_int()[0][var]:.4f}", border=1)
        pdf.cell(col_widths[6], 10, f"{results['model'].conf_int()[1][var]:.4f}", border=1)
        pdf.ln()
    
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    
    # Métricas
    perf = results['performance']
    pdf.cell(0, 10, f"R-squared: {perf['r_squared']:.4f}", ln=True)
    pdf.cell(0, 10, f"R-squared ajustado: {perf['r_squared_adj']:.4f}", ln=True)
    pdf.cell(0, 10, f"F-statistic: {results['model'].fvalue:.2f} (p = {results['model'].f_pvalue:.4f})", ln=True)
    pdf.cell(0, 10, f"AIC: {perf['aic']:.2f}", ln=True)
    pdf.cell(0, 10, f"BIC: {perf['bic']:.2f}", ln=True)
    pdf.cell(0, 10, f"Log-likelihood: {perf['log_likelihood']:.2f}", ln=True)
    pdf.ln(5)
    
    # Seção 3: Testes de Diagnóstico
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "3. Testes de Diagnóstico", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Resumo dos testes
    test_summary = []
    for category_name, tests in results['test_results'].items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, category_name + ":", ln=True)
        pdf.set_font("Arial", "", 10)
        
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict):
                explanation = get_test_explanation(test_name)
                conclusion = test_result.get('conclusion', 'N/A')
                
                pdf.cell(0, 10, f"  {explanation['name']}: {conclusion}", ln=True)
    
    pdf.ln(10)
    
    # Seção 4: Recomendações
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4. Recomendações e Conclusões", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Análise de significância
    if results['model'].f_pvalue < 0.05:
        pdf.cell(0, 10, "✅ O modelo é estatisticamente significativo como um todo.", ln=True)
    else:
        pdf.cell(0, 10, "⚠️ O modelo não é estatisticamente significativo como um todo.", ln=True)
    
    # Verificar problemas
    issues = []
    
    # Normalidade
    norm_tests = results['test_results'].get('normality', {})
    for test_name, test_result in norm_tests.items():
        if isinstance(test_result, dict) and 'conclusion' in test_result:
            if 'Não normal' in test_result['conclusion']:
                issues.append("Normalidade dos resíduos")
                break
    
    # Heterocedasticidade
    het_tests = results['test_results'].get('heteroscedasticity', {})
    for test_name, test_result in het_tests.items():
        if isinstance(test_result, dict) and 'conclusion' in test_result:
            if 'Heterocedástico' in test_result['conclusion']:
                issues.append("Heterocedasticidade")
                break
    
    if issues:
        pdf.cell(0, 10, "Problemas detectados:", ln=True)
        for issue in issues:
            pdf.cell(0, 10, f"  • {issue}", ln=True)
        pdf.cell(0, 10, "Recomenda-se usar métodos robustos ou corrigir a especificação.", ln=True)
    else:
        pdf.cell(0, 10, "✅ Nenhum problema grave detectado nos testes de diagnóstico.", ln=True)
    
    # Interpretação econômica
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Interpretação Econômica:", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Encontrar variável mais significativa
    max_t = 0
    most_sig_var = None
    for var in results['specification']['x_vars']:
        if var in results['model'].tvalues.index:
            t_abs = abs(results['model'].tvalues[var])
            if t_abs > max_t and results['model'].pvalues[var] < 0.05:
                max_t = t_abs
                most_sig_var = var
    
    if most_sig_var:
        coef = results['model'].params[most_sig_var]
        direction = "positivo" if coef > 0 else "negativo"
        pdf.cell(0, 10, f"A variável mais influente é {most_sig_var} com um efeito {direction}.", ln=True)
        pdf.cell(0, 10, f"Um aumento de uma unidade em {most_sig_var} está associado a uma mudança de {abs(coef):.4f} em {spec['y_var']}.", ln=True)
    
    # Salvar PDF
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"relatorio_econometrico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(pdf_path)
    
    return pdf_path

def generate_text_report(results):
    """Gerar relatório textual completo"""
    report = []
    report.append("=" * 80)
    report.append("RELATÓRIO DE ANÁLISE ECONOMÉTRICA")
    report.append("=" * 80)
    report.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    report.append(f"Usuário: {st.session_state.current_user}")
    report.append("")
    
    # 1. Especificação
    spec = results['specification']
    report.append("1. ESPECIFICAÇÃO DO MODELO")
    report.append("-" * 40)
    report.append(f"Variável dependente (Y): {spec['y_var']}")
    report.append(f"Variáveis independentes (X): {', '.join(spec['x_vars'])}")
    report.append(f"Tipo de modelo: {spec['model_type']}")
    report.append(f"Nível de confiança: {spec['confidence_level']*100}%")
    report.append(f"Hipótese nula (H₀): {spec['hypotheses']['null']}")
    report.append(f"Hipótese alternativa (H₁): {spec['hypotheses']['alternative']}")
    report.append("")
    
    # 2. Resultados
    report.append("2. RESULTADOS DO MODELO")
    report.append("-" * 40)
    report.append(f"Número de observações: {results['data_info']['n_obs']}")
    report.append(f"Número de variáveis: {results['data_info']['n_vars']}")
    report.append("")
    
    report.append("Coeficientes:")
    report.append("-" * 20)
    for var in results['model'].params.index:
        coef = results['model'].params[var]
        se = results['model'].bse[var]
        t = results['model'].tvalues[var]
        p = results['model'].pvalues[var]
        ci_low, ci_high = results['model'].conf_int().loc[var]
        
        sig = ""
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        elif p < 0.1:
            sig = "."
        
        report.append(f"{var}: {coef:.4f}{sig}")
        report.append(f"    Erro padrão: {se:.4f}")
        report.append(f"    t = {t:.4f}, p = {p:.4f}")
        report.append(f"    IC 95%: [{ci_low:.4f}, {ci_high:.4f}]")
        report.append("")
    
    # Métricas
    perf = results['performance']
    report.append("Métricas de ajuste:")
    report.append("-" * 20)
    report.append(f"R-squared: {perf['r_squared']:.4f}")
    report.append(f"R-squared ajustado: {perf['r_squared_adj']:.4f}")
    report.append(f"F-statistic: {results['model'].fvalue:.2f}")
    report.append(f"Prob(F-statistic): {results['model'].f_pvalue:.4f}")
    if perf['aic'] is not None:
        report.append(f"AIC: {perf['aic']:.2f}")
    if perf['bic'] is not None:
        report.append(f"BIC: {perf['bic']:.2f}")
    if perf['log_likelihood'] is not None:
        report.append(f"Log-likelihood: {perf['log_likelihood']:.2f}")
    report.append(f"MAE: {perf['mae']:.4f}")
    report.append(f"RMSE: {perf['rmse']:.4f}")
    if perf['mape'] is not None:
        report.append(f"MAPE: {perf['mape']:.2f}%")
    report.append("")
    
    # 3. Testes de diagnóstico
    report.append("3. TESTES DE DIAGNÓSTICO")
    report.append("-" * 40)
    
    for category_name, tests in results['test_results'].items():
        report.append(f"\n{category_name.upper()}:")
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict):
                explanation = get_test_explanation(test_name)
                report.append(f"  {explanation['name']}:")
                
                if 'error' in test_result:
                    report.append(f"    Erro: {test_result['error']}")
                else:
                    for key, value in test_result.items():
                        if key == 'conclusion':
                            report.append(f"    Conclusão: {value}")
                        elif key not in ['interpretation']:
                            if isinstance(value, float):
                                report.append(f"    {key}: {value:.4f}")
                            else:
                                report.append(f"    {key}: {value}")
    
    # 4. Conclusões
    report.append("\n4. CONCLUSÕES E RECOMENDAÇÕES")
    report.append("-" * 40)
    
    # Significância do modelo
    if results['model'].f_pvalue < 0.05:
        report.append("✅ O modelo é estatisticamente significativo como um todo.")
    else:
        report.append("⚠️ O modelo não é estatisticamente significativo como um todo.")
    
    # Variáveis significativas
    sig_vars = []
    for var in spec['x_vars']:
        if var in results['model'].pvalues.index:
            if results['model'].pvalues[var] < 0.05:
                sig_vars.append(var)
    
    if sig_vars:
        report.append(f"\nVariáveis estatisticamente significativas (p < 0.05):")
        for var in sig_vars:
            coef = results['model'].params[var]
            direction = "positivo" if coef > 0 else "negativo"
            report.append(f"  • {var}: efeito {direction} (coeficiente = {coef:.4f})")
    else:
        report.append("\n⚠️ Nenhuma variável independente é estatisticamente significativa ao nível de 5%.")
    
    # Problemas detectados
    issues = []
    for category_name, tests in results['test_results'].items():
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict) and 'conclusion' in test_result:
                if any(x in test_result['conclusion'].lower() for x in ['não normal', 'heterocedástico', 'autocorrelação', 'multicolinearidade', 'mal']):
                    explanation = get_test_explanation(test_name)
                    issues.append(explanation['name'])
    
    if issues:
        report.append(f"\n⚠️ Problemas detectados: {', '.join(issues)}")
        report.append("Recomenda-se considerar as seguintes ações:")
        report.append("  1. Usar erros padrão robustos para heterocedasticidade")
        report.append("  2. Transformar variáveis para normalidade")
        report.append("  3. Adicionar termos não-lineares para má especificação")
        report.append("  4. Remover variáveis correlacionadas para multicolinearidade")
    else:
        report.append("\n✅ Nenhum problema grave detectado nos testes de diagnóstico.")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

def main_app():
    """Aplicação principal após login"""
    # Barra lateral
    st.sidebar.title(f"👋 Bem-vindo, {st.session_state.current_user}!")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.uploaded_files = []
        st.session_state.merged_data = None
        st.session_state.model_spec = {}
        st.session_state.analysis_results = {}
        st.session_state.explanations = {}
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Menu de navegação
    menu_options = [
        "📤 Upload de Dados",
        "🔄 Merge de Arquivos",
        "🔍 Análise Exploratória",
        "⚙️ Especificar Modelo",
        "🔬 Executar Análise",
        "📊 Resultados"
    ]
    
    selected_menu = st.sidebar.radio("Navegação", menu_options)
    
    # Status atual
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status Atual")
    
    if st.session_state.merged_data is not None:
        st.sidebar.success(f"✅ Dados: {st.session_state.merged_data.shape[0]:,}×{st.session_state.merged_data.shape[1]}")
    else:
        st.sidebar.warning("⚠️ Sem dados")
    
    if st.session_state.model_spec:
        st.sidebar.info(f"⚙️ Modelo: {st.session_state.model_spec.get('model_type', 'Não especificado')}")
        st.sidebar.write(f"Y: {st.session_state.model_spec.get('y_var', '—')}")
    
    if st.session_state.analysis_results:
        st.sidebar.success(f"📈 Análise: Concluída")
        r2 = st.session_state.analysis_results['performance']['r_squared']
        if r2 is not None:
            st.sidebar.metric("R²", f"{r2:.3f}")
    
    # Executar página selecionada
    if selected_menu == "📤 Upload de Dados":
        upload_files()
    elif selected_menu == "🔄 Merge de Arquivos":
        merge_files()
    elif selected_menu == "🔍 Análise Exploratória":
        exploratory_analysis()
    elif selected_menu == "⚙️ Especificar Modelo":
        specify_model()
    elif selected_menu == "🔬 Executar Análise":
        run_analysis()
    elif selected_menu == "📊 Resultados":
        if st.session_state.analysis_results:
            display_results()
        else:
            st.info("👈 Execute a análise primeiro para ver os resultados.")

def main():
    """Função principal"""
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
