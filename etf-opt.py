"""
Projeto de Otimização de Carteira de ETFs
Código em Python implementando o pipeline de otimização
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configurações globais
TOP_N_ETFS = 15
RISK_FREE_RATE = 0.03
MIN_ALLOCATION = 0.05
MAX_ALLOCATION = 0.30
GEOGRAPHIC_CONSTRAINTS = {'US': 0.6, 'EU': 0.3}
STRATEGY_CONSTRAINTS = {'Value': 0.4, 'Growth': 0.3}

# Funções auxiliares
def calculate_annual_return(price_history: List[float]) -> float:
    """Calcula retorno anualizado do histórico de preços"""
    if not price_history or len(price_history) < 2:
        return 0.0
    
    initial_price = price_history[0]
    final_price = price_history[-1]
    n_years = len(price_history) / 252  # Assumindo dias úteis
    
    if initial_price <= 0:
        return 0.0
        
    return (final_price / initial_price) ** (1 / n_years) - 1

def calculate_volatility(price_history: List[float]) -> float:
    """Calcula volatilidade anualizada do histórico de preços"""
    if not price_history or len(price_history) < 2:
        return 0.0
    
    # Calcular retornos diários
    returns = [price_history[i] / price_history[i-1] - 1 for i in range(1, len(price_history))]
    
    # Volatilidade anualizada (assumindo 252 dias úteis)
    return np.std(returns) * np.sqrt(252)

def calculate_max_drawdown(price_history: List[float]) -> float:
    """Calcula drawdown máximo do histórico de preços"""
    if not price_history or len(price_history) < 2:
        return 0.0
    
    # Calcular drawdowns
    peak = price_history[0]
    drawdowns = []
    
    for price in price_history:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        drawdowns.append(drawdown)
    
    return max(drawdowns)

def preprocess_data(etf_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processamento completo dos dados de ETFs:
    - Normalização de dados temporais
    - Encoding de variáveis categóricas
    - Tratamento de valores ausentes
    """
    print("Iniciando pré-processamento dos dados...")
    
    # Cópia para não modificar os dados originais
    df = etf_data.copy()
    
    # Normalização de dados temporais (retornos baseados em períodos anualizados)
    if 'price_history' in df.columns:
        df['annual_return'] = df['price_history'].apply(calculate_annual_return)
        df['volatility'] = df['price_history'].apply(calculate_volatility)
        df['max_drawdown'] = df['price_history'].apply(calculate_max_drawdown)
        df['sharpe_ratio'] = (df['annual_return'] - RISK_FREE_RATE) / df['volatility']
        df.drop('price_history', axis=1, inplace=True)
    
    # Criar métricas compostas para ranqueamento
    if 'last_five_years_return' in df.columns and 'last_five_years_volatility' in df.columns:
        df['last_five_years_return_per_risk'] = (
            df['last_five_years_return'] / df['last_five_years_volatility']
        )
        
    # Ajuste por TER (Taxa de despesa total)
    if 'ter' in df.columns and 'last_five_years_return_per_risk' in df.columns:
        df['adjusted_return_per_risk'] = df['last_five_years_return_per_risk'] * (1 - df['ter'])
    
    # Tratamento de valores ausentes para colunas numéricas
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Encoding de variáveis categóricas
    categorical_cols = ['strategy', 'domicile_country', 'asset_class', 'replication_method']
    cat_cols_present = [col for col in categorical_cols if col in df.columns]
    
    if cat_cols_present:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoded_cats = cat_encoder.fit_transform(df[cat_cols_present])
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=cat_encoder.get_feature_names_out(cat_cols_present),
            index=df.index
        )
        df = pd.concat([df.drop(cat_cols_present, axis=1), encoded_df], axis=1)
    
    print(f"Pré-processamento concluído. Shape final dos dados: {df.shape}")
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara features para o modelo de classificação
    """
    print("Preparando features para o modelo de classificação...")
    
    # Target para o modelo de classificação (ajustado por TER)
    target_col = 'adjusted_return_per_risk' if 'adjusted_return_per_risk' in df.columns else 'last_five_years_return_per_risk'
    
    if target_col not in df.columns:
        raise ValueError(f"Coluna target '{target_col}' não encontrada nos dados")
    
    # Features relevantes para classificação
    feature_cols = [
        'volatility', 'max_drawdown', 'age_years', 'fund_size_millions',
        'number_of_holdings', 'ter'
    ]
    
    # Adicionar colunas codificadas
    encoded_cols = [col for col in df.columns if col.startswith(('strategy_', 'domicile_', 'asset_class_', 'replication_'))]
    feature_cols.extend([col for col in encoded_cols if col in df.columns])
    
    # Filtrar apenas colunas existentes
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Verificar se há features suficientes
    if len(feature_cols) < 3:
        raise ValueError("Número insuficiente de features para treinamento do modelo")
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    print(f"Features preparadas. X shape: {X_scaled.shape}, y shape: {y.shape}")
    return X_scaled, y

def rank_etfs_with_gradient_boosting(X: pd.DataFrame, y: pd.Series, 
                                     etf_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fase 1: Ranqueamento de ETFs usando XGBoost
    """
    print("Fase 1: Iniciando ranqueamento de ETFs com Gradient Boosting...")
    
    # Split de treinamento/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treinar modelo XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Predições para todos os ETFs
    etf_data['predicted_score'] = model.predict(X)
    
    # Selecionar os top_n ETFs baseado no score predito
    top_etfs = etf_data.sort_values('predicted_score', ascending=False).head(TOP_N_ETFS)
    
    # Análise de feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Features mais importantes para o ranqueamento:")
    print(feature_importance.head(5))
    
    print(f"Fase 1 concluída. Selecionados {len(top_etfs)} ETFs top.")
    return top_etfs

def calculate_returns_and_covariance(etf_data: pd.DataFrame, 
                                    window_size: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calcula retornos esperados e matriz de covariância para otimização
    """
    print("Fase 2: Calculando retornos esperados e matriz de covariância...")
    
    # Extrair histórico de preços para cálculo da covariância
    if 'returns_history' in etf_data.columns:
        # Cada elemento em returns_history deve ser uma série de retornos diários
        returns_df = pd.DataFrame({
            etf_id: returns 
            for etf_id, returns in zip(etf_data.index, etf_data['returns_history'])
        })
    else:
        # Se não tiver histórico de retornos, usar estimativas de retorno e volatilidade
        # para criar uma matriz de covariância simplificada
        returns = etf_data['annual_return'] if 'annual_return' in etf_data.columns else 0.08
        volatilities = etf_data['volatility'] if 'volatility' in etf_data.columns else 0.15
        
        # Criar matriz de correlação aproximada (0.5 entre todos os pares)
        n_etfs = len(etf_data)
        correlation = np.ones((n_etfs, n_etfs)) * 0.5
        np.fill_diagonal(correlation, 1.0)
        
        # Calcular matriz de covariância a partir da correlação e volatilidades
        cov_matrix = np.outer(volatilities, volatilities) * correlation
        cov_matrix = pd.DataFrame(
            cov_matrix, 
            index=etf_data.index, 
            columns=etf_data.index
        )
        
        return pd.Series(returns, index=etf_data.index), cov_matrix
    
    # Calcular covariância com janela rolante
    cov_matrix = returns_df.rolling(window=window_size).cov().dropna()
    
    # Média dos retornos diários (anualizados)
    returns = returns_df.mean() * 252
    
    print(f"Fase 2: Cálculo de retornos e covariância concluído.")
    return returns, cov_matrix

def markowitz_optimization(returns: pd.Series, cov_matrix: pd.DataFrame, 
                           etf_data: pd.DataFrame) -> pd.Series:
    """
    Fase 2: Otimização de Markowitz para encontrar pesos ótimos
    """
    print("Fase 2: Iniciando otimização de Markowitz...")
    
    n_etfs = len(returns)
    
    # Função objetivo: maximizar Sharpe Ratio
    def objective(weights):
        weights = np.array(weights)
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
        # Queremos maximizar, então retornamos o negativo para minimização
        return -sharpe_ratio
    
    # Restrições
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Soma dos pesos = 1
    ]
    
    # Limites por ETF
    bounds = [(MIN_ALLOCATION, MAX_ALLOCATION) for _ in range(n_etfs)]
    
    # Adicionar restrições geográficas se especificadas
    if GEOGRAPHIC_CONSTRAINTS and 'domicile_country' in etf_data.columns:
        for country, max_alloc in GEOGRAPHIC_CONSTRAINTS.items():
            country_etfs = etf_data['domicile_country'] == country
            if country_etfs.any():
                # Função para restrição: soma dos pesos dos ETFs do país <= max_alloc
                def geographic_constraint(weights, country_mask=country_etfs.values):
                    return max_alloc - np.sum(weights[country_mask])
                
                constraints.append({
                    'type': 'ineq',
                    'fun': geographic_constraint
                })
    
    # Adicionar restrições por estratégia se especificadas
    if STRATEGY_CONSTRAINTS and 'strategy' in etf_data.columns:
        for strategy, max_alloc in STRATEGY_CONSTRAINTS.items():
            strategy_etfs = etf_data['strategy'] == strategy
            if strategy_etfs.any():
                def strategy_constraint(weights, strategy_mask=strategy_etfs.values):
                    return max_alloc - np.sum(weights[strategy_mask])
                
                constraints.append({
                    'type': 'ineq',
                    'fun': strategy_constraint
                })
    
    # Pesos iniciais iguais
    initial_weights = np.ones(n_etfs) / n_etfs
    
    # Executar otimização
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        print(f"Aviso: Otimização não convergiu. {result.message}")
        # Fallback para pesos iguais
        optimal_weights = pd.Series(initial_weights, index=returns.index)
    else:
        optimal_weights = pd.Series(result.x, index=returns.index)
    
    # Calcular métricas da carteira
    portfolio_return = np.sum(returns * optimal_weights)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    portfolio_sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    
    print(f"Fase 2 concluída. Sharpe Ratio da carteira otimizada: {portfolio_sharpe:.4f}")
    print(f"Retorno esperado: {portfolio_return:.4f}, Volatilidade: {portfolio_volatility:.4f}")
    
    return optimal_weights

def bayesian_parameter_refinement(initial_weights: pd.Series, 
                                 returns: pd.Series, 
                                 cov_matrix: pd.DataFrame,
                                 n_samples: int = 1000) -> pd.Series:
    """
    Fase 3: Refinamento Bayesiano dos pesos via simulação Monte Carlo
    Versão simplificada com menos amostras
    """
    print("Fase 3: Iniciando refinamento Bayesiano via simulação Monte Carlo...")
    
    # Incerteza nas estimativas de retorno (desvio padrão)
    # Assumimos que a incerteza é proporcional à volatilidade
    volatilities = np.sqrt(np.diag(cov_matrix))
    return_uncertainty = volatilities / np.sqrt(252 * 5)  # Baseado em 5 anos de dados diários
    
    # Gerar amostras dos retornos usando distribuição normal
    np.random.seed(42)
    return_samples = np.random.normal(
        loc=returns.values,
        scale=return_uncertainty,
        size=(n_samples, len(returns))
    )
    
    # Simulação de Monte Carlo
    all_weights = []
    all_returns = []
    all_volatilities = []
    all_sharpes = []
    
    for i in range(n_samples):
        # Usar retornos simulados
        sample_returns = pd.Series(return_samples[i], index=returns.index)
        
        # Otimização simplificada para cada amostra
        n_etfs = len(returns)
        
        # Função objetivo: maximizar Sharpe Ratio
        def objective(weights):
            weights = np.array(weights)
            portfolio_return = np.sum(sample_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
            return -sharpe_ratio
        
        # Restrições simplificadas
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Soma dos pesos = 1
        ]
        
        # Limites por ETF
        bounds = [(MIN_ALLOCATION, MAX_ALLOCATION) for _ in range(n_etfs)]
        
        # Otimização para esta amostra
        result = minimize(
            objective,
            initial_weights.values,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        if result.success:
            weights = result.x
            portfolio_return = np.sum(sample_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            portfolio_sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
            
            all_weights.append(weights)
            all_returns.append(portfolio_return)
            all_volatilities.append(portfolio_volatility)
            all_sharpes.append(portfolio_sharpe)
    
    # Selecionar as melhores carteiras (top 10% por Sharpe)
    all_results = pd.DataFrame({
        'return': all_returns,
        'volatility': all_volatilities,
        'sharpe': all_sharpes
    })
    
    top_indices = all_results['sharpe'].nlargest(int(n_samples * 0.1)).index
    top_weights = np.array([all_weights[i] for i in top_indices])
    
    # Calcular pesos médios das melhores carteiras
    refined_weights = pd.Series(
        top_weights.mean(axis=0),
        index=returns.index
    )
    
    # Ajustar pesos para somar 1
    refined_weights = refined_weights / refined_weights.sum()
    
    # Calcular métricas da carteira refinada
    portfolio_return = np.sum(returns * refined_weights)
    portfolio_volatility = np.sqrt(np.dot(refined_weights.T, np.dot(cov_matrix, refined_weights)))
    portfolio_sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    
    print(f"Fase 3 concluída. Sharpe Ratio após refinamento: {portfolio_sharpe:.4f}")
    print(f"Retorno esperado: {portfolio_return:.4f}, Volatilidade: {portfolio_volatility:.4f}")
    
    return refined_weights

def create_portfolio(etf_data: pd.DataFrame, weights: pd.Series) -> Dict:
    """
    Cria o portfolio final com alocações e métricas
    """
    # Verificar se os ETFs nos pesos existem nos dados
    valid_etfs = [etf for etf in weights.index if etf in etf_data.index]
    
    if len(valid_etfs) < len(weights):
        print(f"Aviso: {len(weights) - len(valid_etfs)} ETFs nos pesos não encontrados nos dados")
    
    # Filtrar ETFs válidos
    filtered_weights = weights.loc[valid_etfs]
    filtered_weights = filtered_weights / filtered_weights.sum()  # Normalizar
    
    # Criar portfolio
    portfolio = {
        'etfs': [],
        'total_weight': filtered_weights.sum(),
        'expected_return': 0,
        'expected_volatility': 0,
        'sharpe_ratio': 0,
        'geographic_allocation': {},
        'strategy_allocation': {}
    }
    
    # Adicionar cada ETF ao portfolio
    for etf_id, weight in filtered_weights.items():
        etf_info = etf_data.loc[etf_id].to_dict()
        etf_info['weight'] = weight
        portfolio['etfs'].append(etf_info)
        
        # Calcular alocações geográficas
        if 'domicile_country' in etf_info:
            country = etf_info['domicile_country']
            if country in portfolio['geographic_allocation']:
                portfolio['geographic_allocation'][country] += weight
            else:
                portfolio['geographic_allocation'][country] = weight
        
        # Calcular alocações por estratégia
        if 'strategy' in etf_info:
            strategy = etf_info['strategy']
            if strategy in portfolio['strategy_allocation']:
                portfolio['strategy_allocation'][strategy] += weight
            else:
                portfolio['strategy_allocation'][strategy] = weight
    
    # Calcular métricas do portfolio se houver retornos e covariância disponíveis
    if 'annual_return' in etf_data.columns and 'volatility' in etf_data.columns:
        portfolio['expected_return'] = sum(
            etf_data.loc[etf_id, 'annual_return'] * weight 
            for etf_id, weight in filtered_weights.items()
        )
        
        # Simplificação da volatilidade usando correlação média
        avg_correlation = 0.5
        weighted_vol_squared = 0
        
        for i, (etf_id1, weight1) in enumerate(filtered_weights.items()):
            vol1 = etf_data.loc[etf_id1, 'volatility']
            weighted_vol_squared += (weight1 * vol1) ** 2
            
            for j, (etf_id2, weight2) in enumerate(filtered_weights.items()):
                if j > i:  # Evitar duplicação
                    vol2 = etf_data.loc[etf_id2, 'volatility']
                    weighted_vol_squared += 2 * avg_correlation * weight1 * weight2 * vol1 * vol2
        
        portfolio['expected_volatility'] = np.sqrt(weighted_vol_squared)
        portfolio['sharpe_ratio'] = (
            portfolio['expected_return'] - RISK_FREE_RATE
        ) / portfolio['expected_volatility']
    
    return portfolio

def mock_etf_data():
    """Cria dados de ETF simulados para demonstração"""
    np.random.seed(42)
    n_etfs = 50  # Reduzido para simplificar
    
    # Simular histórico de preços
    def generate_price_history(annual_return, volatility, days=1260):  # 5 anos de dias úteis
        daily_return = annual_return / 252
        daily_vol = volatility / np.sqrt(252)
        
        # Gerar retornos diários
        daily_returns = np.random.normal(daily_return, daily_vol, days)
        
        # Converter para preços
        prices = [100]  # Preço inicial
        for r in daily_returns:
            prices.append(prices[-1] * (1 + r))
        
        return prices
    
    # Estratégias de ETF
    strategies = ['Value', 'Growth', 'Blend', 'Dividend']
    countries = ['US', 'EU', 'UK', 'Global']
    asset_classes = ['Equity', 'Fixed Income', 'Multi-Asset']
    replication_methods = ['Physical', 'Synthetic']
    
    # Criar DataFrame
    etfs = []
    
    for i in range(n_etfs):
        # Parâmetros simulados
        annual_return = np.random.normal(0.08, 0.04)  # Média 8%, desvio 4%
        volatility = np.random.normal(0.15, 0.05)  # Média 15%, desvio 5%
        
        # Garantir valores razoáveis
        volatility = max(0.05, volatility)
        
        # Gerar histórico de preços
        price_history = generate_price_history(annual_return, volatility)
        
        etf = {
            'name': f"ETF {i+1}",
            'ticker': f"ETF{i+1}",
            'strategy': np.random.choice(strategies),
            'domicile_country': np.random.choice(countries),
            'asset_class': np.random.choice(asset_classes),
            'replication_method': np.random.choice(replication_methods),
            'ter': np.random.uniform(0.0005, 0.01),  # 0.05% a 1%
            'age_years': np.random.uniform(1, 20),
            'fund_size_millions': np.random.uniform(10, 10000),
            'number_of_holdings': np.random.randint(20, 1000),
            'annual_return': annual_return,
            'volatility': volatility,
            'max_drawdown': calculate_max_drawdown(price_history),
            'last_five_years_return': (price_history[-1] / price_history[0]) ** (1/5) - 1,
            'last_five_years_volatility': calculate_volatility(price_history),
            'price_history': price_history
        }
        
        etfs.append(etf)
    
    return pd.DataFrame(etfs)

def run_etf_optimization_pipeline():
    """Função principal para executar o pipeline de otimização"""
    print("Iniciando pipeline de otimização de carteira de ETFs...")
    
    # Passo 1: Criar ou carregar dados
    etf_data = mock_etf_data()
    print(f"Dados carregados. Total de {len(etf_data)} ETFs.")
    
    # Passo 2: Pré-processamento
    processed_data = preprocess_data(etf_data)
    
    # Passo 3: Preparar features
    X, y = prepare_features(processed_data)
    
    # Passo 4: Ranquear ETFs
    top_etfs = rank_etfs_with_gradient_boosting(X, y, processed_data)
    
    # Passo 5: Calcular retornos e covariância
    returns, cov_matrix = calculate_returns_and_covariance(top_etfs)
    
    # Passo 6: Otimização de Markowitz
    weights = markowitz_optimization(returns, cov_matrix, top_etfs)
    
    # Passo 7: Refinamento Bayesiano (opcional)
    refined_weights = bayesian_parameter_refinement(weights, returns, cov_matrix)
    
    # Passo 8: Criar portfólio final
    portfolio = create_portfolio(top_etfs, refined_weights)
    
    # Exibir resultados
    print("\n--- Resultados da Otimização ---")
    print(f"Carteira otimizada com {len(portfolio['etfs'])} ETFs")
    print(f"Retorno esperado: {portfolio['expected_return']:.4f}")
    print(f"Volatilidade esperada: {portfolio['expected_volatility']:.4f}")
    print(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.4f}")
    
    print("\nAlocação geográfica:")
    for country, weight in portfolio['geographic_allocation'].items():
        print(f"  {country}: {weight:.2%}")
    
    print("\nAlocação por estratégia:")
    for strategy, weight in portfolio['strategy_allocation'].items():
        print(f"  {strategy}: {weight:.2%}")
    
    print("\nPesos dos ETFs na carteira otimizada:")
    for etf in portfolio['etfs']:
        print(f"  {etf['name']} ({etf['ticker']}): {etf['weight']:.2%}")
    
    return portfolio

# Execute o código
if __name__ == "__main__":
    portfolio = run_etf_optimization_pipeline()