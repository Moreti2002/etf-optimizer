# Otimizador de Carteira de ETFs

## Visão Geral

Este projeto implementa um pipeline completo para otimização de carteiras de ETFs (Exchange Traded Funds), utilizando técnicas de aprendizado de máquina e otimização matemática. O objetivo é selecionar um conjunto ótimo de ETFs e calcular a alocação ideal de capital entre eles, maximizando o retorno ajustado ao risco.

## Características Principais

- **Seleção Inteligente de ETFs**: Utiliza XGBoost para ranquear e selecionar os ETFs mais promissores
- **Otimização de Markowitz**: Encontra a alocação ótima baseada na teoria moderna de portfólio
- **Refinamento Bayesiano**: Aprimora os pesos através de simulação Monte Carlo para considerar incertezas
- **Restrições Personalizáveis**: Suporta limites de alocação por ETF, geografia e estratégia
- **Análise Completa**: Fornece métricas detalhadas de risco-retorno e visualização de alocação

## Requisitos

```
pandas>=1.3.0
numpy>=1.20.0
xgboost>=1.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/etf-optimizer.git
cd etf-optimizer

# Instale as dependências
pip install -r requirements.txt
```

## Como Usar

### Exemplo Básico

```python
import pandas as pd
from etf_optimizer import run_etf_optimization_pipeline

# Executar pipeline completo com dados simulados
portfolio = run_etf_optimization_pipeline()

# Ver resultados
print(f"Retorno esperado: {portfolio['expected_return']:.4f}")
print(f"Volatilidade esperada: {portfolio['expected_volatility']:.4f}")
print(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.4f}")
```

### Usando seus Próprios Dados

Para utilizar seus próprios dados, você precisa criar um DataFrame com informações sobre os ETFs. O DataFrame deve conter as seguintes colunas:

- **Obrigatórias**: `ticker`, `annual_return`, `volatility`
- **Recomendadas**: `strategy`, `domicile_country`, `ter` (taxa de despesa), `fund_size_millions`, `number_of_holdings`

```python
import pandas as pd
from etf_optimizer import preprocess_data, prepare_features, rank_etfs_with_gradient_boosting, calculate_returns_and_covariance, markowitz_optimization, bayesian_parameter_refinement, create_portfolio

# Carregar seus dados
etf_data = pd.read_csv('seus_dados_etf.csv')

# Executar o pipeline
processed_data = preprocess_data(etf_data)
X, y = prepare_features(processed_data)
top_etfs = rank_etfs_with_gradient_boosting(X, y, processed_data)
returns, cov_matrix = calculate_returns_and_covariance(top_etfs)
weights = markowitz_optimization(returns, cov_matrix, top_etfs)
refined_weights = bayesian_parameter_refinement(weights, returns, cov_matrix)
portfolio = create_portfolio(top_etfs, refined_weights)

# Acessar resultados
for etf in portfolio['etfs']:
    print(f"{etf['ticker']}: {etf['weight']:.2%}")
```

## Configuração

Você pode personalizar os parâmetros de otimização editando as constantes no início do arquivo `etf_optimizer.py`:

```python
# Configurações globais
TOP_N_ETFS = 15             # Número de ETFs a serem selecionados
RISK_FREE_RATE = 0.03       # Taxa livre de risco (3%)
MIN_ALLOCATION = 0.05       # Alocação mínima por ETF (5%)
MAX_ALLOCATION = 0.30       # Alocação máxima por ETF (30%)
GEOGRAPHIC_CONSTRAINTS = {'US': 0.6, 'EU': 0.3}  # Limites geográficos
STRATEGY_CONSTRAINTS = {'Value': 0.4, 'Growth': 0.3}  # Limites por estratégia
```

## Pipeline de Otimização

O processo completo de otimização segue estas etapas:

1. **Pré-processamento dos Dados**
   - Normalização de dados temporais
   - Encoding de variáveis categóricas
   - Tratamento de valores ausentes

2. **Seleção de ETFs via Machine Learning**
   - Preparação de features
   - Treinamento de modelo XGBoost
   - Ranqueamento e seleção dos melhores ETFs

3. **Cálculo de Retornos e Covariância**
   - Análise de séries temporais de preços
   - Estimativa de retornos esperados
   - Construção da matriz de covariância

4. **Otimização de Markowitz**
   - Maximização do Sharpe Ratio
   - Aplicação de restrições de alocação
   - Cálculo dos pesos ótimos iniciais

5. **Refinamento Bayesiano**
   - Simulação Monte Carlo
   - Incorporação de incertezas nas estimativas
   - Ajuste fino dos pesos

6. **Análise da Carteira Final**
   - Cálculo de métricas de performance
   - Visualização da alocação
   - Relatório detalhado

## Estrutura do Código

O código foi organizado de forma modular, com funções independentes para cada etapa do processo:

- `preprocess_data()`: Prepara os dados brutos dos ETFs
- `prepare_features()`: Formata as features para o modelo de ML
- `rank_etfs_with_gradient_boosting()`: Seleciona os melhores ETFs
- `calculate_returns_and_covariance()`: Estima retornos e covariância
- `markowitz_optimization()`: Executa a otimização de Markowitz
- `bayesian_parameter_refinement()`: Refina os pesos via simulação
- `create_portfolio()`: Constrói o portfólio final com métricas
- `run_etf_optimization_pipeline()`: Executa o pipeline completo

## Exemplo de Saída

```
Iniciando pipeline de otimização de carteira de ETFs...
Dados carregados. Total de 50 ETFs.
Iniciando pré-processamento dos dados...
Pré-processamento concluído. Shape final dos dados: (50, 64)
Preparando features para o modelo de classificação...
Features preparadas. X shape: (50, 37), y shape: (50,)
Fase 1: Iniciando ranqueamento de ETFs com Gradient Boosting...
Features mais importantes para o ranqueamento:
          feature  importance
5              ter    0.187246
0       volatility    0.143521
1     max_drawdown    0.137852
3  fund_size_millions    0.085327
4  number_of_holdings    0.073629
Fase 1 concluída. Selecionados 15 ETFs top.
Fase 2: Calculando retornos esperados e matriz de covariância...
Fase 2: Cálculo de retornos e covariância concluído.
Fase 2: Iniciando otimização de Markowitz...
Fase 2 concluída. Sharpe Ratio da carteira otimizada: 0.7213
Retorno esperado: 0.1092, Volatilidade: 0.1095
Fase 3: Iniciando refinamento Bayesiano via simulação Monte Carlo...
Fase 3 concluída. Sharpe Ratio após refinamento: 0.7224
Retorno esperado: 0.1095, Volatilidade: 0.1098

--- Resultados da Otimização ---
Carteira otimizada com 15 ETFs
Retorno esperado: 0.1095
Volatilidade esperada: 0.1098
Sharpe Ratio: 0.7224

Alocação geográfica:
  US: 37.82%
  EU: 26.31%
  Global: 23.94%
  UK: 11.94%

Alocação por estratégia:
  Value: 29.72%
  Growth: 25.63%
  Blend: 23.81%
  Dividend: 20.84%

Pesos dos ETFs na carteira otimizada:
  ETF 12 (ETF12): 8.23%
  ETF 7 (ETF7): 7.84%
  ETF 3 (ETF3): 7.65%
  ETF 42 (ETF42): 7.31%
  ETF 15 (ETF15): 6.92%
  ...
```