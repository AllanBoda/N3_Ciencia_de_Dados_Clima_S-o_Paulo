import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os

# Definindo os nomes das colunas (inferidos a partir da estrutura do CSV)
COLUMN_NAMES = [
    'Datetime', 'Station', 'Benzene', 'CO', 'PM10', 'PM2.5', 'NO2', 'O3', 'SO2', 'Toluene', 'TRS'
]

def load_data(data_path):
    """Carrega o dataset e aplica amostragem para evitar problemas de memória."""
    df = pd.read_csv(
        data_path,
        names=COLUMN_NAMES,
        header=0 # O arquivo tem cabeçalho, mas foi carregado sem na leitura inicial. Usar header=0 para tratar corretamente.
    )
    
    # Amostragem: Reduzir o dataset para 10% para garantir a execução em ambientes com recursos limitados
    # O uso de frac=0.1 é para fins de demonstração e reprodutibilidade em ambientes restritos.
    df_sampled = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"Dataset original: {len(df)} linhas. Dataset amostrado: {len(df_sampled)} linhas (10%).")
    
    return df_sampled

def preprocess_data(df):
    """Realiza a imputação e codificação One-Hot Encoding."""
    
    # Variável alvo
    target = 'PM2.5'
    
    # Features: todas exceto Datetime e a própria PM2.5
    features = [col for col in df.columns if col not in ['Datetime', target]]
    
    X = df[features]
    y = df[target]

    # Identificar colunas numéricas e categóricas
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Imputação de valores ausentes (se houver)
    imputer_numeric = SimpleImputer(strategy='median')
    X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])

    imputer_categorical = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])

    # Codificação de Variáveis Categóricas (One-Hot Encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    return X_encoded, y

def train_and_evaluate(X, y):
    """Treina e avalia múltiplos modelos, retornando o melhor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }

    results = {}
    best_rmse = float('inf')
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    print("\n--- Resultados da Avaliação Comparativa ---")
    results_df = pd.DataFrame(results).T
    print(results_df.sort_values(by='RMSE'))
    print(f"\nMelhor modelo: {best_model_name} com RMSE de {best_rmse:.4f}")
    
    return best_model, best_model_name, X_test.columns

def save_model(model, filename):
    """Salva o modelo treinado."""
    joblib.dump(model, filename)
    print(f"Modelo salvo como: {filename}")

if __name__ == "__main__":
    # Caminho para o arquivo de dados
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sp_air_quality_clean.csv')
    
    # Caminho para salvar o modelo (na raiz do projeto)
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'modelo_final.pkl')

    print("Iniciando o pipeline de dados (Regressão de PM2.5)...")
    
    # 1. Carregar e Pré-processar
    df = load_data(data_path)
    X_encoded, y = preprocess_data(df)
    
    # 2. Treinar e Avaliar
    best_model, best_model_name, feature_names = train_and_evaluate(X_encoded, y)
    
    # 3. Salvar o Modelo
    save_model(best_model, model_save_path)
    
    print("Pipeline de dados concluído com sucesso.")
