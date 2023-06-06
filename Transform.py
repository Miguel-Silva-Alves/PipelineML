from sklearn.base import BaseEstimator, TransformerMixin

# Um transformador para colunas
class TColumns(BaseEstimator, TransformerMixin):
  # Função de fit ds dados de entrada
  def fit(self, X, y=None):
    return self
  # Função de transformação dos dados de entrada
  def transform(self, X):
    # Primeiro realizamos a cópia do DataFrame 'X' de entrada
    data = X.copy()
    data['descricao'] = data['descricao'].str.replace('[,.:;!?]+', ' ', regex=True).copy()
    data['descricao'] = data['descricao'].str.replace('[/<>()|\+\-\$%&#@\'\"]+', ' ', regex=True).copy()
    data['descricao'] = data['descricao'].str.replace('[0-9]+', '', regex=True)
    # Retornamos um novo dataframe com as colunas
    return data.descricao