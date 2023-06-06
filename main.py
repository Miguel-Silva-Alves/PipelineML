# lib para leitura dos dados
import pandas as pd
# funções transformação dos dados para input do modelo
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# modelo de classificação
from sklearn.svm import LinearSVC
# funções para construção do pipeline
from sklearn.pipeline import Pipeline
# Transform columns
from Transform import TColumns



# Carregando Dataframe
df = pd.read_csv('https://raw.githubusercontent.com/dadosaocubo/nlp/master/base_mercadologica.csv')

# Stop Words to be removed
stop_words = ['em','sao','ao','de','da','do','para','c',
              'kg','un','ml','pct','und','das','no','ou',
              'pc','gr','pt','cm','vd','com','sem','gfa',
              'jg','la','1','2','3','4','5','6','7','8',
              '9','0','a','b','c','d','e','lt','f','g',
              'h','i','j','k','l','m','n','o','p','q',
              'r','s','t','u','v','x','w','y','z']

# Criando uma instância do transformador das colunas
tco = TColumns()
 
# Criando uma instância do CountVectorizer
cvt = CountVectorizer(strip_accents='ascii', lowercase=True, stop_words=stop_words)
 
# Criando uma instância do TfidfTransformer
tfi = TfidfTransformer(use_idf=True)
 
# Criando uma instância do modelo LinearSVC
clf = LinearSVC()
 
# Criando a Pipeline, adicionando o nosso transformador seguido de um modelo de classificação
skl_pipeline = Pipeline(steps=[('Transformer', tco),
                              ('CountVectorizer', cvt),
                              ('TfidfTransformer', tfi),
                              ('Model', clf)])

# Executando Pipeline
entrada = df[['descricao']]
saida = df['departamento']
skl_pipeline.fit(entrada, saida)

# loop para utilização do pipeline
fim = '0'
while (fim != '-1'):
  descricao = input('Informe o item para classificar ou -1 para encerrar o programa: ')
  fim = descricao
  if fim != '-1':
    df_predict = pd.DataFrame([descricao], columns=['descricao'])
    print('O item {} está na seção {}\n'.format(descricao.upper(), skl_pipeline.predict(df_predict)[0]))
  else:
    print('Obrigado! Volte sempre!!!')