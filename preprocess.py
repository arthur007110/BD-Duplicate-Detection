from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Carregar os dados dos arquivos CSV
buy = pd.read_csv('base/Buy.csv')
abt = pd.read_csv('base/Abt.csv')

# Preencher valores nulos na coluna de preço com a média
buy['price'] = pd.to_numeric(buy['price'], errors='coerce')
buy['price'].fillna(buy['price'].mean(), inplace=True)
abt['price'] = pd.to_numeric(abt['price'], errors='coerce')
abt['price'].fillna(abt['price'].mean(), inplace=True)

# Substituir valores nulos nas colunas de texto por uma string vazia
buy['name'].fillna('', inplace=True)
buy['description'].fillna('', inplace=True)
abt['name'].fillna('', inplace=True)
abt['description'].fillna('', inplace=True)

# Tokenização e Vetorização para Buy.csv
tfidf_vectorizer_buy = TfidfVectorizer()
tfidf_matrix_buy = tfidf_vectorizer_buy.fit_transform(
    buy['name'] + ' ' + buy['description'])

# Cálculo da Similaridade de Cosseno para Buy.csv
cosine_sim_buy = cosine_similarity(tfidf_matrix_buy, tfidf_matrix_buy)

# Definição de um limiar de similaridade
threshold = 0.8

# Identificação de Duplicatas em Buy.csv
duplicates_buy = []
for i in range(len(cosine_sim_buy)):
    for j in range(i+1, len(cosine_sim_buy[i])):
        if cosine_sim_buy[i][j] > threshold:
            duplicates_buy.append(
                (buy.iloc[i]['name'], buy.iloc[i]['description'], buy.iloc[j]['name'], buy.iloc[j]['description']))

# Tokenização e Vetorização para Abt.csv
tfidf_vectorizer_abt = TfidfVectorizer()
tfidf_matrix_abt = tfidf_vectorizer_abt.fit_transform(
    abt['name'] + ' ' + abt['description'])

# Cálculo da Similaridade de Cosseno para Abt.csv
cosine_sim_abt = cosine_similarity(tfidf_matrix_abt, tfidf_matrix_abt)

# Identificação de Duplicatas em Abt.csv
duplicates_abt = []
for i in range(len(cosine_sim_abt)):
    for j in range(i+1, len(cosine_sim_abt[i])):
        if cosine_sim_abt[i][j] > threshold:
            duplicates_abt.append(
                (abt.iloc[i]['name'], abt.iloc[i]['description'], abt.iloc[j]['name'], abt.iloc[j]['description']))

# Função para exibir um par de duplicatas de forma mais legível


def print_pair(pair):
    print("Duplicata 1:")
    print("Nome: " + pair[0])
    print("Descrição: " + pair[1])
    print("\n")
    print("Duplicata 2:")
    print("Nome: " + pair[2])
    print("Descrição: " + pair[3])
    print("\n")


# Visualização das Duplicatas em Buy.csv
print("Duplicatas em Buy.csv Identificadas:")
for pair in duplicates_buy:
    print_pair(pair)

# Visualização das Duplicatas em Abt.csv
print("\nDuplicatas em Abt.csv Identificadas:")
for pair in duplicates_abt:
    print_pair(pair)
