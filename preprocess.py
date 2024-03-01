import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


def calculate_similarity(items_a, items_b):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_a = tfidf_vectorizer.fit_transform(items_a)
    tfidf_matrix_b = tfidf_vectorizer.transform(items_b)
    similarity_matrix = cosine_similarity(tfidf_matrix_a, tfidf_matrix_b)
    return similarity_matrix


def read_csv(file_path, important_fields):
    data = {}
    with open(file_path, newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item_id = row["id"] # n
            item_data = " ".join([row[field] for field in important_fields])
            data[item_id] = item_data
    return data

start = time.time()

abt_data = read_csv('base/Abt.csv', ['name', 'description'])
buy_data = read_csv('base/Buy.csv', ['name', 'description'])

abt_items = list(abt_data.values())
buy_items = list(buy_data.values()) 
similarity_matrix = calculate_similarity(abt_items, buy_items) # n^2

# Definição de um limiar de similaridade
# 0.8 - Precisão 100% 1.0, baixissima revocação 0.02, F-measure 0.04
# 0.7 - Precisão alta 0.91, revocação 0.11, F-measure 0.2
# 0.6 - Precisão 0.82, revocação 0.32, F-measure 0.46
# 0.5 - Precisão 0.60, revocação 0.56, F-measure 0.58 - Melhor resultado
# 0.55 - Precisão 0.71, revocação 0.44, F-measure 0.54
# 0.4 - Precisão 0.35, revocação 0.76, F-measure 0.48
threshold = 0.5

duplicates = []
for i, abt_item in enumerate(abt_items):
    for j, buy_item in enumerate(buy_items): # n^2 for de for
        if similarity_matrix[i][j] >= threshold: 
            duplicates.append(
                (list(abt_data.keys())[i], list(buy_data.keys())[j]))

with open('result.csv', mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['abt_id', 'buy_id'])
    for abt_id, buy_id in duplicates:
        writer.writerow([abt_id, buy_id])

validation_data = []
with open('base/abt_buy_perfectMapping.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        validation_data.append((row["idAbt"], row["idBuy"]))

true_positives = 0
for abt_id, buy_id in duplicates:
    if (abt_id, buy_id) in validation_data:
        true_positives += 1

precision = true_positives / len(duplicates) if len(duplicates) > 0 else 0
recall = true_positives / \
    len(validation_data) if len(validation_data) > 0 else 0
f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
end = time.time()
print(f"Precision: {precision}, Recall: {recall}, F-Measure: {f_measure}")

print(f"Tempo de execução: {end - start} segundos")