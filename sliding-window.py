import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time

def getIdFromItem(item, dataset):
    for key, value in dataset.items():
        if value == item:
            return key
    return None

def sliding_window(sequence, window_size):
    result = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        result.append(window)
    return result


def read_csv(file_path, important_fields):
    data = {}
    with open(file_path, newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item_id = row["id"] # n
            item_data = " ".join([row[field] for field in important_fields])
            data[item_id] = item_data
    return data

def calculate_similarity(items_a, items_b):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_a = tfidf_vectorizer.fit_transform(items_a)
    tfidf_matrix_b = tfidf_vectorizer.transform(items_b)
    similarity_matrix = cosine_similarity(tfidf_matrix_a, tfidf_matrix_b)
    return similarity_matrix

start = time.time()

abt_data = read_csv('base/Abt.csv', ['name', 'description'])
buy_data = read_csv('base/Buy.csv', ['name', 'description'])
abt_items = list(abt_data.values())
buy_items = list(buy_data.values()) 

abt_records = sorted(abt_items, key=lambda x: x[0])
buy_records = sorted(buy_items, key=lambda x: x[0])

win_size = 40
abt_blocks = sliding_window(abt_records, win_size)
buy_blocks = sliding_window(buy_records, win_size)

duplicates = []

tam = len(abt_blocks)
tam2 = len(buy_blocks)
if(tam < tam2):
    x = tam
    dif = tam2 - tam
else:
    x = tam2

tg = 0.5
for i in range(x): # for pelo menor bloco
    similarity_matrix = calculate_similarity(abt_blocks[i], buy_blocks[i])
    for k in range(len(similarity_matrix)):
        for l in range(len(similarity_matrix[k])):
            if similarity_matrix[k][l] > tg:
                id = getIdFromItem(abt_blocks[i][k], abt_data)
                idb = getIdFromItem(buy_blocks[i][l], buy_data)
                if(([id,idb]) not in duplicates):
                    duplicates.append(([id,idb]))

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
