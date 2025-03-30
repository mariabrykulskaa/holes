#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import random
import pandas as pd
from collections import Counter
from tqdm import tqdm

import chardet  # для авто-определения кодировки
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

###############################################################################
# 1. ЗАГРУЗКА «ДЫР» ИЗ ru_holes.npz
###############################################################################
HOLES_PATH = "/Users/maria/Desktop/stb-tda/holes/ru_holes.npz"
data = np.load(HOLES_PATH, allow_pickle=True)

best_7_cycles = data["best_7_cycles_embeddings"]  # (7, max_len, embedding_dim)

final_hole_embeddings = []
final_hole_centers = []

for i in range(len(best_7_cycles)):
    hole_3d = best_7_cycles[i]
    valid_rows = ~np.all(hole_3d == 0, axis=1)
    hole_2d = hole_3d[valid_rows]
    final_hole_embeddings.append(hole_2d)
    center = hole_2d.mean(axis=0)
    final_hole_centers.append(center)

final_hole_centers = np.array(final_hole_centers)  # (7, embedding_dim)

###############################################################################
# 2. ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ ПРЕДОБУЧЕННОЙ МОДЕЛИ (НАПР. CBoW)
###############################################################################
CBOW_PATH = "/Users/maria/Desktop/stb-tda/notebooks/ru_cbow_dictionary.npy"
cbow_dict = np.load(CBOW_PATH, allow_pickle=True).item()
print("CBOW-словарь загружен. Размер словаря:", len(cbow_dict))

###############################################################################
# 3. ФУНКЦИИ ДЛЯ ВЫЧЛЕНЕНИЯ ЭМБЕДДИНГОВ ИЗ СПИСКА СЛОВ (ЧТОБЫ РАБОТАТЬ С ЧАНКАМИ)
###############################################################################
def get_word_space_from_words(model, words):
    """
    Принимает на вход список токенов (words).
    Возвращает np.array эмбеддингов только для тех слов, которые есть в словаре model.
    """
    filtered_words = [w for w in words if w in model]
    if len(filtered_words) == 0:
        return np.empty((0, len(next(iter(model.values())))))
    # Можно взять все слова без уникализации, если нужно учитывать повторения.
    unique_words = set(filtered_words)
    return np.vstack([model[w] for w in unique_words])

###############################################################################
# 3.1. УПРОЩЁННАЯ ФУНКЦИЯ ОПРЕДЕЛЕНИЯ КОДИРОВКИ (ЧИТАЕТ ТОЛЬКО ЧАСТЬ ФАЙЛА)
###############################################################################
def detect_encoding(filename, read_limit=1_000_000):
    """
    Читает первые read_limit байт (по умолчанию 1 MB) и пытается определить кодировку.
    Если chardet возвращает MacRoman или None, используем 'utf-8'.
    """
    with open(filename, 'rb') as f:
        rawdata = f.read(read_limit)  # NEW: ограниченное чтение
    result = chardet.detect(rawdata)
    detected = result['encoding']
    if detected is None or detected.lower() in ['macroman', 'mac-roman']:
        return 'utf-8'
    return detected

###############################################################################
# 3.2. ГЕНЕРАТОР ЧАНКОВ — ПОСТРОЧНОЕ ЧТЕНИЕ (ЧТОБЫ НЕ ГРУЗИТЬ ВСЁ В ПАМЯТЬ)
###############################################################################
def yield_chunks(filename, chunk_size=2000):
    """
    Генератор, который читает файл построчно, копит слова в память,
    и выдает чанки примерно по chunk_size слов, не загружая файл целиком.
    """
    enc = detect_encoding(filename)
    chunk = []
    with open(filename, 'r', encoding=enc, errors='replace') as f:
        for line in f:
            words_in_line = line.split()
            for word in words_in_line:
                chunk.append(word)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

###############################################################################
# 4. ФУНКЦИИ ДЛЯ РАСЧЁТА РАССТОЯНИЙ И ПРИЗНАКОВ
###############################################################################
def get_dist_to_centers_array(objects_embs, hole_centers):
    dist_matrix = pairwise_distances(objects_embs, hole_centers, metric='cosine')
    avg_col = dist_matrix.mean(axis=1).reshape(-1, 1)
    return np.hstack([dist_matrix, avg_col])

def get_dist_array(objects_embs, holes_list, apply_func=np.min):
    dist_list = []
    for hole_emb in holes_list:
        dm = pairwise_distances(objects_embs, hole_emb, metric='cosine')
        dist_vec = apply_func(dm, axis=1)
        dist_list.append(dist_vec)
    dist_matrix = np.vstack(dist_list).T
    avg_col = dist_matrix.mean(axis=1).reshape(-1, 1)
    return np.hstack([dist_matrix, avg_col])

def get_most_common_closest_hole(min_dist_matrix):
    closest_indices = min_dist_matrix.argmin(axis=1)
    cnt = Counter(closest_indices)
    counts = np.array([cnt[h] for h in range(min_dist_matrix.shape[1])], dtype=float)
    counts /= len(closest_indices)
    best_hole = float(counts.argmax())
    return np.hstack([counts, best_hole])

def get_tda_based_features_for_text(objects_embs, hole_centers, hole_embs):
    """
    8 признаков:
      F1: усреднённое (по объектам) расстояние до каждого центра дыр (H значений)
      F2: усреднённое (по объектам и дыркам) расстояние до центров (1 значение)
      F3: усреднённое минимальное расстояние до каждой дыры (H значений)
      F4: усреднённое (по объектам и дыркам) минимальное расстояние (1 значение)
      F5: усреднённое максимальное расстояние до каждой дыры (H значений)
      F6: усреднённое (по объектам и дыркам) максимальное расстояние (1 значение)
      F7: доля объектов, для которых данная дыра – ближайшая (H значений)
      F8: индекс дыры, которая «победила» по количеству ближайших объектов (1 значение)
    """
    H = len(hole_embs)
    if len(objects_embs) == 0:
        return np.zeros((3*H + 3 + H + 1,), dtype=float)
    
    # 1) и 2) Расстояния до центров
    dist2centers = get_dist_to_centers_array(objects_embs, hole_centers)  # (N, H+1)
    F1 = dist2centers[:, :H].mean(axis=0)  # (H,)
    F2 = dist2centers[:, -1].mean()        # скаляр

    # 3) и 4) минимальное расстояние до дыр
    min_dist_mat = get_dist_array(objects_embs, hole_embs, apply_func=np.min)  # (N, H+1)
    F3 = min_dist_mat[:, :H].mean(axis=0)  # (H,)
    F4 = min_dist_mat[:, -1].mean()        # скаляр

    # 5) и 6) максимальное расстояние до дыр
    max_dist_mat = get_dist_array(objects_embs, hole_embs, apply_func=np.max)  # (N, H+1)
    F5 = max_dist_mat[:, :H].mean(axis=0)  # (H,)
    F6 = max_dist_mat[:, -1].mean()        # скаляр

    # 7) и 8) доля, для какой дыры данный объект ближе, и индекс «победителя»
    distribution = get_most_common_closest_hole(min_dist_mat[:, :H])
    F7 = distribution[:H]  # (H,)
    F8 = distribution[-1]  # скаляр

    features = np.concatenate([F1, [F2], F3, [F4], F5, [F6], F7, [F8]], axis=0)
    return features

###############################################################################
# 5. ФОРМИРУЕМ НАБОР ДАННЫХ (ОГРАНИЧИВАЕМ ЧАНКИ НА ФАЙЛ)
###############################################################################
# Файлы для ботов (AI)
bot_files = {
    "balaboba": "/Users/maria/Desktop/stb-tda/semantic_space/bots/RU/russian_bigbalaboba_corpus.txt",
    "mGPT":      "/Users/maria/Desktop/stb-tda/semantic_space/bots/RU/russian_bigmGPT_corpus.txt",
    "lstm":      "/Users/maria/Desktop/stb-tda/semantic_space/bots/RU/russian_newlstm_corpus.txt"
}
# Файл для литературы (HUMAN)
lit_file = "/Users/maria/Desktop/stb-tda/semantic_space/lit/russian_newlit_corpus.txt"

CHUNK_SIZE = 100        # примерный размер чанка
MAX_CHUNKS_PER_FILE = 1875  # максимальное число чанков, собираемых с каждого файла
MAX_TOTAL_CHUNKS = 7500     # итоговое ограничение (при условии 4 файлов по 1875)

records = []

# 5.1. Собираем чанки для ботов (AI)
for bot_name, path in bot_files.items():
    print(f"\n=== Обработка текстов бота {bot_name} ===")
    file_count = 0
    for chunk_idx, words_chunk in enumerate(tqdm(yield_chunks(path, chunk_size=CHUNK_SIZE), 
                                                  desc=f"Чтение чанков {bot_name}")):
        if file_count >= MAX_CHUNKS_PER_FILE:
            break
        objects_embs = get_word_space_from_words(cbow_dict, words_chunk)
        if objects_embs.shape[0] > 10000:
            idx = np.random.choice(objects_embs.shape[0], 10000, replace=False)
            objects_embs = objects_embs[idx]

        feats = get_tda_based_features_for_text(objects_embs, final_hole_centers, final_hole_embeddings)
        row = {"label": "AI", "source": bot_name, "chunk_idx": chunk_idx}
        for i, val in enumerate(feats, start=1):
            row[f"f{i}"] = val
        records.append(row)
        file_count += 1

# 5.2. Собираем чанки для литературы (HUMAN)
print(f"\n=== Обработка текстов людей (литература) ===")
file_count = 0
for chunk_idx, words_chunk in enumerate(tqdm(yield_chunks(lit_file, chunk_size=CHUNK_SIZE),
                                             desc="Чтение чанков литературы")):
    if file_count >= MAX_CHUNKS_PER_FILE:
        break
    objects_embs = get_word_space_from_words(cbow_dict, words_chunk)
    if objects_embs.shape[0] > 10000:
        idx = np.random.choice(objects_embs.shape[0], 10000, replace=False)
        objects_embs = objects_embs[idx]

    feats = get_tda_based_features_for_text(objects_embs, final_hole_centers, final_hole_embeddings)
    row = {"label": "HUMAN", "source": "literature", "chunk_idx": chunk_idx}
    for i, val in enumerate(feats, start=1):
        row[f"f{i}"] = val
    records.append(row)
    file_count += 1

print(f"\nСобрано чанков: {len(records)} (из возможных до {MAX_TOTAL_CHUNKS}).")

# 5.3. Перемешиваем и ограничиваем итоговое число чанков, если их больше MAX_TOTAL_CHUNKS
random.shuffle(records)
records = records[:MAX_TOTAL_CHUNKS]

df = pd.DataFrame(records)
print("Пример строк таблицы признаков (по чанкам):")
print(df.head())
print(f"Всего чанков в выборке (итог): {len(df)}")

# Целевая переменная: 1 – HUMAN, 0 – AI
X = df.drop(["label", "source", "chunk_idx"], axis=1).values
y = (df["label"] == "HUMAN").astype(int)

# Делим выборку вручную: первые 5000 в train, оставшиеся 2500 в test
train_size = 5000
test_size = 2500

if len(X) < train_size + test_size:
    raise ValueError("Собрано чанков меньше, чем нужно для заданных 5000 (train) и 2500 (test).")

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:train_size + test_size]
y_test = y[train_size:train_size + test_size]

###############################################################################
# 6. ОБУЧЕНИЕ И ОЦЕНКА КЛАССИФИКАТОРОВ
###############################################################################
print("\n=== SVC (метод опорных векторов) ===")
svc = SVC(kernel='linear', probability=True, random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_scores = svc.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_scores))
print("AvgPrec :", average_precision_score(y_test, y_scores))

print("\n=== Decision Tree (дерево решений) ===")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_scores = dt.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_scores))
print("AvgPrec :", average_precision_score(y_test, y_scores))

print("\n=== Random Forest (случайный лес) ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_scores = rf.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_scores))
print("AvgPrec :", average_precision_score(y_test, y_scores))

print("\nОбработка завершена.")
