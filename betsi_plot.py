import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from sklearn.metrics import pairwise_distances

np.random.seed(42)

data = np.load('russian_nofraglit_SVD_dict.npy', allow_pickle=True).item()

embeddings = np.array(list(data.values()))

print(f'Количество эмбеддингов: {embeddings.shape[0]}')
print(f'Размерность эмбеддингов: {embeddings.shape[1]}')

percentage = 0.075
num_samples = int(len(embeddings) * percentage)

random_indices = np.random.choice(embeddings.shape[0], num_samples, replace=False)
embeddings_sampled = embeddings[random_indices]

print(f'Количество эмбеддингов после выборки (10%): {embeddings_sampled.shape[0]}')

from sklearn.preprocessing import normalize
embeddings_sampled = normalize(embeddings_sampled, norm='l2')

print('Вычисление матрицы косинусных расстояний...')
distance_matrix = pairwise_distances(embeddings_sampled, metric='cosine')

print('Вычисление персистентной гомологии...')
result = ripser(distance_matrix, distance_matrix=True, maxdim=1)
dgms = result['dgms']

max_epsilon = distance_matrix.max()

eps_values = np.linspace(0, 1, 1000)

def compute_betti_numbers(dgms, epsilon):
    betti = []
    for dim in range(len(dgms)):
        count = 0
        for interval in dgms[dim]:
            birth, death = interval
            if birth <= epsilon and (death > epsilon or np.isinf(death)):
                count += 1
        betti.append(count)
    return betti

betti_1 = []

print('Подсчёт чисел Бетти...')
for eps in eps_values:
    betti = compute_betti_numbers(dgms, eps)
    # Числа Бетти 0 и 1
    betti_1.append(betti[1] if len(betti) > 1 else 0)

plt.figure(figsize=(12, 8))
plt.plot(eps_values, betti_1, label='Betti 1')
plt.xlabel('ε (косинусное расстояние)')
plt.ylabel('Число Бетти')
plt.title('Числа Бетти 1 в зависимости от ε для русских эмбеддингов (10% выборка, SVD)')
plt.legend()
plt.grid(True)
plt.show()