using NPZ
using LinearAlgebra
using Distances
using Ripserer
using Plots

# 1. Загрузка .npz-файла
data = npzread("russian_cleanlit_SVD_32_subset.npz")

# 2. Извлекаем эмбеддинги в матрицу (строки = эмбеддинги)
embedding_keys = collect(keys(data))
N = length(embedding_keys)
println("Всего эмбеддингов в подвыборке: ", N)

X = hcat([data[k] for k in embedding_keys]...)'  # Adjoint
X = Matrix(X)  # Преобразуем в обычную матрицу

println("Размер матрицы эмбеддингов: ", size(X))

# 3. Вычисляем персистентную гомологию (например, H₀ и H₁).
# Используем косинусную метрику (или другую).
diagram = ripserer(X; metric=CosineDist(), maxdim=1)

# 4. Выводим диаграмму рождения–смерти
p = plot(diagram)
display(p)
