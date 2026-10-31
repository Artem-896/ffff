import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

# 1. ПІДГОТОВКА ДАНИХ
num_samples = 1200 
X = np.random.rand(num_samples, 2)  

y = (X[:, 0] + X[:, 1] >= 1).astype(int)

split_idx = int(0.8 * num_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Розмір навчальної вибірки: {X_train.shape[0]} точок")
print(f"Розмір тестової вибірки: {X_test.shape[0]} точок\n")

# 2. СТВОРЕННЯ НЕЙРОННОЇ МЕРЕЖІ
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. НАВЧАННЯ МОДЕЛІ
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,  
    verbose=1
)

# 4. ОЦІНКА ТОЧНОСТІ НА ТЕСТОВІЙ ВИБІРЦІ
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n--- Оцінка якості ---")
print(f"Точність (Accuracy) на тестовій вибірці: {test_acc * 100:.2f}%")
print(f"Втрати (Loss) на тестовій вибірці: {test_loss:.4f}")

# 5. ПЕРЕДБАЧЕННЯ ДЛЯ ТОЧОК ТЕСТОВОЇ ВИБІРКИ
test_predictions = model.predict(X_test)
test_predicted_classes = (test_predictions > 0.5).astype(int).flatten()

print("\n--- Приклади передбачень для перших 5 точок з тестової вибірки ---")
for i in range(5):
    print(f"Точка: ({X_test[i][0]:.3f}, {X_test[i][1]:.3f}) | "
          f"Справжній клас: {y_test[i]} | "
          f"Передбачений клас: {test_predicted_classes[i]} "
          f"(Ймовірність: {test_predictions[i][0]:.4f})")

# 6. ПЕРЕВІРКА НА 5 НОВИХ ТОЧКАХ
new_points = np.array([
    [0.2, 0.3], 
    [0.8, 0.4], 
    [0.5, 0.5],  
    [0.1, 0.1], 
    [0.7, 0.6]   
])

new_predictions = model.predict(new_points)
new_predicted_classes = (new_predictions > 0.5).astype(int).flatten()

print("\n--- Перевірка на 5 нових точках ---")
for i, point in enumerate(new_points):
    true_class = 1 if (point[0] + point[1] >= 1) else 0
    print(f"Нова точка: ({point[0]}, {point[1]}) | "
          f"Справжній клас: {true_class} | "
          f"Передбачений клас: {new_predicted_classes[i]} "
          f"(Ймовірність: {new_predictions[i][0]:.4f})")

# 7. ПОБУДОВА ГРАФІКІВ ВТРАТ ТА ТОЧНОСТІ
plt.figure(figsize=(14, 5))

# Графік Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Навчальна вибірка (Loss)', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Валідаційна вибірка (Loss)', color='orange', linewidth=2)
plt.title('Зміна функції втрат (Loss) протягом навчання')
plt.xlabel('Епоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Графік Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Навчальна вибірка (Accuracy)', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Валідаційна вибірка (Accuracy)', color='orange', linewidth=2)
plt.title('Зміна точності (Accuracy) протягом навчання')
plt.xlabel('Епоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()