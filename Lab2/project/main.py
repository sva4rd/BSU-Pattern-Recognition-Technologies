from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

import tensorflow as tf


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, num_rbf_units, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.sigmas = None
        self.centers = None
        self.num_rbf_units = num_rbf_units

    def build(self, input_shape):
        self.centers = self.add_weight(shape=(self.num_rbf_units, input_shape[1]),
                                       initializer='uniform',
                                       trainable=True,
                                       name='centers')
        self.sigmas = self.add_weight(shape=(self.num_rbf_units, 784),
                                      initializer='ones',
                                      trainable=True,
                                      name='sigmas')
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        normalized_inputs = tf.math.l2_normalize(inputs, axis=-1)
        squared_diff = tf.square(normalized_inputs[:, None, :] - self.centers)
        rbf_out = tf.exp(-0.5 * tf.reduce_sum(squared_diff / tf.square(self.sigmas), axis=-1))
        return rbf_out


# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка данных
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# Функция для построения модели радиально-базисных функций
def build_model(num):
    model = tf.keras.Sequential()
    model.add(RBFLayer(num))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.build(input_shape=(None, 784))

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Список для сохранения результатов точности
accuracy_results = []

# Варьирование количества нейронов в скрытом слое
num_neurons_list = [5, 10, 15, 20, 25]
for num_neurons in num_neurons_list:
    # Построение модели
    model = build_model(num_neurons)

    # Обучение модели
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    # Оценка точности на тестовой выборке
    _, accuracy = model.evaluate(x_test, y_test)
    accuracy_results.append(accuracy)

# Построение графика зависимости точности от количества нейронов
plt.plot(num_neurons_list, accuracy_results, 'o-')
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Neurons')
plt.show()
