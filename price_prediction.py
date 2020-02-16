import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers

from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv', index_col=0)
del data['Metro']
del data['DateUpdate']
data_train, data_test = train_test_split(data, shuffle=True)
y_train = data_train['Price_USD']
y_test = data_test['Price_USD']
del data_train['Price_USD']
del data_test['Price_USD']


def normalize(train_data, test_data):
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std
    return train_data, test_data, mean, std

x_train, x_test, mean, std = normalize(data_train, data_test)
partial_train_data = x_train[:130000]
val_data = x_train[130000:]
partial_train_targets = y_train[:130000]
val_targets = y_train[130000:]


class Model:
    def __init__(self):
        self.mae = object
        self.epochs = 89
        self.model = object
        self.fit_model = object
        self.batch_size = 256

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
        self.model = model

    def fit_validation(self, model):
        validation_history = model.fit(partial_train_data, partial_train_targets,
                                       validation_data=(val_data, val_targets),
                                       epochs=self.epochs, batch_size=self.batch_size)
        return validation_history

    def correct_epochs(self, validation_history):
        self.epochs = np.argmin(validation_history.history['val_mae']) + 1

    def fit(self, model):
        model.fit(x_train, y_train,
                  epochs=self.epochs,
                  batch_size=self.batch_size)

    def test(self, model):
        _, self.mae = model.evaluate(x_test, y_test)

    def predict(self, model):
        print('Enter your params!\n')
        age = float(input('Age: '))
        boobs = float(input('Boobs: '))
        height = float(input('Height: '))
        size = float(input('Size: '))
        weight = float(input("Weight: "))
        data = pd.DataFrame({"Age": [age], "Boobs": [boobs], "Height": [height], "Size": [size], "Weight": [weight]})
        data -= mean
        data /= std
        price = model.predict(data)
        return price[0]


class Plotter(Model):

    def validation_plot(self, validation_history):
        mae = validation_history.history['mae']
        val_mae = validation_history.history['val_mae']
        epochs = range(1, self.epochs + 1)
        plt.plot(epochs, mae, 'b', label='Training mean absolute error')
        plt.plot(epochs, val_mae, 'g', label='Validation mean absolute error')
        plt.xlabel('Epochs')
        plt.ylabel('Dollars_USD')
        plt.legend()
        plt.show()

    @staticmethod
    def validation_plot_smooth(validation_history, factor=0.9):
        smoothed_points = []
        points = validation_history.history['val_mae']
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        smoothed_points = smoothed_points[10:]
        plt.plot(range(1, len(smoothed_points) + 1), smoothed_points, 'g', label='Validation mean absolute error')
        plt.xlabel('Epochs')
        plt.ylabel('Dollars_USD')
        plt.show()

"""
model = Model()
model.build_model()
model.fit(model.model)
model.test(model.model)
price = model.predict(model.model)
print(f"Your possible price: {price[0]}$")
"""