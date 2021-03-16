"""
https://keras.io/examples/vision/mnist_convnet/
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
# Model / data parameters

N_samples = 1000
max_height = 10
max_weight = 1
g = 9.82

def get_test_train_data():
    h = np.random.uniform(1, max_height, N_samples)
    # h_noise = h + add_noise
    # m = np.random.rand(N_samples)*max_weight
    # 1/2 mv2 = mgh
    # v = np.sqrt(2*g*h)
    t_noise = np.random.normal(0, 0.1, N_samples)
    t = fun(h,g)+t_noise
    # plt.show()

    a, a_cov = fit_mechanistic(h, t)
    h_test = np.arange(1,max_height, 0.1)

    t_true = fun(h_test, g)
    
    f = fun(h_test, a)
    model = fit_neural(h, t)
    f_nn = model.predict(h_test)

    rmse = np.sqrt(((f-t_true)**2).sum()/len(h_test))
    rmse_nn = np.sqrt(((f_nn-t_true)**2).sum()/len(h_test))

    plt.scatter(h, t, s = 2, facecolors='none', edgecolors='b')
    plt.plot(h_test, f, lw = 2, c = "r", label = "Mechanistic model, rmse: {0:.2e}".format(rmse))
    plt.plot(h_test, f_nn, lw = 2, c = "g", label = "Deep learning, rmse: {0:.2e}".format(rmse_nn))
    plt.legend()
    plt.ylabel("Time [s]")
    plt.xlabel("Drop height [m]")
    plt.figtext(0.05, 0.05, "Estimated g: {0:.3f}".format(a))
    plt.show()

    print(rmse)
    print(rmse_nn)

def fun(h, a):
    return np.sqrt(2*h/a)

def fit_mechanistic(h, t_train):
    M = np.vstack([np.sqrt(h), np.zeros(len(h))]).T
    # M = np.vstack([np.sqrt(h), np.ones(len(h))]).T
    m, cov = np.linalg.lstsq(M, t_train)[:2]
    print(m)
    a = 2/m**2
    print(a)
    return a[0], cov

def fit_neural(h, t_train):
    normalizer = preprocessing.Normalization(input_shape = [1,])
    model = keras.Sequential([
        normalizer,
        layers.Dense(units=16, activation = "sigmoid"),
        layers.Dense(units=16, activation = "sigmoid"),
        layers.Dense(units=16, activation = "sigmoid"),
        layers.Dense(units=16, activation = "sigmoid"),
        layers.Dense(units=1)])
    model.summary()
    model.compile(
    optimizer="adam",
    loss='mean_absolute_error')
    history = model.fit(h, t_train, epochs=100,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split = 0.2)
    # plot_loss(history)
    return model

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()



# def get_test_train_data():
#     # the data, split between train and test sets
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#     # Scale images to the [0, 1] range
#     x_train = x_train.astype("float32") / 255
#     x_test = x_test.astype("float32") / 255
#     # Make sure images have shape (28, 28, 1)
#     x_train = np.expand_dims(x_train, -1)
#     x_test = np.expand_dims(x_test, -1)
#     print("x_train shape:", x_train.shape)
#     print(x_train.shape[0], "train samples")
#     print(x_test.shape[0], "test samples")


#     # convert class vectors to binary class matrices
#     y_train = keras.utils.to_categorical(y_train, num_classes)
#     y_test = keras.utils.to_categorical(y_test, num_classes)
#     return x_train, y_train, x_test, y_test

def create_model():

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def run():
    x_train, y_test, x_test, y_test = get_test_train_data()
    model = create_model()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == '__main__':
    # run()
    get_test_train_data()