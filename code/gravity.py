"""
https://keras.io/examples/vision/mnist_convnet/
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Model / data parameters

N_samples = 1000
max_height = 10
max_weight = 1
g = 9.82

def run(n_samples, plot = False):
    h = np.random.uniform(1, max_height, n_samples)
    # h_noise = h + add_noise
    # m = np.random.rand(n_samples)*max_weight
    # 1/2 mv2 = mgh
    # v = np.sqrt(2*g*h)
    t_noise = np.random.normal(0, 0.1, n_samples)
    t = fun(h,g)+t_noise
    # plt.show()

    a, a_cov = fit_mechanistic(h, t)
    model = fit_neural(h, t)


    h_test = np.arange(1,max_height, 0.1)
    t_true = fun(h_test, g)
    
    f = fun(h_test, a)
    f_nn = model.predict(h_test).flatten()


    rmse = np.sqrt(((f-t_true)**2).sum()/len(h_test))
    rmse_nn = np.sqrt(((f_nn-t_true)**2).sum()/len(h_test))

    if plot:

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
    return rmse, rmse_nn

def fun(h, a):
    return np.sqrt(2*h/a)

def fit_mechanistic(h, t_train):
    M = np.vstack([h, np.zeros(len(h))]).T
    # M = np.vstack([np.sqrt(h), np.zeros(len(h))]).T
    # M = np.vstack([np.sqrt(h), np.ones(len(h))]).T
    m, cov = np.linalg.lstsq(M, t_train**2)[:2]
    print(m)
    # a = 2/m[0]**2
    a = 2/m[0]
    print(a)
    return a, cov

def fit_neural(h, t_train):
    normalizer = preprocessing.Normalization(input_shape = [1,])
    model = keras.Sequential([
        normalizer,
        layers.Dense(units=16, activation = "relu"),
        layers.Dense(units=16, activation = "relu"),
        layers.Dense(units=16, activation = "relu"),
        layers.Dense(units=16, activation = "relu"),
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







def run_many():
    train_size = list(np.arange(2, 10, 1)) + list(np.arange(10, 100, 10)) + list(np.arange(100, 5000, 100))
    rmse_nn_arr = np.zeros((len(train_size), 5))
    rmse_arr = np.zeros((len(train_size), 5))
    for j in range(5):
        for i, n in enumerate(train_size):
            rmse_arr[i, j], rmse_nn_arr[i, j] = run(n, False) 
    np.savetxt("rmse_nn.csv", rmse_nn_arr)
    np.savetxt("rmse.csv", rmse_arr)
    np.savetxt("train_size.csv", train_size)

def plot_many():
    rmse_nn = np.loadtxt("rmse_nn.csv")
    rmse = np.loadtxt("rmse.csv")
    train_size = np.loadtxt("train_size.csv")


    df_rmse = pd.DataFrame(rmse)
    df_rmse_nn = pd.DataFrame(rmse_nn)
    df_rmse["Training set size"] = train_size
    df_rmse["Method"] = "Mechanistic model"
    
    df_rmse_nn["Training set size"] = train_size
    df_rmse_nn["Method"] = "Deep learning"

    df = pd.concat([df_rmse, df_rmse_nn])
    df_long = pd.melt(df, id_vars=['Method', "Training set size"], value_vars=[0,1,2,3,4], var_name = "Parallell", value_name = "RMSE")
    print(df_long.columns)
    fig, ax = plt.subplots(1)
    sns.lineplot(data = df_long, x = "Training set size", y = "RMSE", hue = "Method")
    ax.set_xscale("log")
    plt.show()




if __name__ == '__main__':
    # rmse, rmse_nn
    # run(N_samples, True)
    # run_many()
    plot_many()