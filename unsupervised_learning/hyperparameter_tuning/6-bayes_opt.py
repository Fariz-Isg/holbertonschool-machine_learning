#!/usr/bin/env python3
"""
Bayesian Optimization of a Neural Network using GPyOpt.
Optimizes: learning_rate, units, dropout, l2, batch_size
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import GPyOpt
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

REPORT = []


def build_and_train(learning_rate, units, dropout, l2, batch_size):
    """Builds, trains, and evaluates a model with given hyperparameters."""
    units = int(units)
    batch_size = int(batch_size)
    reg = keras.regularizers.l2(l2)

    model = keras.Sequential([
        keras.layers.Dense(units, activation='relu',
                           kernel_regularizer=reg, input_shape=(784,)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units // 2, activation='relu',
                           kernel_regularizer=reg),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    ckpt_name = ('best_lr{:.4f}_u{}_d{:.2f}'
                 '_l2{:.5f}_b{}.keras').format(
        learning_rate, units, dropout, l2, batch_size)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_name, save_best_only=True,
            monitor='val_accuracy', mode='max'),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3,
            restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    best_val_acc = max(history.history['val_accuracy'])
    return best_val_acc


def objective(params):
    """Objective function for GPyOpt (minimizes negative accuracy)."""
    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout = float(params[0][2])
    l2 = float(params[0][3])
    batch_size = int(params[0][4])

    val_acc = build_and_train(learning_rate, units, dropout, l2, batch_size)
    result = 1.0 - val_acc

    entry = ('lr={:.4f}, units={}, dropout={:.2f}, '
             'l2={:.5f}, batch={} => val_acc={:.4f}').format(
        learning_rate, units, dropout, l2, batch_size, val_acc)
    print(entry)
    REPORT.append(entry)

    return result


domain = [
    {'name': 'learning_rate', 'type': 'continuous',
     'domain': (1e-4, 1e-2)},
    {'name': 'units', 'type': 'discrete',
     'domain': (64, 128, 256, 512)},
    {'name': 'dropout', 'type': 'continuous',
     'domain': (0.1, 0.5)},
    {'name': 'l2', 'type': 'continuous',
     'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete',
     'domain': (32, 64, 128, 256)},
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective,
    domain=domain,
    acquisition_type='EI',
    maximize=False,
    num_cores=1,
    verbosity=True
)

optimizer.run_optimization(max_iter=30)

optimizer.plot_convergence()
plt.savefig('convergence.png')
plt.close()

with open('bayes_opt.txt', 'w') as f:
    f.write('Bayesian Optimization Report\n')
    f.write('=' * 40 + '\n\n')
    for entry in REPORT:
        f.write(entry + '\n')
    f.write('\nBest hyperparameters:\n')
    best = optimizer.x_opt
    f.write('  learning_rate : {:.4f}\n'.format(best[0]))
    f.write('  units         : {}\n'.format(int(best[1])))
    f.write('  dropout       : {:.2f}\n'.format(best[2]))
    f.write('  l2            : {:.5f}\n'.format(best[3]))
    f.write('  batch_size    : {}\n'.format(int(best[4])))
    f.write('  best val_acc  : {:.4f}\n'.format(1.0 - optimizer.fx_opt))

print('Best hyperparameters:', optimizer.x_opt)
print('Best val accuracy:', 1.0 - optimizer.fx_opt)
