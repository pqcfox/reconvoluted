import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from keras.callbacks import Callback


class AccuracyLossPlot(Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.train_accs.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))

    @staticmethod
    def _plot_pair(train, test, label):
        t = range(len(train))
        line_train = plt.plot(t, train, label='Train')[0]
        line_test = plt.plot(t, test, label='Validation')[0]
        plt.legend(handles=[line_train, line_test])
        plt.xlabel('Batch Number')
        plt.ylabel(label)
        plt.show()

    def on_train_end(self, logs={}):
        self._plot_pair(self.train_losses, self.val_losses, 'Loss')
        self._plot_pair(self.train_accs, self.val_accs, 'Accuracy')


class FirstWeightPlot(Callback):
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        super().__init__()

    def on_train_end(self, logs={}):
        first_layer = self.model.layers[0]
        first_weights = first_layer.get_weights()[0]
        squeezed = np.squeeze(first_weights)
        fig = plt.figure()
        gridspec = GridSpec(4, 8)
        for index, kernel in enumerate(squeezed):
            ax = fig.add_subplot(gridspec[index])
            ax.matshow(kernel)
        plt.show()
