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

    def on_epoch_begin(self, epoch, logs={}):
        self.train_losses.append([])
        self.train_accs.append([])

    def on_batch_end(self, epoch, logs={}):
        self.train_losses[-1].append(logs.get('loss'))
        self.train_accs[-1].append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))

    @staticmethod
    def _plot_pair(train, test, label):
        epochs = len(train)
        batches_per_epoch = len(train[0])
        max_t = epochs * batches_per_epoch
        batch_t = np.arange(max_t)
        epoch_t = np.arange(0, max_t, batches_per_epoch)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(batch_t, np.array(train), label='Train')
        print(epoch_t.shape)
        print(np.array(test).shape)
        ax.plot(epoch_t, np.array(test), label='Validation')
        ax.set_xlabel('Batch Number')
        ax.set_ylabel(label)
        ax.legend()
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
            ax.axis('off')
            ax.matshow(kernel)
        plt.show()
