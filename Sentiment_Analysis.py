# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
import csv
from typing import Callable, Tuple


import jax.numpy as jn
import matplotlib.pyplot as plt
import numpy as np
from jax import lax

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import trange
from sklearn.model_selection import train_test_split

# %pip --quiet install objax
import objax
from objax.typing import JaxArray
import jax
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jax.numpy as jn
import random


objax.random.DEFAULT_GENERATOR.seed(42)
np.random.seed(42)

!gdown --id 11r58MB8wRBO1o1gEC-zxiADZIuwMnhf7

max_vocab = 2000  # this parameter is for the maximum number of words in the "dictionary"
max_len = 200  # maximum length of each review
embedding_size = 30  # size of embedding
num_hidden_units_GRU = 30  # GRU cells in the RNN layer
num_hidden_units = 60  # hidden unit of dense network after GRU

vocab_size = max_vocab
filename = 'IMDB Dataset.csv'

def data_processing(filename, max_vocab, max_len):
  # filename: the name of the .csv file
  # max_vocab: The maximum number of words
  # max_len:
  messages = []  # a list contains the reviews
  labels = []  # a list contains the labels
  with open(filename, 'r') as file:
      reader = csv.reader(file)
      firstline = True
      for row in reader:
        if firstline:
            firstline = False
            continue
        else:
            messages.append(row[0])
            labels.append(int(row[1]=='positive'))

  tokenizer = Tokenizer(num_words=max_vocab)
  tokenizer.fit_on_texts(messages)
  messages_seq = tokenizer.texts_to_sequences(messages)
  data = pad_sequences(messages_seq, maxlen=max_len)
  train_size = 0.8
  messages_train, messages_valid_test, labels_train, labels_valid_test  = train_test_split(data, labels, train_size=train_size)
  messages_valid, messages_test, labels_valid, labels_test  = train_test_split(messages_valid_test, labels_valid_test, train_size=0.5)
  return np.array(messages_train), np.array(labels_train), np.array(messages_valid), np.array(labels_valid), np.array(messages_test), np.array(labels_test)


messages_train, labels_train, messages_valid, labels_valid, messages_test, labels_test = data_processing(filename, max_vocab, max_len)

print('one input in the training set:', messages_train[5])
print('its corresponding label:', labels_train[5])


class Embed(objax.Module):
    def __init__(self, size: int, latent: int, init: Callable = objax.nn.init.xavier_truncated_normal):
        self.w = objax.TrainVar(init((size, latent)))

    def __call__(self, x: JaxArray) -> JaxArray:
        return self.w.value[x]


class GRU(objax.Module):
    def __init__(self, nin: int, nout: int,
                 init_w: Callable = objax.nn.init.xavier_truncated_normal,
                 init_b: Callable = objax.nn.init.truncated_normal):
        self.update_w = objax.TrainVar(init_w((nin, nout)))
        self.update_u = objax.TrainVar(init_w((nout, nout)))
        self.update_b = objax.TrainVar(init_b((nout,), stddev=0.01))
        self.reset_w = objax.TrainVar(init_w((nin, nout)))
        self.reset_u = objax.TrainVar(init_w((nout, nout)))
        self.reset_b = objax.TrainVar(init_b((nout,), stddev=0.01))
        self.output_w = objax.TrainVar(init_w((nin, nout)))
        self.output_u = objax.TrainVar(init_w((nout, nout)))
        self.output_b = objax.TrainVar(init_b((nout,), stddev=0.01))

    def __call__(self, x: JaxArray, initial_state: JaxArray) -> Tuple[JaxArray, JaxArray]:
        def scan_op(state: JaxArray, x: JaxArray) -> JaxArray:  # State must come first for lax.scan
            # fill this in
            update_gate = jax.nn.sigmoid((x @ self.update_w.value) + (state @ self.update_u.value) + self.update_b.value)

            # fill this in
            reset_gate = jax.nn.sigmoid((x @ self.reset_w.value) + (state @ self.reset_u.value) + self.reset_b.value)

            # fill this in
            output_gate = jn.tanh((x @ self.output_w.value) + ((reset_gate * state) @ self.output_u.value) + self.output_b.value)

            #return update_gate * state + (1 - update_gate) * output_gate, 0  # we don't use the output, return 0.
            return (1-update_gate) * state + (update_gate) * output_gate, 0   # we don't use the output, return 0.

        return lax.scan(scan_op, initial_state, x.transpose((1, 0, 2)))[0]

def cumulative_sum(carry, x):
    cum_sum, = carry
    return (cum_sum + x,), cum_sum + x

array = jn.array([1, 2, 3, 4, 5])
output = jn.zeros_like(array[0])
output_sum, _ = jax.lax.scan(cumulative_sum, (output,), array)

print("input sequence:", array)
print("cumulative sum:", output_sum)

# fill this in:
embedding = Embed(max_len , embedding_size)
gru = GRU(embedding_size , num_hidden_units_GRU)
gru_rnn = objax.nn.Sequential([embedding , gru , objax.nn.Linear(num_hidden_units_GRU , num_hidden_units) ,
                               objax.functional.relu , objax.nn.Linear(num_hidden_units , 2) ])

print(f'{" Network ":-^79}')
print(gru_rnn.vars())


## Your implementaiton of the optimizer should go here
opt = objax.optimizer.SGD(gru_rnn.vars())


def loss_function(x: JaxArray, y: JaxArray):
    logits = gru_rnn(x, initial_state=jn.zeros((x.shape[0], num_hidden_units_GRU)))
    return objax.functional.loss.cross_entropy_logits_sparse(logits, y).sum()


gv = objax.GradValues(loss_function, gru_rnn.vars())


@objax.Function.with_vars(gv.vars() + opt.vars())
def train_op(x: JaxArray, y: JaxArray, lr: float):
    g, loss = gv(x, y)
    opt(lr, g)
    return loss


train_op = objax.Jit(train_op)
eval_op = objax.Jit(lambda x: gru_rnn(x, initial_state=jn.zeros((x.shape[0], num_hidden_units_GRU))),
                    gru_rnn.vars())

def accuracy(data_loader):
    """Compute the accuracy for a provided data loader"""
    acc_total = 0
    x, y = data_loader
    batch_size_acc = 500
    for batch_idx in np.array_split(np.arange(len(x)), len(x) // batch_size_acc):
        x_batch, target_class = x[batch_idx], y[batch_idx]
        predicted_class = eval_op(x_batch).argmax(1)
        acc_total += (predicted_class == target_class).sum()
    return acc_total / len(x)

learning_rate = 1e-3 # learning rate
num_epochs = 20 # number of epochs
batch_size = 250  # batch size
training_data = (messages_train, labels_train)
validation_data = (messages_valid, labels_valid)
test_data = (messages_test, labels_test)

# you code for the training loop should start here

def training_loop(EPOCHS, BATCH, LEARNING_RATE):
  avg_train_loss_epoch = []
  avg_val_loss_epoch = []
  avg_test_loss_epoch=[]

  train_acc_epoch = []
  val_acc_epoch = []
  test_acc_epoch=[]

  for epoch in range(EPOCHS):
      avg_train_loss = 0 # (averaged) training loss per batch
      avg_val_loss =  0  # (averaged) validation loss per batch
      avg_test_loss=0

      train_acc = 0      # training accuracy per batch
      val_acc = 0        # validation accuracy per batch
      test_acc=0

      # shuffle the examples prior to training to remove correlation
      train_indices = np.arange(len(messages_train))
      np.random.shuffle(train_indices)
      for it in range(0, messages_train.shape[0], BATCH):
          batch = train_indices[it : BATCH + it]
          avg_train_loss += float(train_op(messages_train[batch], labels_train[batch], LEARNING_RATE)[0]) * len(batch)
          train_prediction = eval_op(messages_train[batch]).argmax(1)
          train_acc += (np.array(train_prediction).flatten() == labels_train[batch]).sum()
      train_acc_epoch.append(train_acc/messages_train.shape[0])
      avg_train_loss_epoch.append(avg_train_loss/messages_train.shape[0])

      # run validation
      val_indices = np.arange(len(messages_valid))
      np.random.shuffle(val_indices)
      for it in range(0, messages_valid.shape[0], BATCH):
          batch = val_indices[it : BATCH + it]
          avg_val_loss += float(loss_function(messages_valid[batch], labels_valid[batch])) * len(batch)
          val_prediction = eval_op(messages_valid[batch]).argmax(1)
          val_acc += (np.array(val_prediction).flatten() == labels_valid[batch]).sum()
      val_acc_epoch.append(val_acc/messages_valid.shape[0])
      avg_val_loss_epoch.append(avg_val_loss/messages_valid.shape[0])

      print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/messages_train.shape[0], avg_val_loss/labels_valid.shape[0], 100*train_acc/messages_train.shape[0], 100*val_acc/labels_valid.shape[0]))

      test_indices = np.arange(len(messages_test))
      np.random.shuffle(test_indices)
      for it in range(0, messages_test.shape[0], BATCH):
          batch = test_indices[it : BATCH + it]
          avg_test_loss += float(loss_function(messages_test[batch], labels_test[batch])) * len(batch)
          test_prediction = eval_op(messages_test[batch]).argmax(1)
          test_acc += (np.array(test_prediction).flatten() == labels_test[batch]).sum()
      test_acc_epoch.append(test_acc/messages_test.shape[0])
      avg_test_loss_epoch.append(avg_test_loss/messages_test.shape[0])

  print('test accuracy is',test_acc_epoch[len(test_acc_epoch)-1]*100,'%')
  #Plot training loss
  plt.title("Train vs Validation Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.plot(avg_val_loss_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train vs Validation Accuracy")
  plt.plot(train_acc_epoch, label="Train")
  plt.plot(val_acc_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.legend(loc='best')
  plt.show()

training_loop(num_epochs,batch_size,learning_rate)


# Define (again) your model here
embedding = Embed(max_len , embedding_size)
gru = GRU(embedding_size , num_hidden_units_GRU)

gru_rnn2 = objax.nn.Sequential([embedding , gru , objax.nn.Linear(num_hidden_units_GRU , num_hidden_units) ,
                               objax.functional.relu , objax.nn.Linear(num_hidden_units , 2) ])

## Your implementaiton of the optimizer should go here
opt2 = objax.optimizer.Adam(gru_rnn2.vars())

"""You will also need the following functions."""

def loss_function(x: JaxArray, y: JaxArray):
    logits = gru_rnn2(x, initial_state=jn.zeros((x.shape[0], num_hidden_units_GRU)))
    return objax.functional.loss.cross_entropy_logits_sparse(logits, y).sum()


gv2 = objax.GradValues(loss_function, gru_rnn2.vars())


@objax.Function.with_vars(gv2.vars() + opt2.vars())
def train_op(x: JaxArray, y: JaxArray, lr: float):
    g, loss = gv2(x, y)
    opt2(lr, g)
    return loss


train_op = objax.Jit(train_op)
eval_op = objax.Jit(lambda x: gru_rnn2(x, initial_state=jn.zeros((x.shape[0], num_hidden_units_GRU))),
                    gru_rnn2.vars())

learning_rate = 1e-3
num_epochs = 20
batch_size = 250
training_data = (messages_train, labels_train)
validation_data = (messages_valid, labels_valid)
test_data = (messages_test, labels_test)

# you code for the training loop should start here

def training_Adam(EPOCHS, BATCH, LEARNING_RATE):
  avg_train_loss_epoch = []
  avg_val_loss_epoch = []
  avg_test_loss_epoch=[]

  train_acc_epoch = []
  val_acc_epoch = []
  test_acc_epoch=[]

  for epoch in range(EPOCHS):
      avg_train_loss = 0 # (averaged) training loss per batch
      avg_val_loss =  0  # (averaged) validation loss per batch
      avg_test_loss=0

      train_acc = 0      # training accuracy per batch
      val_acc = 0        # validation accuracy per batch
      test_acc=0

      # shuffle the examples prior to training to remove correlation
      train_indices = np.arange(len(messages_train))
      np.random.shuffle(train_indices)
      for it in range(0, messages_train.shape[0], BATCH):
          batch = train_indices[it : BATCH + it]
          avg_train_loss += float(train_op(messages_train[batch], labels_train[batch], LEARNING_RATE)[0]) * len(batch)
          train_prediction = eval_op(messages_train[batch]).argmax(1)
          train_acc += (np.array(train_prediction).flatten() == labels_train[batch]).sum()
      train_acc_epoch.append(train_acc/messages_train.shape[0])
      avg_train_loss_epoch.append(avg_train_loss/messages_train.shape[0])

      # run validation
      val_indices = np.arange(len(messages_valid))
      np.random.shuffle(val_indices)
      for it in range(0, messages_valid.shape[0], BATCH):
          batch = val_indices[it : BATCH + it]
          avg_val_loss += float(loss_function(messages_valid[batch], labels_valid[batch])) * len(batch)
          val_prediction = eval_op(messages_valid[batch]).argmax(1)
          val_acc += (np.array(val_prediction).flatten() == labels_valid[batch]).sum()
      val_acc_epoch.append(val_acc/messages_valid.shape[0])
      avg_val_loss_epoch.append(avg_val_loss/messages_valid.shape[0])

      print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/messages_train.shape[0], avg_val_loss/labels_valid.shape[0], 100*train_acc/messages_train.shape[0], 100*val_acc/labels_valid.shape[0]))

      test_indices = np.arange(len(messages_test))
      np.random.shuffle(test_indices)
      for it in range(0, messages_test.shape[0], BATCH):
          batch = test_indices[it : BATCH + it]
          avg_test_loss += float(loss_function(messages_test[batch], labels_test[batch])) * len(batch)
          test_prediction = eval_op(messages_test[batch]).argmax(1)
          test_acc += (np.array(test_prediction).flatten() == labels_test[batch]).sum()
      test_acc_epoch.append(test_acc/messages_test.shape[0])
      avg_test_loss_epoch.append(avg_test_loss/messages_test.shape[0])

  print('test accuracy is',test_acc_epoch[len(test_acc_epoch)-1]*100,'%')
  #Plot training loss
  plt.title("Train vs Validation Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.plot(avg_val_loss_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train vs Validation Accuracy")
  plt.plot(train_acc_epoch, label="Train")
  plt.plot(val_acc_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.legend(loc='best')
  plt.show()

training_Adam(num_epochs,batch_size,learning_rate)

# Your implementation of the model should go here
embedding = Embed(max_len , embedding_size)
gru = GRU(embedding_size , num_hidden_units_GRU)

gru_rnn3 = objax.nn.Sequential([embedding , gru , objax.nn.Linear(num_hidden_units_GRU , num_hidden_units) ,
                               objax.functional.relu , objax.nn.Linear(num_hidden_units , 2) ])

## Your implementaiton of the optimizer should go here
opt3 = objax.optimizer.Adam(gru_rnn3.vars())


def loss_function(x: JaxArray, y: JaxArray):
    logits = gru_rnn3(x, initial_state=jn.zeros((x.shape[0], num_hidden_units_GRU)))
    return objax.functional.loss.cross_entropy_logits_sparse(logits, y).sum()


gv3 = objax.GradValues(loss_function, gru_rnn3.vars())


@objax.Function.with_vars(gv3.vars() + opt3.vars())
def train_op(x: JaxArray, y: JaxArray, lr: float):
    g, loss = gv3(x, y)
    opt3(lr, g)
    return loss


train_op = objax.Jit(train_op)
eval_op = objax.Jit(lambda x: gru_rnn3(x, initial_state=jn.zeros((x.shape[0], num_hidden_units_GRU))),
                    gru_rnn3.vars())

learning_rate = 1e-3
num_epochs = 20
batch_size = 250
max_patience_window = 5
training_data = (messages_train, labels_train)
validation_data = (messages_valid, labels_valid)
test_data = (messages_test, labels_test)

# you code for the training loop should start here

def training_ES(EPOCHS, BATCH, LEARNING_RATE):
  avg_train_loss_epoch = []
  avg_val_loss_epoch = []
  avg_test_loss_epoch=[]

  train_acc_epoch = []
  val_acc_epoch = []
  test_acc_epoch=[]

  for epoch in range(EPOCHS):
      avg_train_loss = 0 # (averaged) training loss per batch
      avg_val_loss =  0  # (averaged) validation loss per batch
      avg_test_loss=0

      train_acc = 0      # training accuracy per batch
      val_acc = 0        # validation accuracy per batch
      test_acc=0

      # shuffle the examples prior to training to remove correlation
      train_indices = np.arange(len(messages_train))
      np.random.shuffle(train_indices)
      for it in range(0, messages_train.shape[0], BATCH):
          batch = train_indices[it : BATCH + it]
          avg_train_loss += float(train_op(messages_train[batch], labels_train[batch], LEARNING_RATE)[0]) * len(batch)
          train_prediction = eval_op(messages_train[batch]).argmax(1)
          train_acc += (np.array(train_prediction).flatten() == labels_train[batch]).sum()
      train_acc_epoch.append(train_acc/messages_train.shape[0])
      avg_train_loss_epoch.append(avg_train_loss/messages_train.shape[0])

      # run validation
      val_indices = np.arange(len(messages_valid))
      np.random.shuffle(val_indices)
      for it in range(0, messages_valid.shape[0], BATCH):
          batch = val_indices[it : BATCH + it]
          avg_val_loss += float(loss_function(messages_valid[batch], labels_valid[batch])) * len(batch)
          val_prediction = eval_op(messages_valid[batch]).argmax(1)
          val_acc += (np.array(val_prediction).flatten() == labels_valid[batch]).sum()
      val_acc_epoch.append(val_acc/messages_valid.shape[0])
      avg_val_loss_epoch.append(avg_val_loss/messages_valid.shape[0])

      print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/messages_train.shape[0], avg_val_loss/labels_valid.shape[0], 100*train_acc/messages_train.shape[0], 100*val_acc/labels_valid.shape[0]))

      test_indices = np.arange(len(messages_test))
      np.random.shuffle(test_indices)
      for it in range(0, messages_test.shape[0], BATCH):
          batch = test_indices[it : BATCH + it]
          avg_test_loss += float(loss_function(messages_test[batch], labels_test[batch])) * len(batch)
          test_prediction = eval_op(messages_test[batch]).argmax(1)
          test_acc += (np.array(test_prediction).flatten() == labels_test[batch]).sum()
      test_acc_epoch.append(test_acc/messages_test.shape[0])
      avg_test_loss_epoch.append(avg_test_loss/messages_test.shape[0])

      Tol = 0
      if len(avg_val_loss_epoch) > 1:
        if (avg_val_loss_epoch[epoch] > avg_val_loss_epoch[epoch-1]):
          Tol += 1
          if (Tol >= 5):
            print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/training_data[0].shape[0], avg_val_loss/validation_data[0].shape[0], 100*train_acc/training_data[0].shape[0], 100*val_acc/validation_data[0].shape[0]))
            print('Early stopping')
            break
        else:
          Tol = 0

  print('test accuracy is',test_acc_epoch[len(test_acc_epoch)-1]*100,'%')
  #Plot training loss
  plt.title("Train vs Validation Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.plot(avg_val_loss_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train vs Validation Accuracy")
  plt.plot(train_acc_epoch, label="Train")
  plt.plot(val_acc_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.legend(loc='best')
  plt.show()

training_ES(num_epochs,batch_size,learning_rate)
