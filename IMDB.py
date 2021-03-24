#!/usr/bin/env python
# coding: utf-8

# In[63]:


from keras.datasets import imdb


# In[64]:


(train_data, train_labels), (test_data, test_labels)= imdb.load_data(num_words=10000)


# In[65]:


train_data[0]


# In[66]:


train_labels[0] 


# In[67]:


max([max(sequence) for sequence in train_data]) 


# In[68]:


word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])


# In[69]:


decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[70]:


print(train_data[11])


# In[71]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1. 
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[72]:


x_train[0]


# In[73]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[74]:


from keras import models 
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[75]:


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


# In[76]:


from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
loss='binary_crossentropy', metrics=['accuracy'])


# In[77]:


from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
loss=losses.binary_crossentropy,
metrics=[metrics.binary_accuracy])


# In[78]:


x_val = x_train[:10000] # 검증 데이터
partial_x_train = x_train[10000:] # 훈련 데이터
y_val = y_train[:10000] # 검증 label
partial_y_train = y_train[10000:] # 훈련 label


# In[91]:


model.compile(optimizer='rmsprop',
loss='binary_crossentropy', metrics=['acc'])
history = model.fit(partial_x_train,
partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val)) 


# In[93]:


history_dict = history.history


# In[94]:


history_dict.keys()


# In[82]:


acc = history_dict['acc']


# In[83]:


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[84]:


plt.clf() # 생성한 그래프를 clear
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


# In[85]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[86]:


results


# In[87]:


model.predict(x_test)


# In[ ]:




