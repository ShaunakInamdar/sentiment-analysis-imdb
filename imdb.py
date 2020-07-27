from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#load data
top_words=5000 #this allows us to consider only the top words in the reviews and ignore words like "the", "of" etc.
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=top_words)
max_words=500#padding and truncating lengths of reviews to fit
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
#defining CNN
model = Sequential()
model.add(Embedding(top_words, 64, input_length=max_words))
model.add(Conv1D(filters=64, kernel_size=3, padding='same',activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, padding='same',activation='relu'))
#model.add(Conv1D(filters=32, kernel_size=3, padding='valid',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
model.summary()
#fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), validation_split=0.2, epochs=2, batch_size=128, verbose =2)
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: ",(scores[1]*100))
