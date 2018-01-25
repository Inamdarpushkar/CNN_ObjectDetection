

####  Designing and trainig a CNN model
#layers
model = Sequential()

model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=(80, 80, 3)))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.50))

model.add(Dense(2))
model.add(Activation('softmax'))

#compile
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

#model fitting
hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=17, verbose=1, validation_split=0.2)
