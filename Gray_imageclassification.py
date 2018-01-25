
# Ships detection in satellite images (CNN-Gray)

### 1. Importing libraries (OpenCv,numpy,sklearn,Keras)

## Import Libraries
import os,cv2
import sys, random
import numpy as np
import matplotlib.pyplot as plt
import itertools

#sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix



## Keras
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam,adadelta


### 2. Loading the dataset

'''This block of code reads images from different labeled  folders,
resizes all images (rows and columns), converts it to the gray channel
and appends it to the list'''

Path=os.getcwd() # current directory
data_path=Path+'/Data' #two folders inside Data floder(ships and no ships)
data_dir_list=os.listdir(data_path) #Two folders in this case
img_data_list=[] #Appends images from different labeled folders to this list
for dataset in data_dir_list:
    if dataset=='.DS_Store':
        pass
    else:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize=cv2.resize(input_img,(80,80))
            img_data_list.append(input_img_resize)


#Input parameters

img_rows=80 #image dimesions~ hight
img_cols=80 #image dimesions~ width
num_channel=1 #image dimesions~ Gray image
num_epoch=20 # one epoch == one complete cycle of forward and back-propogation
num_classes=2 #number of output classes


### 3. Preprocessing the dataset
'''Converts list of images to arrays using numpy, casts its type to float
(for computation) and normalizes the images values by dividing
 it with max (255)'''

img_data=np.array(img_data_list) #Converts images to list of arrays
img_data=img_data.astype('float32') #converting data it to float
img_data/=255 #normalization
print (img_data.shape)


'''As theano and tensorflow takes differnt input dimensions this block of
converts it to appropriate dimensions
e.g. (number of images, channels, rows, columns)'''

if num_channel==1: #for one channel
    if K.image_dim_ordering()=='th':
        img_data= np.expand_dims(img_data, axis=1)
        print (img_data.shape)
    else:
        img_data= np.expand_dims(img_data, axis=4)
        print (img_data.shape)
else: # for RGB
    if K.image_dim_ordering()=='th':
        img_data=np.rollaxis(image_data,3,1)
        print(img_data.shape)
num_of_samples=img_data.shape[0] # Total number of images


#### Labels
'''As we read data from different folders sequentially we
can assing labels mannualy'''

labels=np.ones((num_of_samples,),dtype='int64') #sets length based on total number of images
labels[0:699]=1
labels[700:2799]=0
names=['ships','no_ships']
Y = np_utils.to_categorical(labels, num_classes) #to_categorical converts lables to array of inategers (dummy variable)


#### Shuffeling and spliting data into train, validation, test dataset


x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
input_shape=img_data[0].shape


### 4. Designing CNN model


model = Sequential()

model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))


# Compiling CNN model (80-20)
#Adam optimizer default parameters
#with 20 Epochs
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])


# Viewing model_configuration
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


# Model fit

hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_split=0.2)


### 5. Plotting the loss and accuracy curve


train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()


# Model evaluation

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# Predicting new image class

test_image = X_test[0:1]
print (test_image.shape)
print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])


#### Testing a new iamge
#Input new image, processing it to an original input dimentsions
test_image = cv2.imread('/Users/Pushkar/Downloads/ships-in-satellite-imagery/resize/1-2750.png')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(80,80))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)


# Dimensions based on theano/tensorflow

if num_channel==1:
    if K.image_dim_ordering()=='th':
        test_image= np.expand_dims(test_image, axis=0)
        test_image= np.expand_dims(test_image, axis=0)
        print (test_image.shape)
    else:
        test_image= np.expand_dims(test_image, axis=3)
        test_image= np.expand_dims(test_image, axis=0)
        print (test_image.shape)
else:
    if K.image_dim_ordering()=='th':
        test_image=np.rollaxis(test_image,2,0)
        test_image= np.expand_dims(test_image, axis=0)
        print (test_image.shape)
    else:
        test_image= np.expand_dims(test_image, axis=0)
        print (test_image.shape)


### Predicting CLass of input image (0-noship,1-ship)

# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))


### 6. Visualizing the intermediate layers

def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

layer_num=4
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)
print (np.shape(activations))
feature_maps = activations[0][0]
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
    feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.show()

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax = fig.add_subplot(subplot_num, subplot_num, i+1)
    ax.imshow(feature_maps[:,:,i],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()


### 7. Plotting the confusion matrix to understand the model performance

# Printing the confusion matrix


Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
target_names = ['class 0(No_ships)', 'class 1(Ships)']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# In[27]:

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[82]:

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
                     title='Normalized confusion matrix')
#plt.figure()
plt.show()
