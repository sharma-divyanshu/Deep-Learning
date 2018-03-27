from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization

classifier = Sequential()

classifier.add(Conv2D(32, 3, 3, input_shape=(150, 150, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#no need to add input shape for second layer
classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())


classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())

classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                  target_size=(150, 150),
                                                  batch_size=128,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(150, 150),
                                                        batch_size=128,
                                                        class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/32,
                         epochs=50,
                         validation_data=test_set,
                         validation_steps=2000/32)

classifier.save('classifier.h5')
classifier.save_weights('weights.h5')

#import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#training_set.class_indices
#if result[0][0] == 1:
#    prediction = 'dog'
#else:
#    prediction = 'cat'