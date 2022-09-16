import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
from IPython.display import clear_output

#Imports for CNN
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

sample_size = 500
width = 100
height = 100

files = ['Fresh', 'Spoiled']

adress = 'C:\\Users\\Arthur\\Documents\\Faculdade\\2022\\semestre2\\TI6\\Teste\\input\\class'

data = {}
for f in files:
    data[f]=[]
for col in files:
    os.chdir(os.path.join(adress,col))
    for i in os.listdir(os.getcwd()):
        if i.endswith('.jpg'):
            data[col].append(i)

table = pd.DataFrame(data).head()

#sizes = [len(data['Fresh']), len(data['Spoiled'])]

#plt.figure(figsize=(10,5), dpi=100)

#plt.pie(x=sizes,autopct='%1.0f%%',shadow=False, textprops={'color':"w","fontsize":15}, startangle=90,explode=(0,.01))
#plt.legend(files,bbox_to_anchor=(0.4, 0, .7, 1))
#plt.title("Data Split")
#plt.show()

start = time.time()
image_data = []
image_target = []

for title in files:
    os.chdir(adress.format(title))
    counter = 0
    for i in data[title]:
        img = cv2.imread(os.path.join(adress,title,i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        image_data.append(cv2.resize(img,(width, height)))
        image_target.append(title)
        counter += 1
        if counter == sample_size:
            break
    clear_output(wait=True)
   #print("Compiled Class",title)
calculate_time = time.time() - start    
#print("Calculate Time",round(calculate_time,5))

image_data = np.array(image_data)
size = image_data.shape[0]
image_data.shape

#plt.figure(figsize=(15,15))
#for i in range(1,17):
#    fig = np.random.choice(np.arange(size))
#    plt.subplot(4,4,i)
#    plt.imshow(image_data[fig])
#    if image_target[fig]=='Fresh':
#        c='green'
#    else:
#        c='red'
#    plt.title(image_target[fig], color=c)
#    plt.xticks([]), plt.yticks([])
#plt.show()

labels = LabelEncoder()
labels.fit(image_target)

X = image_data / 255.0
y = labels.transform(image_target)
train_images, test_images, train_labels, test_labels = train_test_split(X,y, test_size=0.3, random_state=123)

model = models.Sequential()
model.add(layers.Conv2D(35, (3, 3), activation='relu', input_shape=(width,height,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=2, 
                    validation_data=(test_images, test_labels))

#plt.style.use('ggplot')
#plt.figure(figsize=(10, 5))
###plt.plot(history.history['accuracy'], label='accuracy')
##lt.plot(history.history['val_accuracy'], label = 'val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.ylim([0.5, 1.01])
#plt.legend(loc='lower right')

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

result=model.evaluate(test_images, test_labels)
print(result)

for i in range(len(model.metrics_names)):
    print(model.metrics_names[i],":",result[i])

model.summary()
