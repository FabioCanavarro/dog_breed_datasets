import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
datasetpaths = [os.path.join(r"dataset", i) for i in os.listdir(r"dataset")]
print(datasetpaths)
breeds = ["Golden Retriever",
"German Shepherd",
"Labrador Retriever",
"Bulldog",
"Beagle",
"Poodle",
"Rottweiler",
"Yorkshire Terrier",
"Boxer",
"Dachshund"]
len(breeds)
shapey = 10
breedsonehot = {"Golden Retriever":0,
"German Shepherd":1,
"Labrador Retriever":2,
"Bulldog":3,
"Beagle":4,
"Poodle":5,
"Rottweiler":6,
"Yorkshire Terrier":7,
"Boxer":8,
"Dachshund":9}

#kr.utils.to_categorical
x = []

datasetlabel = datasetpaths[0]
datasetlabel
x = plt.imread(os.path.join(datasetlabel, os.listdir(datasetlabel)[0]))
print(x.shape)
plt.imshow(x)
x = plt.imread(os.path.join(datasetlabel, os.listdir(datasetlabel)[1]))
print(x.shape)
plt.imshow(x)
allheight = []
allwidth = []
def get_image_size(image_path):
    """
    This function takes an image path and returns its width and height as a tuple.
    """
    x = plt.imread(image_path).shape
    width, height = x[0],x[1]
    return width, height

for datasetlabel in datasetpaths:
    widths = []
    heights = []
    for filename in os.listdir(datasetlabel):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_shape = plt.imread(os.path.join(datasetlabel, filename)).shape
            widths.append(image_shape[0])
            heights.append(image_shape[1])
    plt.figure(figsize=(10, 6))

    # Histogram for width
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins='auto', edgecolor='black')
    plt.xlabel("Image Width")
    plt.ylabel("Frequency")
    plt.title("Width Distribution")

    # Histogram for height
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins='auto', edgecolor='black')
    plt.xlabel("Image Height")
    plt.ylabel("Frequency")
    plt.title("Height Distribution")

    plt.tight_layout()
    plt.show()
    print("width max: ",max(widths))
    print("width min: ",min(widths))
    
    print("height max: ",max(heights))
    print("height min: ",min(heights))
    allheight.append(heights)
    allwidth.append(widths)

print("all width max: ",max(max(allwidth)))
print("all width min: ",min(min(allwidth)))

print("all height max: ",max(max(allheight)))
print("all height min: ",min(min(allheight)))


x = plt.imread(os.path.join(datasetlabel, os.listdir(datasetlabel)[0]))
#os.path.join(datasetlabel,x)
x = tf.image.resize_with_crop_or_pad(x,140,162)
plt.imshow(x)
print(x.shape)
#since all diff shape best shape would be 140,162



def imgfrompath(x):
    image = plt.imread(x)
    image_resize = tf.image.resize_with_crop_or_pad(image,140,162)
    return image_resize
plt.imshow(imgfrompath(os.path.join(datasetlabel, os.listdir(datasetlabel)[1])))
Beagle = [os.path.join(datasetpaths[0],i) for i in os.listdir(datasetpaths[0])]

Boxer = [os.path.join(datasetpaths[1],i) for i in os.listdir(datasetpaths[1])]

Bulldog = [os.path.join(datasetpaths[2],i) for i in os.listdir(datasetpaths[2])]

Dachshund = [os.path.join(datasetpaths[3],i) for i in os.listdir(datasetpaths[3])]

German_Shepherd = [os.path.join(datasetpaths[4],i) for i in os.listdir(datasetpaths[4])]

Golden_Retriever = [os.path.join(datasetpaths[5],i) for i in os.listdir(datasetpaths[5])]

Labrador_Retriever = [os.path.join(datasetpaths[6],i) for i in os.listdir(datasetpaths[6])]

Poodle = [os.path.join(datasetpaths[7],i) for i in os.listdir(datasetpaths[7])]

Rottweiler = [os.path.join(datasetpaths[8],i) for i in os.listdir(datasetpaths[8])]

Yorkshire_Terrier = [os.path.join(datasetpaths[9],i) for i in os.listdir(datasetpaths[9])]

breedvar = [Beagle,Boxer,Bulldog,Dachshund,German_Shepherd,Golden_Retriever,Labrador_Retriever,Poodle,Rottweiler,Yorkshire_Terrier]
kr.utils.to_categorical(9,shapey)
# np.argmax(p) for converting onehot to original indices
def preprocess(paths):
    x = np.expand_dims(imgfrompath(paths[0]),axis=0)

    
    #labels


    labelname = paths[0].split("\\")[-1].split("_")[0]
    y = np.expand_dims(kr.utils.to_categorical(breedsonehot[labelname],shapey),0)
    
    
    for i in range(1,len(paths)):

        #x
        temp = np.expand_dims(imgfrompath(paths[i]),axis=0)
        x = np.concatenate((x,temp),axis=0)
        
        labelname = paths[i].split("\\")[-1].split("_")[0]
        #y
        tempy = np.expand_dims(kr.utils.to_categorical(breedsonehot[labelname],shapey),0)
        y = np.concatenate((y,tempy),0)
    return x , y


x,y =preprocess(Beagle)
print(x.shape)
print(y.shape)
x_full = preprocess(breedvar[0])[0]
y_full = preprocess(breedvar[0])[1]
for i in range(1,len(breedvar)):
    x,y = preprocess(breedvar[i])
    x_full = np.concatenate((x_full,x),axis=0)
    y_full = np.concatenate((y_full,y),axis=0)
print(x_full.shape)
print(y_full.shape)
plt.imshow(x_full[0])
print(np.argmax(y_full[0]))

from tensorflow.keras import layers
import tensorflow.keras as kr

inputs = layers.Input(shape=(140, 162, 3))
x = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.SeparableConv2D(256, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = kr.Model(inputs, outputs)
model.summary()

#split data
#80 - 10 - 10
# 773 - 97, 97
#shuffle
np.random.seed(0)
np.random.shuffle(x_full)
np.random.seed(0)
np.random.shuffle(y_full)
np.random.seed(0)

x_train = x_full[:773+97]
x_test = x_full[773+97:773+97+97]

y_train = y_full[:773+97]
y_test = y_full[773+97:773+97+97]

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(x_train.dtype, x_train.min(), x_train.max())


print(x_train.shape)
print(x_test.shape)

print()

print(y_train.shape)
print(y_test.shape)

%load_ext tensorboard
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")




model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

callback = [
    kr.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True),
    kr.callbacks.ModelCheckpoint("checkpointmodel.keras", verbose=1, save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.1,
    callbacks=callback
)
model.evaluate(x_train,y_train)
model.save("Best_model.keras")