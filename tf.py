import numpy as np
import cv2
from random import randint
from sklearn.model_selection import train_test_split
# TensorFlow e Keras
import tensorflow as tf
import tensorflow.keras as keras
 
# Bibliotecas de ajuda
import glob

tf.logging.set_verbosity(tf.logging.ERROR)

# Settings:
img_size = 32
num_class = 6
test_size = 0.2

#fgbg = cv2.createBackgroundSubtractorMOG2()

def histEqual(img):
  #Equalização do histograma
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

def get_img(data_path):
    # Getting image array from path:
    img = cv2.imread(data_path)
    img = cv2.resize(img,(100,100))
    img = histEqual(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,40,255,cv2.THRESH_TOZERO_INV)
    img = cv2.resize(img,(img_size,img_size))
    return img

def get_img2(data_path):
    # Getting image array from path:
    img = cv2.imread(data_path)
    img = cv2.resize(img,(100,100))
    img = histEqual(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,90,255,cv2.THRESH_TOZERO_INV)
    img = cv2.resize(img,(img_size,img_size))
    return img

def get_img3(img):
    # Getting image array from path:
    img = cv2.resize(img,(100,100))
    img = histEqual(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,90,255,cv2.THRESH_TOZERO_INV)
    img = cv2.resize(img,(img_size,img_size))
    img = img / 255.0
    return img


def getWebCam():
  cam = cv2.VideoCapture(0)

  while cam.isOpened():
    rval, frame = cam.read()
    frame = frame[120:360,150:490,:]
    cv2.imshow("preview",cv2.flip(frame,1))
    key = cv2.waitKey(20)
    if key == 27:
      cam.release()
      break
    elif key == 32: # SPACE take frame
      img_test = get_img3(frame)
      img_test = cv2.resize(img_test,(10*img_size,10*img_size))
      cv2.imshow("preview",img_test)
      key = cv2.waitKey(0)
      tests = []
      img_test = cv2.resize(img_test,(img_size,img_size))
      tests.append(img_test)
      tests = np.array(tests)
      choice = model.predict_classes(tests[0:1])
      probs = model.predict(tests[0:1])[0]
      print("Number", choice,  "with prob ",probs[choice])


def get_dataset(dataset_path='Dataset'):
    # Getting all data from data path:
    X = []
    Y = []
    for i in range(6): # choose dataset number
      filepath = "Dataset/" + str(i) + "/*.JPG"
      filenames = glob.glob(filepath)

      for file in filenames:
        X.append(get_img(file))
        Y.append(i)

    X = np.array(X)
    Y = np.array(Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=randint(0,50))
    return X, X_test, Y, Y_test


#-----------------------------------------MAIN-------------------------------------------

train_images, test_images, train_labels, test_labels = get_dataset()
train_images = train_images / 255.0

test_images = test_images / 255.0

#Preparar o modelo de treino
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (img_size, img_size)),
    keras.layers.Dense(78, activation = tf.nn.relu),
    keras.layers.Dense(num_class, activation = tf.nn.softmax)])

#Definição de função de custo, otimizador e métricas
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Treinar o modelo
model.fit(train_images, train_labels, epochs = 30)

#Testar o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#NEVES e LUCAS fotos
testes=[]
for i in range(6):
  img_test = cv2.flip(get_img2("neves_"+ str(i)+".jpg"),1)
  img_test = img_test / 255.0
  testes.append(img_test)

testes = np.array(testes)
for i in range(6):
  img_test = cv2.resize(testes[i],(10*img_size,10*img_size))
  cv2.imshow('p',img_test)
  choice = model.predict_classes(testes[i:i+1])
  probs = model.predict(testes[i:i+1])[0]
  print("Number", choice,  "with prob ",probs[choice])
  key = cv2.waitKey(0)
  if key == 27:
    break

#TESTS fotos
for i in range(20):
  x = randint(0,200)
  img_test = cv2.resize(test_images[x],(10*img_size,10*img_size))
  cv2.imshow('p',img_test)
  choice = model.predict_classes(test_images[x:x+1])
  probs = model.predict(test_images[x:x+1])[0]
  print("Number", choice,  "with prob ",probs[choice], "real ", test_labels[x])
  key = cv2.waitKey(0)
  if key == 27:
    break

getWebCam()