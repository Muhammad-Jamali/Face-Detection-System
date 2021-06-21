import os
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
import pandas as pd
import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog
import pickle
from sklearn import svm


# Funtions
def folder_browser():
    inp1.config(state=NORMAL)
    inp1.delete(0,"end")
    input = filedialog.askdirectory(initialdir="/")
    inp1.insert(0, input)
    inp1.config(state=DISABLED)

def preprocess_camera():

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    hogfv_arr = np.empty((0,2305), float)

    img_class = 0
    class_df = pd.DataFrame()

    if not os.path.isfile("./dataset_y.csv"):
        pd.DataFrame({'label': [], 'class': []}).to_csv("dataset_y.csv")
    else:    
        class_df = pd.read_csv("./dataset_y.csv")

    if class_df.empty:
        pd.DataFrame({'class': [0],
                'label': [inp2.get()]
                }).to_csv("./dataset_y.csv", header='column_names')
        img_class = 0
    else:
        if class_df[class_df.label == inp2.get()]['class'].empty:
            pd.DataFrame({'class': [class_df.tail(1)['class'].values[0] + 1],
                    'label': [inp2.get()]
                    }).to_csv("./dataset_y.csv", mode='a', header=False)
            img_class = class_df.tail(1)['class'].values[0] + 1
        else:
            img_class = class_df[class_df.label == inp2.get()]['class'].values[0]

    while True:
            
        # Read the frame
        _, img = cap.read()

        # Detect the faces
        faces = face_cascade.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in faces:
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_image =  img[y:y + h, x:x + w]

            cv2.putText(image, "Train Dataset Creation", (x, y - 10) , cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            hogfv, hog_image = hog(resize(face_image,(256,256)), orientations=9, pixels_per_cell=(16,16), cells_per_block=(16,16),visualize=True)
                    
            hogfv_class_arr = np.append(hogfv.reshape(1,-1), img_class)
            
            hogfv_arr = np.append(hogfv_arr, [hogfv_class_arr], axis=0)
                    
        cv2.imshow('img', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
            
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


    df = pd.DataFrame(hogfv_arr)

    # if file does not exist write header 
    if not os.path.isfile('dataset.csv'):
        df.to_csv('dataset.csv', header='column_names')
    else: # else it exists so append without writing the header
        df.to_csv('dataset.csv', mode='a', header=False)

def preprocess_image():
    if inp1['state'] == 'disabled':
        dir = inp1.get()
        hogfv_arr = np.empty((0,2305), float)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        img_class = 0
        class_df = pd.DataFrame()

        if not os.path.isfile("./dataset_y.csv"):
            pd.DataFrame({'class': [], 'label': []}).to_csv("dataset_y.csv")
            class_df = pd.read_csv("./dataset_y.csv")
        else:    
            class_df = pd.read_csv("./dataset_y.csv")
        
        for a in os.listdir(dir):
            if class_df.empty:
                img_class = 0
            elif not class_df[class_df['label'] == a].empty:
                img_class = class_df[class_df['label'] == a]['class'].values[0]
            else:
                img_class = class_df['class'].tail(1).values[0] + 1
                
                
            for b in os.listdir(dir + "/" + a):
                print(dir + "/" + a + "/" + b)
                img = imread(dir + "/" + a + "/" + b)

                faces = face_cascade.detectMultiScale(img, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_image =  img[y:y + h, x:x + w]

                    hogfv, hog_image = hog(resize(face_image,(256,256)), orientations=9, pixels_per_cell=(16,16), cells_per_block=(16,16),visualize=True)
                            
                    hogfv_class_arr = np.append(hogfv.reshape(1,-1), img_class)
                    
                    hogfv_arr = np.append(hogfv_arr, [hogfv_class_arr], axis=0)
        
            df = pd.DataFrame(hogfv_arr)

            # if file does not exist write header 
            if not os.path.isfile('dataset.csv'):
                df.to_csv('dataset.csv', header='column_names')
            else: # else it exists so append without writing the header
                df.to_csv('dataset.csv', mode='a', header=False)
            
            if class_df[class_df['label'] == a].empty:
                pd.DataFrame({'class': [img_class], 'label': [str(a)]}).to_csv("./dataset_y.csv", mode='a', header=False)
            
            class_df = pd.read_csv("./dataset_y.csv")

def train():
    df = pd.read_csv('./dataset.csv')

    X_Train = df.iloc[: , 1:-1]
    Y_Train = df.iloc[: , -1]

    model = svm.SVC(kernel='rbf')
    model.fit(X_Train, Y_Train)

    pickle.dump(model, open('finalized_model.sav', 'wb'))

def test():
    class_dict = pd.read_csv("./dataset_y.csv")
    
    model = pickle.load(open('finalized_model.sav', 'rb'))

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        

        faces = face_cascade.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in faces:
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_image =  img[y:y + h, x:x + w]

            hogfv, hog_image = hog(resize(face_image,(256,256)), orientations=9, pixels_per_cell=(16,16), cells_per_block=(16,16),visualize=True)
            
            y_pred = model.predict(hogfv.reshape(1,-1))

            cv2.putText(image, str(class_dict[class_dict['class'] == y_pred[0]]['label'].values[0]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)


                    
        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
        
            
    cap.release()
    cv2.destroyAllWindows()

def file_browser():
    inp3.config(state=NORMAL)
    inp3.delete(0,"end")
    input = filedialog.askopenfile(initialdir=os.curdir)
    inp3.insert(0, input.name)
    inp3.config(state=DISABLED)

def test_image():
    class_dict = pd.read_csv("./dataset_y.csv")
    model = pickle.load(open('finalized_model.sav', 'rb'))

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print(inp3.get())
    img = imread(inp3.get())

    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in faces:
        image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image =  img[y:y + h, x:x + w]

        hogfv, hog_image = hog(resize(face_image,(256,256)), orientations=9, pixels_per_cell=(16,16), cells_per_block=(16,16),visualize=True)
        
        y_pred = model.predict(hogfv.reshape(1,-1))
        
        cv2.putText(image, str(class_dict[class_dict['class'] == y_pred[0]]['label'].values[0]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        
        print(str(class_dict[class_dict['class'] == y_pred[0]]['label'].values[0]))

        cv2.imshow('img', resize(img, (700,1000)))
    
   

window = Tk()

window.title("Face Detection")
window.geometry("600x400")

label1 = Label(window, text="Browse to Dataset: ")
label1.grid(column=1, row=1, sticky="w")

inp1 = Entry(window)
inp1.grid(column=2, row=1)

btn1 = Button(window, text="Browse folder", command=folder_browser)
btn1.grid(column=3, row=1, sticky="nsew")

label2 = Label(window, text="Enter name:")
label2.grid(column=1, row=2, sticky="w" )

inp2 = Entry(window)
inp2.grid(column=2, row=2)

btn2 = Button(window, text="Camera Preprocess", command=preprocess_camera)
btn2.grid(column=1, row=3, sticky="nsew", pady=3, padx=3)

btn3 = Button(window, text="Image Preprocess", command=preprocess_image)
btn3.grid(column=2, row=3, sticky="nsew", pady=3)

btn4 = Button(window, text="Train Model", command=train)
btn4.grid(column=1, row=4, sticky="nsew", pady=3)

label2 = Label(window, text="Select image:")
label2.grid(column=1, row=5, sticky="w" )

inp3 = Entry(window)
inp3.grid(column=2, row=5)

btn5 = Button(window, text="Browse file", command=file_browser)
btn5.grid(column=3, row=5, sticky="nsew", pady=3)

btn6 = Button(window, text="Test Live", command=test)
btn6.grid(column=1, row=6, sticky="nsew", pady=3)

btn7 = Button(window, text="Test Image", command=test_image)
btn7.grid(column=2, row=6, sticky="nsew", pady=3)

window.mainloop()