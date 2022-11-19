import cv2
import numpy as np

#cone sta funcion reviso si un punto ya esta en alguna de mis listas
def isInTheList(elemento,arreglo):
    for i in arreglo:
        if np.array_equal(i,elemento):
            return True            
    return False  
    
mapa=cv2.imread('mapa.png')
gray = cv2.cvtColor(mapa,cv2.COLOR_BGR2GRAY)
cv2.imshow('mapa',gray)
ret,th1 = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
kernel = np.ones((11,11), np.uint8) 
th1 = cv2.dilate(th1,kernel,1)
kernel = np.ones((11,11), np.uint8) 
th1 = cv2.erode(th1,kernel,1)
th1 = cv2.GaussianBlur(th1,(5,5),cv2.BORDER_DEFAULT) 
cv2.imshow('thres',th1)
cv2.waitKey(0)


#Aplico la deteccion de Esquinas de Harris. para mas informacion consulten https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
dst = cv2.cornerHarris(th1,2,3,0.05)
ret, dst = cv2.threshold(dst,0.04*dst.max(),255,0)
dst = np.uint8(dst)
ret,th2 = cv2.threshold(th1,235,255,cv2.THRESH_BINARY)
th2 = cv2.dilate(th2,kernel,1)
#aqui devuelvo la imagen binarizada a tres canales
th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)

cv2.imshow('thres',th2)


cv2.waitKey(0)