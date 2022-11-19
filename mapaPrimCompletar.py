import cv2
import numpy as np
import math


#cone sta funcion reviso si un punto ya esta en alguna de mis listas
def isInTheList(elemento,arreglo):
    for i in arreglo:
        if np.array_equal(i,elemento):
            return True            
    return False

def comprobarRecta(coord1, coord2, imagen):
    m = (coord2[1] - coord1[0]) / (coord2[0] - coord1[0])
    b = coord1[1] - m * coord1[0]

    for x in range(coord1[0], coord2[0]):
        y = m * x + b
        y = int(y)

        print(np.shape(imagen))

        if np.all(imagen[y, x] != 255):
            break
            return False

    if np.all(th2[iY, iX] == 255):
        pass


def intermediates(p1, p2,imagen):

    nb_points = p2[0] - p1[0]
    nb_points = abs(nb_points)

    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    lista = []
    conectado = True
    x = 0
    y = 0

    for i in range(1, nb_points+1):
        x = int(p1[0] + i * x_spacing)
        y = int(p1[1] +  i * y_spacing)

        if np.any(th2[y, x] != 255):
            conectado = False
            break

    return conectado

def algoritmoPrim(verticesConectados, vertices):
    visitados = []
    costos = verticesConectados
    aristas = []

    vert = 0
    col = 0

    while True:

        print('vertIcE:', vert)

        if isInTheList(vert,visitados):
            print('me rompo')
            break

        visitados.append(vert)

        m = costos[0][0]
        for i, c in enumerate(costos):

            if c[0] < m:

                if isInTheList(c[1],visitados):
                    print("caso 1")
                    vert = c[2]
                    m = c[0]
                    col = [1]
                elif isInTheList(c[2],visitados):
                    print("caso 2")
                    vert = c[1]
                    m = c[0]
                    col = c[2]
                     

        coord1 = vertices[vert]
        coord2 = vertices[col]
        aristas.append([coord1,coord2])
        #del costos[vert]

    return aristas


#para cargar el mapa
mapa=cv2.imread('mapa.png')
#pasamos la imagen a escala de grises
gray = cv2.cvtColor(mapa,cv2.COLOR_BGR2GRAY)
#muestro la imagen en escala de grises
#cv2.imshow('mapa',gray)
#obtengo un binarizacion en blaco todos lo pixeles cuyo valor en sea entre 254 y 255
ret,th1 = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
#hago un kernel de 11x11 de unos. Los Kernels se acostumbra hacerse de tamaño no par y cuadrados
#para que se den una idea algo asi:

kernel = np.ones((11,11), np.uint8) 
#aplico un filtro de dilatacion. Este filtro hace que los puntos los puntos blancos se expandan 
#probocando que algunos puntitos negros desaparecan #le pueden hacer un cv.imshow para que vean el resultado
th1 = cv2.dilate(th1,kernel,1)
kernel = np.ones((11,11), np.uint8) 
#Despues aplico uno de erosion que hace lo opuesto al de dilatacion
th1 = cv2.erode(th1,kernel,1)
#aplico un flitro gausiando de 5x5  para suavisar los bordes 
th1 = cv2.GaussianBlur(th1,(5,5),cv2.BORDER_DEFAULT) 
#muestro como queda mi mapa
#cv2.imshow('thres',th1)
#Aplico la deteccion de Esquinas de Harris. para mas informacion consulten https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
dst = cv2.cornerHarris(th1,2,3,0.05)
ret, dst = cv2.threshold(dst,0.04*dst.max(),255,0)
dst = np.uint8(dst)
ret,th2 = cv2.threshold(th1,235,255,cv2.THRESH_BINARY)
th2 = cv2.dilate(th2,kernel,1)
#aqui devuelvo la imagen binarizada a tres canales
th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst,30, cv2.CV_32S)
vertices=np.int0(centroids)

aux1=vertices
aux2=vertices
verticesConectados=[]
aristas=[]
interP=[]
#aqui voy a buscar cuales son las esquinas que estan conectadas
for h in range(len(aux1)):
    i=aux1[h]
    for k in range(h,len(aux2)):
        j=aux2[k]
        if not (i==j).all(): 

            con = intermediates(i,j, th2)

            if not isInTheList([i,j], aristas) and con:
                aristas.append([i,j])
                costo = math.sqrt((j[0]-i[0])**2 + (j[1]-i[1])**2)
                verticesConectados.append([int(costo),h,k])

aristas2 = algoritmoPrim(verticesConectados, vertices)


for arista in aristas2:
    cv2.line(th2, tuple(arista[0]), tuple(arista[1]), (0,255,0), 1)
    cv2.waitKey(0)


#aqui pinto los puntos de las esquinas que son circulos de de radio de 5 pixeles, el -1 indica que van rellenados los circulos
#point tiene la forma [fila, columna]
for point in vertices:
    cv2.circle(th2,(point[0], point[1]), 5, (255,0,0), -1)    


#aqui muestro como quedo de chingon el grafo
cv2.imshow('points',th2)

cv2.waitKey(0)
