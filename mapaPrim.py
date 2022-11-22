import cv2
import numpy as np
import math

#cone sta funcion reviso si un punto ya esta en alguna de mis listas
def isInTheList(elemento,arreglo):
    for i in arreglo:
        if np.array_equal(i,elemento):
            return True            
    return False

def comprobarMedios(i, j, th2):
 
    conectado = False

    c = (i[0] + j[0])//2, (i[1] + j[1])//2
    l = (i[0] + c[0])//2, (i[1] + c[1])//2
    l1 = (i[0] + l[0])//2, (i[1] + l[1])//2
    l2 = (l[0] + c[0])//2, (l[1] + c[1])//2
    u = (c[0] + j[0])//2, (c[1] + j[1])//2
    u1 = (u[0] + j[0])//2, (u[1] + j[1])//2
    u2 = (c[0] + u[0])//2, (c[1] + u[1])//2

    b1 = np.all(th2[c[1], c[0]] == 255)
    b2 = np.all(th2[l[1], l[0]] == 255)
    b3 = np.all(th2[u[1], u[0]] == 255)
    b4 = np.all(th2[l1[1], l1[0]] == 255)
    b5 = np.all(th2[u1[1], u1[0]] == 255)
    b6 = np.all(th2[l2[1], l2[0]] == 255)
    b7 = np.all(th2[u2[1], u2[0]] == 255)

    if(b1 and b2 and b3 and b4 and b5 and b6 and b7):
        conectado = True
    else:
        conectado = False

    return conectado

def algoritmoPrim(vertices, mAdy, visitados):
    camino = []
    tam = len(vertices)
    first_key = list(visitados)[0]
    visitados[first_key] = True
    MAX = 999999
    count = 0

    while count < tam - 1:
        min = MAX
        for m in visitados:
            if visitados[m]:
                for n,ady in enumerate(mAdy[m]):
                    if ady[0] < min:
                        ib = ady[2]
                        k = str(ib[0]) + ',' + str(ib[1])
                        #Para evitar bucles
                        if(not visitados[k]):
                            eX = m
                            eY = n
                            min = ady[0]
        
        a = mAdy[eX][eY][1]
        b = mAdy[eX][eY][2]

        key1 = str(a[0]) + ',' + str(a[1]) 
        key2 = str(b[0]) + ',' + str(b[1])
        visitados[key2] = True
        camino.append([vertices[key1],vertices[key2]])
        count += 1

    return camino
    
mapa=cv2.imread('mapa.png')
gray = cv2.cvtColor(mapa,cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)

kernel = np.ones((11,11), np.uint8) 
th1 = cv2.dilate(th1,kernel,1)
kernel = np.ones((11,11), np.uint8) 
th1 = cv2.erode(th1,kernel,1)
th1 = cv2.GaussianBlur(th1,(5,5),cv2.BORDER_DEFAULT) 
dst = cv2.cornerHarris(th1,2,3,0.05)
ret, dst = cv2.threshold(dst,0.04*dst.max(),255,0)
dst = np.uint8(dst)
ret,th2 = cv2.threshold(th1,235,255,cv2.THRESH_BINARY)
th2 = cv2.dilate(th2,kernel,1)
th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst,30, cv2.CV_32S)
vertices=np.int0(centroids)

aux1=vertices
aux2=vertices
verticesConectados={}
aristas = {}
visitados = {}

#aqui voy a buscar cuales son las esquinas que estan conectadas
for h in range(len(aux1)):
    i=aux1[h]
    ady = []
    conectado = False
    for k in range(len(aux2)):
        j=aux2[k]
        if not (i==j).all():
            if comprobarMedios(i,j,th2):
                costo = int(math.sqrt((j[0]-i[0])**2 + (j[1]-i[1])**2))
                conectado = True
                ady.append([costo,i,j])
        
    if conectado:
        key = str(i[0]) + ',' + str(i[1])
        verticesConectados.update({key:i})
        visitados.update({key:False})
        aristas.update({key:ady})

aristas2 = algoritmoPrim(verticesConectados, aristas, visitados)

for arista in aristas2:
    cv2.line(th2, tuple(arista[0]), tuple(arista[1]), (255,182,193), 2)

for point in vertices:
    cv2.circle(th2,(point[0], point[1]), 5, (255,20,147), -1)    
    cv2.waitKey(0)

cv2.imshow('points',th2)
cv2.waitKey(0)