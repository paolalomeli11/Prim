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


def algoritmoPrim(vertices, mAdy):
    camino = []
    tam = len(vertices)
    visitados = np.full(tam, False)
    visitados[0] = True
    INF = 999999
    count = 0
    found = False

    print(matAdy)

    while count < tam - 1:
        a = 0
        b = 0
        min = INF
        found = False

        for m in range(tam):
            if visitados[m]:
                for n in range(tam):
                    if ((not visitados[n]) and mAdy[m][n]):
                        print("xasda")
                        if min > mAdy[m][n]:
                            min = mAdy[m][n]
                            a = m
                            b = n
                            found = True

        if found:
            camino.append([vertices[a],vertices[b]])
        visitados[b] = True
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
verticesConectados=[]
matAdy = []
aristas=[]
aristas2=[]
conectado = False

#aqui voy a buscar cuales son las esquinas que estan conectadas
for h in range(len(aux1)):
    i=aux1[h]
    ady = []
    for k in range(h,len(aux2)):
        j=aux2[k]
        conectado = False
        if not (i==j).all():
            if not isInTheList([i,j], aristas) and comprobarMedios(i,j,th2):
                costo = int(math.sqrt((j[0]-i[0])**2 + (j[1]-i[1])**2))
                aristas.append([i,j,costo])
                ady.append(costo)
                conectado = True
            else:
                ady.append(0)
        else:
            ady.append(0)
    if conectado:
        verticesConectados.append(i)
    matAdy.append(ady)


aristas2 = algoritmoPrim(verticesConectados, matAdy)

for arista in aristas2:
    cv2.line(th2, tuple(arista[0]), tuple(arista[1]), (0,255,0), 1)

for point in vertices:
    cv2.circle(th2,(point[0], point[1]), 5, (255,0,0), -1)    
    cv2.waitKey(0)

cv2.imshow('points',th2)
cv2.waitKey(0)