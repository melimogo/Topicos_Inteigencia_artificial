
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np

def visualizador_clasificador(clasificador, X,y, titulo):
    #definimos valores maximos y minimos de la malla que vamos a graficar
    min_x,max_x = X[:,0].min()-1.0, X[:,0].max()+1.0
    min_y,max_y = X[:,1].min()-1.0, X[:,1].max()+1.0
    
    #definir el paso de la malla
    paso = 0.1
    
    #definimos la malla
    x_vals,y_vals =np.mgrid[min_x:max_x:paso, min_y:max_y:paso]
    
    #np.c: para concatenar los valores
    #np.ravel: coloca todos los datos de varios arreglos o varias dimensiones en 1 sola
    
    #corremos el clasificador sobre la malla
    resultados = clasificador.predict(np.c_[x_vals.ravel(),y_vals.ravel()])
    # reordenamos la salida para que nos quede en forma de malla
    resultados = resultados.reshape(x_vals.shape)
    
    #creamos la figura 
    fig = plt.figure()
    # Elegimos la paleta de colores(colormap)
    plt.pcolormesh(x_vals,y_vals, resultados, cmap=plt.cm.Pastel1)
    
    #Ubicamos los puntos a clasificar
    # X[:,0] es la coordenada en el eje x
    # X[:,1] es la coordenada y
    # c(color) define el color (c=etiquetas)
    # s = es el tamaño de la letra(size)
    # edgcolors = define el borde
    # linewidth = define el ancho de las lineas
    # cmap = define el mapa de color
    plt.scatter(X[:,0],X[:,1],c=y,s=75, edgecolors='black',linewidth=1,cmap=plt.cm.Set3)
    
    #fijamos los limites para los ejes x e y
    plt.xlim(x_vals.min(),x_vals.max())
    plt.ylim(y_vals.min(),y_vals.max())
    
    ## guardar la figura
    fig.savefig(titulo)
    

