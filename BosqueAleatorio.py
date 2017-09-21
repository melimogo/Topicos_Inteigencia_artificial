
# coding: utf-8

#libreria para analizador de argumentos desde la consola al codigo de python
import argparse

#El resto de las librerias
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from Utilidades import visualizador_clasificador


## construir analizador de argumentos
def analizador_argumentos():
    #agregamos descripcion
    analizador = argparse.ArgumentParser(description = 'Clasificador basado en aprendizaje combinado. ')
    ## Agregamos argumentos. Pueden ser lo que quieran
    # --tipo-clasificador = como lo debe de pasar la persona
    # dest = variable para guardar 
    # Required = 
    # Choices = los valores que puede tomar esa variable
    analizador.add_argument('--tipo-clasificador', dest = 'tipo_clasificador',required = True, choices = ['ba','bea'], help = 'Escriba el tipo de bosque que desea; puede ser ba o bea')
    #Retornamos el analizador
    return analizador



## Se puede llamar como programa principal se hace dentro del if
## se puede hacer un generador de funciones, o libreria y va fuera del if


if __name__=='__main__':
    #ejecutamos la función del analisis de argumentos
    argumentos = analizador_argumentos().parse_args()
    tipo_clasificador = argumentos.tipo_clasificador
    
    #cargamos los datos de entrada
    datos = np.loadtxt('datos_bosques_aleatorios.txt', delimiter = ',')
    
    #Tiene 3 tipos de datos
    X,y = datos[:,:-1], datos[:,-1]
    #Separamos los datos por clases
    clase_0 = np.array(X[y==0])
    clase_1 = np.array(X[y==1])
    clase_2 = np.array(X[y==2])
    
    #Generamos la figura de los datos de entrada
    fig = plt.figure()
    # Graficamos cada una de las clases
    plt.scatter(clase_0[:,0],clase_0[:,1], s= 75, facecolors = 'white', edgecolors = 'pink', linewidth = 1, marker = 's')
    plt.scatter(clase_1[:,0],clase_1[:,1], s= 75, facecolors = 'white', edgecolors = 'white', linewidth = 1, marker = 'o')
    plt.scatter(clase_2[:,0],clase_2[:,1], s= 75, facecolors = 'white', edgecolors = 'green', linewidth = 1, marker = '*')
    plt.title('Datos de entrada')
    fig.savefig('Datos de entrada.png')
    
    ## HORA DE DIVIDIR LA BASE DE DATOS
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25, random_state = 5 )
    # definir los parametros del clasificador
    # n_estimators = numero de arboles q vamos a utilizar
    
    parametros = { 'n_estimators':100, 'max_depth': 4, 'random_state':0}
    
    if tipo_clasificador == 'ba':
        # Bosque aleatorio
        clasificador = RandomForestClassifier(**parametros)
    else:
        # Bosque extra aleatorio
        clasificador = ExtraTreesClassifier(**parametros)
        
    # Entrenamos el clasificador
    clasificador.fit(X_train, y_train)
    #visualizamos el clasificador
    visualizador_clasificador(clasificador, X_train, y_train, 'Entrenamiento.png')
    
    #validamos el clasificador
    y_test_pred = clasificador.predict(X_test)
    visualizador_clasificador(clasificador, X_test, y_test_pred, 'Validacion.png')

    ## REPORTE DE CLASIFICACIÓN
    
    nombres_clases = ['clase_0','clase_1','clase_2']
    print('\n' + '#'*70)
    print('\n Entrenamiento \n')
    print(classification_report(y_train, clasificador.predict(X_train), target_names = nombres_clases))
    print('#'*70 + '\n')
    print('\n'+ '#'*70 )
    print('\n Validación \n')
    print(classification_report(y_test,y_test_pred, target_names = nombres_clases))
    print('#'*70 + '\n')
    
    
    
    ## REVISAR LA CONFIABILIDAD DE LAS PREDICCIONES
    # Esta confiabilidad es el intervalo de confianza. 
    # Vamos a generar un pequeno toy_set(Conjunto de jugetes=Datos, para jugar con ellos )
    toy_set = np.array([[5,5],[3,6],[6,4],[7,2],[4,4],[5,2]])
    print('\n La medida de confianbilidad es: ')
    for datos in toy_set:
        probabilidad = clasificador.predict_proba([datos])[0]
        clase_pred = 'clase-' + str(np.argmax(probabilidad))
        print('\nDatos: ',datos)
        print('\nProbabilidad: ',probabilidad)
        print('\nClase: ',clase_pred)
    
    
    