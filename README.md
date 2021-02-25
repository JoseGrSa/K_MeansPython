# Semana Tec TC1002S.2
***Integrantes***
- Mateo Gonzalez Cosio - A01023938
- Jose Salgado - A01023661
- Carolina Ortega - A01025254
- Rodrigo Aviles - A01023707

In this challenge we implemented the k-means clustering algorithm in Python. Using a shared repository on GitHub we were able to work remotely.  
The code is implemented with the iris data that was downloaded from the internet.

# How the code works

Our code starts with a distance function in which two lists are received and the function returns the value of the Euclidean distance between them. When we started programming this part, we first opted for the lists to be set by the user but the problem was that there were cases where the program copied lists and substituted them elsewhere. We decided to remove these inputs and use direct data instead.

```python
def distance(list1, list2):
    if len(list1) != len(list2):
        return -1
    d_squared = 0
    for v1, v2 in zip(list1, list2):
        d_squared += (v2 - v1)**2

    return d_squared**(1/2)

```

We used a function get_clusters that uses the points as a list of (x,y) and the center we want to get will be a list of k lists (x,y).  Each point is compared with all the centers and the distance between them is stored. A for is used so that the information of the points where the centers are is added to the empty list. The selected points will be those that are closest to the centers.

```python

def get_clusters(puntos, centros):
    # Puntos es un lista de puntos (x,y)
    # Centro es una lista de k listas (x,y)

    clusters = [[] for _ in range(0, len(centros))]

    for punto in puntos:
        # Tengo un punto que lo quiero comparar contra todos los centros
        # Aqui se van a guardar todas las distancias entre mi punto y todos los centros
        p_vs_c = []
        for centro in centros:
            d = distance(centro, punto)
            p_vs_c.append(d)
        # la minima distancia entre mi punto y todos los centros es el la key del centro correcto
        pos = p_vs_c.index(min(p_vs_c)) # La posicion del centro en clusters
        clusters[pos].append(punto)

    return clusters
```
    
The function center receives the list of k lists named as cluster. With this function we want to obtain the points where the new centers will be , after calculating the average is added to the new list.

```python

def center(cluster):
    for i in cluster:
        cluster_f = []
        for i in range(len(cluster)):
            avg = np.mean(cluster[i], axis=0)
            #avgr = avg.astype(int) Turns list to ints
            cluster_f.append(avg.tolist())
    return cluster_f
    
```
    
When implementing k_means, it was implemented manually with random and with iris.
The k_means function with random generates all points, centers and clusters randomly from points that are in the coordinates to be generated. The user selects the number of coordinates to generate, the number of times he wants to reset the centers (iterations) and the number of centers he wants to obtain.





The centers are representative points since they receive the information from the list of points, then we replace those points with the average of themselves in order to take them as the new centers.

The value of k is taken as the number of clusters we obtained, k is implemented in the k-means algorithm in order to complete the function.

The value of the centers does not have a major impact on the performance of the variables and functions because if it is higher or lower the only thing the center does is to take those values to define new points and centers regardless of their sign.

In all cases every time there is an iteration the resulting numbers are always different even if they go through the same methods and functions. For this reason in the final results our centers have different distances, the distance of our centers is 12.31, the closest centers are 2 and 3 and the farthest ones are 3 and 1.

The centers that are shown in the graphs as final result are calculated thanks to the set of data already analyzed, we can say that for this reason to arrive at them it is indispensable to first calculate all the functions with their respective data and methods.

## Plot
The points and centers are plotted for better visualization of the data using the matplot.lib library. 

![bd4ef2f1-cf31-42c6-914c-01a36c0c6642](https://user-images.githubusercontent.com/71286113/93649243-24da7880-f9d1-11ea-8ee4-f86b737c8d26.jpg)

## KMeans Implementation

## Our implementation

With the functions proposed above we processed the information found in the set the data "iris" and obtained the following graph.

![Graph_basic](https://user-images.githubusercontent.com/71286113/93691943-08fad380-fab2-11ea-8314-730e8000eeea.png)

## Scikit Learn
To ensure that the classification of our data is correct we took the task of using the **scikit-learn** library to classify our data.

We also used the k-means method to classify our data into 3 groups.

This was the resulting graph:

![Graph_sklearn](https://user-images.githubusercontent.com/71286113/93691951-3e9fbc80-fab2-11ea-9993-c1b73c4cc117.png)
