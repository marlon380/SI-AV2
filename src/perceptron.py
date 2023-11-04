import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('DataAV2.csv', delimiter=',')

X = data[:, :-1]
N,p = X.shape
y = data[:, -1].reshape(N, 1)

"""Gráfico de espelhamento"""

doMirrorGraph = False

if doMirrorGraph:
    halfN = N//2

    plt.scatter(X[0:halfN,0],X[0:halfN,1],color='blue',edgecolors='k')
    plt.scatter(X[halfN:,0],X[halfN:,1],color='red',edgecolors='k')
    plt.xlim(-.25,6.2)
    plt.ylim(-.25,6.2)

    plt.show()

"""Função degrau"""

def sign(u):
    return -1 if u<0 else 1

"""Organização no conjunto de dados para que se tenha a nova dimensão, X ∈ R^(p+1)×N"""

def dataTreatment(X,N):
    X = X.T

    X = np.concatenate((
        -np.ones((1,N)),X
    ))

    #Embaralhamento das amostras
    seed = np.random.permutation(N)
    X_random = X[:,seed]
    y_random = y[seed,:]

    #Divide 80% dos dados para treino
    X_treino = X_random[:, 0:int(N*.8)]
    y_treino = y_random[0:int(N*.8),:]

    #Divide 20% dos dados para teste
    X_teste = X_random[:, int(N*.8):]
    y_teste = y_random[int(N*.8):, :]

    # X_teste = X_teste.T
    p_treino, N_treino = X_treino.shape
    p_teste, N_teste = X_teste.shape

    return X_treino, X_teste, y_treino, y_teste, p_teste, N_teste, p_treino, N_treino

X_treino, X_teste, y_treino, y_teste, p_teste, N_teste, p_treino, N_treino = dataTreatment(X, N)

def dataTraining(X,y,N, p):
    LR = 0.001
    erro = True
    epoch = 0
    R = 100
    #Inicializar o vetor de pesos w(t) com valores nulos ou aleatórios.
    w = np.zeros(p).reshape((p,1))
    while(erro and epoch<R):
        erro = False
        e=0
        for t in range(N):
            x_t = X[:,t].reshape((p,1))
            u_t = (w.T@x_t)[0,0]

            y_t = sign(u_t)
            d_t = y[t,0]
            e_t = int(d_t-y_t)
            w = w + (e_t*x_t*LR)/2
            if(y_t!=d_t):
                erro = True
                e+=1

        epoch+=1

    return w

w = dataTraining(X_treino, y_treino, N_treino, p_treino)

def dataTest(X,y,N,p):
    OUTPUT_PREDICT = []
    OUTPUT_TEST = []

    for t in range(N):
        x_t = X[:,t].reshape((p,1))
        u_t = (w.T@x_t)[0,0]

        y_t = sign(u_t)

        if(y_t==-1):
            OUTPUT_PREDICT.append(0)
        else:
            OUTPUT_PREDICT.append(1)

    for t in range(len(y)):
        if((y)[t, 0]==-1):
                OUTPUT_TEST.append(0)
        else:
            OUTPUT_TEST.append(1)

    return OUTPUT_PREDICT, OUTPUT_TEST

OUTPUT_PREDICT, OUTPUT_TEST = dataTest(X_teste, y_teste, N_teste, p_teste)

def calcClassifiers():
    VP = VN = FP = FN = 0

    for i in range(N_teste):
        if y_teste[i] == 1:
            if OUTPUT_PREDICT[i] == 1:
                VP += 1
            else:
                FN += 1
        else:
            if OUTPUT_PREDICT[i] == 1:
                FP += 1
            else:
                VN += 1
    return VP, VN, FP, FN

def calcPerformance(VP,VN,FP,FN):
  Acuracia = (VP + VN) / (VP + VN + FP + FN)
  Sensibilidade = VP / (VP + FN)
  Especificidade = VN / (VN + FP)

  return Acuracia, Sensibilidade, Especificidade

VP, VN, FP, FN = calcClassifiers()

Acuracia, Sensibilidade, Especificidade = calcPerformance(VP,VN,FP,FN)
print("Verdadeiros Positivos:", VP)
print("Verdadeiros Negativos:", VN)
print("Falsos Positivos:", FP)
print("Falsos Negativos:", FN)
print('Acurácia:',Acuracia, 'Sensibilidade:', Sensibilidade, 'Especificidade:', Especificidade)