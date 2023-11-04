import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('DataAV2.csv', delimiter=',')

X = data[:, :-1]
N,p = X.shape
y = data[:, -1].reshape(N, 1)

"""Função de erro Quadrático Médio"""
def EQM(X,y,w):
    seq = 0
    us = []
    p,N = X.shape
    for t in range(X.shape[1]):
        x_t = X[:,t].reshape(X.shape[0],1)
        u_t = w.T@x_t
        us.append(u_t)
        d_t = y[t,0]
        seq+= (d_t - u_t)**2
    
    return seq/(2*X.shape[1])

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
    lr = 1e-2
    pr = .0000001

    maxEpoch = 1000

    epoch = 0
    EQM1 = 1
    EQM2 = 0

    w = np.zeros(p).reshape((3,1))

    while(epoch<maxEpoch and abs(EQM1-EQM2)>pr):
        EQM1 = EQM(X,y,w)
        for t in range(N):
            x_t = X[:,t].reshape(3,1)
            u_t = w.T@x_t
            d_t = y[t,0]
            e_t = (d_t-u_t)
            w = w + (lr*e_t*x_t)

        epoch+=1
        EQM2 = EQM(X,y,w)
    return w

w = dataTraining(X_treino, y_treino, N_treino, p_treino)

def dataTest(X,y,N,p):
    OUTPUT_PREDICT = []
    OUTPUT_TEST = []

    for t in range(N):
        x_t = X[:,t].reshape((p,1))
        u_t = w.T @ x_t

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