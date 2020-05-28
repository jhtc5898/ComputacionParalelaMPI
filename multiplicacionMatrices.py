#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:51:48 2020

@author: Christian
"""
import numpy as np
from mpi4py import MPI 
import random
from time import time

tiempo_inicial = time()

tam=1500
A = np.zeros((tam, tam))
B = np.zeros((tam, tam))
C = np.zeros((tam, tam))


def llenaMatrices(tamMatriz,matriz):
	for i in range(0, tamMatriz):
		for j in range(0, tamMatriz):
			matriz[i][j]=random.randint(0, tamMatriz*tamMatriz)
	return matriz
	
A=llenaMatrices(tam,A)
B=llenaMatrices(tam,B)


#Parte Secuencial
def multiplicaMatrices(matrizA,matrizB):
	matrizResultante = np.zeros((len(A), len(B)))
	aux=0
	for i in range(0,len(A)):
		for j in range(0,len(B[0])):
				for k in range(0,len(B)):
					matrizResultante[i][j] += A[i][k] * B[k][j]
	return matrizResultante

C=multiplicaMatrices(A,B)

"""print("Matriz A")
print(A)
print()
print("Matriz B")
print(B)
print()
print("Matriz resultante")
print(C)"""

tiempo_final = time() 
 
tiempo_ejecucion = tiempo_final - tiempo_inicial
 
print ('El tiempo de ejecucion fue:')
print(tiempo_ejecucion)

"""
#Parte paralela

tiempo_inicial = time()

tam=500

NRA=tam
NCA=tam
NCB=tam
MASTER=0
FROM_MASTER=1
FROM_WORKER=2


a = []
b = []
c = []

for i in range(NRA):
    c.append([])
    for j in range(NCB):
        c[i].append(0)

MPI.Init

comm = MPI.COMM_WORLD 
numtasks = comm.size 
taskid = comm.Get_rank()

numworkers = numtasks-1


if (taskid == MASTER):
    
    for i in range(NRA):
        a.append([])
        for j in range (NCA):
            a[i].append(i+j)
    for i in range(NCA):
        b.append([])
        for j in range (NCB):
            b[i].append(i*j)

    print("Number of worker tasks = %d\n",numworkers);
    
    averow = NRA//numworkers
    extra = NRA%numworkers
    offset = 0
    mtype = FROM_MASTER;
    
    for destino in range(numworkers):
        
        if(destino+1<=extra):
            rows=averow+1
        else:
            rows=averow
        
        print("   sending %d rows to task %d\n",rows,destino+1)
        comm.send(offset,dest=destino+1,tag=mtype)
        comm.send(rows,dest=destino+1,tag=mtype)
        comm.send(a[offset:rows+offset],dest=destino+1,tag=mtype)
        comm.send(b,dest=destino+1,tag=mtype)
        offset = offset + rows
        
    mtype = FROM_WORKER
    
    for i in range(numworkers):
        
        fuente = i
        offset=comm.recv(source=fuente+1,tag=mtype)
        rows=comm.recv(source=fuente+1,tag=mtype)
        c=comm.recv(source=fuente+1,tag=mtype)
        #b=comm.recv(source=fuente+1,tag=mtype)


    print("Here are the first 30 rows of the result (C) matrix\n");
    for i in range(30):
        print(""); 
        for j in range(NCB): 
            print (c[i][j])
     
    print ("")
        
    
     

if (taskid > MASTER):
    mtype = FROM_MASTER;
        
    offset=comm.recv(source=MASTER,tag=mtype)
    rows=comm.recv(source=MASTER,tag=mtype)
    a=comm.recv(source=MASTER,tag=mtype)   
    b=comm.recv(source=MASTER,tag=mtype)
        
    for k in range(NCB):
        for i in range(rows):
            for j in range(NCA):
                c[i][k] = c[i][k] + a[i][j] * b[j][k];
    mtype = FROM_WORKER;
        
    comm.send(offset, dest=MASTER,tag=mtype)
    comm.send(rows, dest=MASTER,tag=mtype)
    comm.send(c, dest=MASTER,tag=mtype)
    
    
    
MPI.Finalize
       
tiempo_final = time() 
 
tiempo_ejecucion = tiempo_final - tiempo_inicial
 
print ('El tiempo de ejecucion fue:')
print(tiempo_ejecucion)"""






