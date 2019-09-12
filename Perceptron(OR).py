# -*- coding: utf-8 -*-
# Perceptron (Input = OR)

from random import choice
from numpy import array, dot, random

# Função: Se ( w.x + b ) <= 0, retorna 0. Se não, retorna 1.
funcao = lambda x: 0 if x <= 0 else 1

# Entrada: Tabela OR
vetor_treinamento = [
    (array([0, 0, 1]), 0),  # ( [x1,x2,bias],  Valor_Esperado)
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 1),
    (array([1, 1, 1]), 1),
]

# Pesos aleatórios
w = random.rand(3)
print('Pesos Iniciais = ', w)
lista_erros = []
eta = 0.1
n = 50

# Treinamento
for i in range(n):
    x, valor_esperado = choice(vetor_treinamento)

    produto = dot(w, x)

    erro = valor_esperado - funcao(
        produto)  # Se o valor_esperado for diferente do valor da função, ou seja, 0 ou 1, deve-se atualizar o W

    lista_erros.append(erro)  # Guardar o erro no final da lista
    # print('w anterior = ',w )
    w += eta * erro * x  # Atualizar os valores dos pesos, até que o erro se anule e ocorra a convergência do Perceptron

"""  
    print('*********')
    print('x = ',x )
    print('w = ',w )
    print('Valor Esperado = ',valor_esperado)
    print('Produto = ',produto)
    print('Funcao(produto) = ',funcao(produto))
    print('Erro = ',erro)
    print('*********')
    print('\n')


print('Pesos Finais = ',w)
"""
# Pós-Treinamento
for x, _ in vetor_treinamento:
    produto = dot(x, w)
    print("{}: {} -> {}".format(x[:2], produto, funcao(produto)))

from pylab import plot, ylim

ylim([-1, 1])
plot(lista_erros)

