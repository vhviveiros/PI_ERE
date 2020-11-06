# %% Nesse exercício, você deve criar um algoritmo que ordena uma lista de números. No entanto, a ordenação é feita da seguinte maneira:

# Aplicando funções de sorteio (random), você deve embaralhar a lista.
# Se após embaralhar, a lista estiver ordenada, o algoritmo encerra.
# Se a lista não estiver ordenada, embaralha novamente.
# Repita a partir do passo 1 até que a lista esteja ordenada.
# OBS: Faça a ordenação ser decrescente (começa do maior valor e termina no menor valor)

import random

lista_exemplo = [7, 3, 1, 4, 5]
reverse = lista_exemplo.copy()
reverse.sort(reverse=True)

while not lista_exemplo == reverse:
    random.shuffle(lista_exemplo)

print(lista_exemplo)
print(reverse)
