# coding: utf-8
# https://mimc.iaik.tugraz.at/code/mimc/mimc_7.sage

from sage.rings.polynomial.polynomial_gf2x import GF2X_BuildIrred_list

n = 7

K = GF(2 ** n, "a")
K.inject_variables()

# set_random_seed(0)
# constants = [K.fetch_int(0)]
# for i in range(1, num_rounds):
#    constants.append(K.random_element())
constants = [0x00, 0x18, 0x70, 0x22, 0x23]


def mimc_encryption(p, k, num_rounds):
    state = (p + (k + K.fetch_int(constants[0]))) ^ 3
    for i in range(1, num_rounds):
        state = (state + (k + K.fetch_int(constants[i]))) ^ 3
    state = state + k
    return state


# set_random_seed(0)
num_rounds = ceil(n / log(3, 2))
print(num_rounds)
print("Number of rounds:", num_rounds)
k = K.fetch_int(0x55)
print("Key:", hex(k.integer_representation()))
p = K.random_element()
c = mimc_encryption(p, k, num_rounds)
print("Plaintext:", hex(p.integer_representation()))
print("Ciphertext:", hex(c.integer_representation()))
