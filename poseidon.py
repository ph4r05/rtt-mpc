#!/usr/bin/env python

"""
https://gist.github.com/HarryR/f6fadd2c524f61727742002a9221a550

Implements the Poseidon permutation:
Starkad and Poseidon: New Hash Functions for Zero Knowledge Proof Systems
 - Lorenzo Grassi, Daniel Kales, Dmitry Khovratovich, Arnab Roy, Christian Rechberger, and Markus Schofnegger
 - https://eprint.iacr.org/2019/458.pdf
Other implementations:
 - https://github.com/shamatar/PoseidonTree/
 - https://github.com/iden3/circomlib/blob/master/src/poseidon.js
 - https://github.com/dusk-network/poseidon252
 - https://www.poseidon-hash.info
"""

from math import log2
from collections import namedtuple
from pyblake2 import blake2b


SNARK_SCALAR_FIELD = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001


_PoseidonParams = namedtuple('_PoseidonParams', ('p', 't', 'nRoundsF', 'nRoundsP', 'seed', 'e', 'constants_C', 'constants_M'))


def poseidon_params(p, t, nRoundsF, nRoundsP, seed, e, constants_C=None, constants_M=None):
    assert nRoundsF % 2 == 0 and nRoundsF > 0
    assert nRoundsP > 0
    assert t >= 2
    assert isinstance(seed, bytes)

    M = 128  # security target, in bits
    n = log2(p)
    N = n * t

    if p % 2 == 3:
        assert e == 3
        grobner_attack_ratio_rounds = 0.32
        grobner_attack_ratio_sboxes = 0.18
        interpolation_attack_ratio = 0.63
    elif p % 5 != 1:
        assert e == 5
        grobner_attack_ratio_rounds = 0.21
        grobner_attack_ratio_sboxes = 0.14
        interpolation_attack_ratio = 0.43
    else:
        # XXX: in other cases use, can we use 7?
        raise ValueError('Invalid p for congruency')

    # Verify that the parameter choice exceeds the recommendations to prevent attacks
    # iacr.org/2019/458 § 3 Cryptanalysis Summary of Starkad and Poseidon Hashes (pg 10)
    # Figure 1
    #print('(nRoundsF + nRoundsP)', (nRoundsF + nRoundsP))
    #print('Interpolation Attackable Rounds', ((interpolation_attack_ratio * min(n, M)) + log2(t)))
    assert (nRoundsF + nRoundsP) > ((interpolation_attack_ratio * min(n, M)) + log2(t))
    # Figure 3
    #print('grobner_attack_ratio_rounds', ((2 + min(M, n)) * grobner_attack_ratio_rounds))
    assert (nRoundsF + nRoundsP) > ((2 + min(M, n)) * grobner_attack_ratio_rounds)
    # Figure 4
    #print('grobner_attack_ratio_sboxes', (M * grobner_attack_ratio_sboxes))
    assert (nRoundsF + (t * nRoundsP)) > (M * grobner_attack_ratio_sboxes)

    # iacr.org/2019/458 § 4.1 Minimize "Number of S-Boxes"
    # In order to minimize the number of S-boxes for given n and t, the goal is to and
    # the best ratio between RP and RF that minimizes:
    #   number of S-Boxes = t · RF + RP
    # - Use S-box x^q
    # - Select R_F to 6 o rhigher
    # - Select R_P that minimizes tRF +RP such that no inequation (1),(3),(4),(5) is satisfied.

    if constants_C is None:
        constants_C = list(poseidon_constants(p, seed + b'_constants', nRoundsF + nRoundsP))
    if constants_M is None:
        constants_M = poseidon_matrix(p, seed + b'_matrix_0000', t)

    # iacr.org/2019/458 § 4.1 6 SNARKs Application via Poseidon-π
    # page 16 formula (8) and (9)
    n_constraints = (nRoundsF * t) + nRoundsP
    if e == 5:
        n_constraints *= 3
    elif e == 3:
        n_constraints *= 2
    #print('n_constraints', n_constraints)

    return _PoseidonParams(p, t, nRoundsF, nRoundsP, seed, e, constants_C, constants_M)


def H(arg):
    if isinstance(arg, int):
        arg = arg.to_bytes(32, 'little')
    # XXX: ensure that (digest_size*8) >= log2(p)
    hashed = blake2b(data=arg, digest_size=32).digest()
    return int.from_bytes(hashed, 'little')


def poseidon_constants(p, seed, n):
    assert isinstance(n, int)
    for _ in range(n):
        seed = H(seed)
        yield seed % p


def poseidon_matrix(p, seed, t):
    """
    iacr.org/2019/458 § 2.3 About the MDS Matrix (pg 8)
    Also:
     - https://en.wikipedia.org/wiki/Cauchy_matrix
    """
    c = list(poseidon_constants(p, seed, t * 2))
    return [[pow((c[i] - c[t+j]) % p, p - 2, p) for j in range(t)]
            for i in range(t)]


DefaultParams = poseidon_params(SNARK_SCALAR_FIELD, 6, 8, 57, b'poseidon', 5)


def poseidon_sbox(state, i, params):
    """
    iacr.org/2019/458 § 2.2 The Hades Strategy (pg 6)
    In more details, assume R_F = 2 · R_f is an even number. Then
     - the first R_f rounds have a full S-Box layer,
     - the middle R_P rounds have a partial S-Box layer (i.e., 1 S-Box layer),
     - the last R_f rounds have a full S-Box layer
    """
    half_F, nRoundsF = params.nRoundsF // 2, params.nRoundsP
    e, p = params.e, params.p
    if i < half_F or i >= (half_F + params.nRoundsP):
        for j, _ in enumerate(state):
            state[j] = pow(_, e, p)
    else:
        state[0] = pow(state[0], e, p)


def poseidon_mix(state, M, p):
    """
    The mixing layer is a matrix vector product of the state with the mixing matrix
     - https://mathinsight.org/matrix_vector_multiplication
    """
    return [ sum([M[i][j] * _ for j, _ in enumerate(state)]) % p
             for i in range(len(M)) ]


def poseidon(inputs, params=None):
    if params is None:
        params = DefaultParams
    assert isinstance(params, _PoseidonParams)
    assert len(inputs) > 0 and len(inputs) < params.t
    state = [0] * params.t
    state[:len(inputs)] = inputs
    for i, C_i in enumerate(params.constants_C):
        state = [_ + C_i for _ in state]  # ARK(.)
        poseidon_sbox(state, i, params)
        state = poseidon_mix(state, params.constants_M, params.p)
    return state[0]


if __name__ == "__main__":
    assert DefaultParams.constants_C[0] == 14397397413755236225575615486459253198602422701513067526754101844196324375522
    assert DefaultParams.constants_C[-1] == 10635360132728137321700090133109897687122647659471659996419791842933639708516
    assert DefaultParams.constants_M[0][0] == 19167410339349846567561662441069598364702008768579734801591448511131028229281
    assert DefaultParams.constants_M[-1][-1] == 20261355950827657195644012399234591122288573679402601053407151083849785332516
    assert poseidon([1,2], DefaultParams) == 12242166908188651009877250812424843524687801523336557272219921456462821518061