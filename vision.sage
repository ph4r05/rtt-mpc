load('common.sage')


class MarvellousParams(object):
    def __init__(self, field, r, c, num_rounds):
        self.field = field
        self.r = r
        self.c = c
        self.m = m = r + c
        self.output_size = c
        assert self.output_size <= r
        self.num_rounds = num_rounds
        self.MDS = generate_mds_matrix('MarvellousMDS', field, m)
        self.round_constants = [
            vector(generate_round_constant('MarvellousK', field, m * i + j)
                   for j in range(m))
            for i in range(2 * num_rounds + 1)]
        if field.is_prime_field():
            self.sbox1 = lambda x: x**3
            inv_3 = int(Zmod(field.order()-1)(1/3))
            self.sbox0 = lambda x: x**inv_3
        else:
            assert field.characteristic() == 2
            B0, B_linear = generate_affine_transformation(field)

            def B_affine(x): return B_linear(x) + B0
            a = field.gen()
            B_inv = matrix(
                [vector(B_linear(a**i)) for i in range(field.degree())]
            ).inverse()

            def B_affine_inv(x): return field(vector(x + B0) * B_inv)
            rand_elt = field.random_element()
            assert B_affine_inv(B_affine(rand_elt)) == rand_elt
            assert B_affine(B_affine_inv(rand_elt)) == rand_elt
            self.sbox1 = lambda x: B_affine(x**(field.order()-2))
            self.sbox0 = lambda x: B_affine_inv(x**(field.order()-2))


def generate_affine_transformation(field):
    """
    Returns a field element offset B0 and a polynomial of the form
      P(X) = B1 * X + B2 * X**2 + B3 * X**4
    which represents an invertible linear transformation (over GF(2)).
    """
    X = PolynomialRing(field, name='X').gen()
    for attempt in range(100):
        coefs = [generate_round_constant('MarvellousB', field, attempt * 4 + i)
                 for i in range(4)]
        # Check that all coefficients are not in any subfield.
        if any(coef.minimal_polynomial().degree() != field.degree()
               for coef in coefs):
            continue

        # Check that the linear transformation is invertible, by checking that
        # p(X)/X has no roots in the field.
        p_div_x = coefs[1] + coefs[2] * X + coefs[3] * X**3
        if len((p_div_x).roots()) > 0:
            continue

        return coefs[0], X * p_div_x
    raise Exception('Failed to find an affine transformation')


# Parameter sets.
Rescue_S45a = MarvellousParams(field=F91, r=2, c=1, num_rounds=10)
Vision_S45a = MarvellousParams(field=Bin91, r=2, c=1, num_rounds=10)
Rescue_S45b = MarvellousParams(field=F91, r=10, c=1, num_rounds=10)
Vision_S45b = MarvellousParams(field=Bin91, r=10, c=1, num_rounds=10)

Rescue_S80a = MarvellousParams(field=F81, r=2, c=2, num_rounds=10)
Vision_S80a = MarvellousParams(field=Bin81, r=2, c=2, num_rounds=10)
Rescue_S80b = MarvellousParams(field=F161, r=2, c=1, num_rounds=14)
Vision_S80b = MarvellousParams(field=Bin161, r=2, c=1, num_rounds=10)
Rescue_S80c = MarvellousParams(field=F161, r=10, c=1, num_rounds=10)
Vision_S80c = MarvellousParams(field=Bin161, r=10, c=1, num_rounds=10)

Rescue_S128a = MarvellousParams(field=F125, r=2, c=2, num_rounds=16)
Vision_S128a = MarvellousParams(field=Bin127, r=2, c=2, num_rounds=12)
Rescue_S128b = MarvellousParams(field=F253, r=2, c=1, num_rounds=22)
Vision_S128b = MarvellousParams(field=Bin255, r=2, c=1, num_rounds=16)
Rescue_S128c = MarvellousParams(field=F125, r=10, c=2, num_rounds=10)
Vision_S128c = MarvellousParams(field=Bin127, r=10, c=2, num_rounds=10)
Rescue_S128d = MarvellousParams(field=F61, r=8, c=4, num_rounds=10)
Vision_S128d = MarvellousParams(field=Bin63, r=8, c=4, num_rounds=10)
Rescue_S128e = MarvellousParams(field=F253, r=10, c=1, num_rounds=10)
Vision_S128e = MarvellousParams(field=Bin255, r=10, c=1, num_rounds=10)

Rescue_S256a = MarvellousParams(field=F125, r=4, c=4, num_rounds=16)
Vision_S256a = MarvellousParams(field=Bin127, r=4, c=4, num_rounds=12)
Rescue_S256b = MarvellousParams(field=F125, r=10, c=4, num_rounds=10)
Vision_S256b = MarvellousParams(field=Bin127, r=10, c=4, num_rounds=10)


# Evaluate the block cipher.
def block_cipher(state, params):
    """
    Evaluates the block cipher with key=0 in forward direction.
    """
    state += params.round_constants[0]
    for r in range(2 * params.num_rounds):
        sbox = params.sbox0 if r % 2 == 0 else params.sbox1
        for i in range(params.m):
            state[i] = sbox(state[i])

        state = params.MDS * state + params.round_constants[r + 1]

    return state


def marvellous_hash(inputs, params):
    return sponge(block_cipher, inputs, params)


# Example for checking a collision in Rescue between the inputs (1, 2) and
# (3, 4):
check_collision(
    hash_func=marvellous_hash,
    params=Rescue_S45a,
    input1=vector(Rescue_S45a.field, [1, 2]),
    input2=vector(Rescue_S45a.field, [3, 4]))

# Example for checking a collision in Vision between the inputs (1, 2) and
# (3, 4):
check_collision(
    hash_func=marvellous_hash,
    params=Vision_S45a,
    input1=binary_vector(Vision_S45a.field, [1, 2]),
    input2=binary_vector(Vision_S45a.field, [3, 4]))
