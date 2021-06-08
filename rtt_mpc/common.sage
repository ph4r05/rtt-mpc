import math

# https://wordpress-434650-1388715.cloudwaysapps.com/developers-community/hash-challenge/hash-challenge-implementation-reference-code/#marvellous
# Prime fields.
F61 = GF(2**61 + 20 * 2**32 + 1)  # 0x2000001400000001
F81 = GF(2**81 + 80 * 2**64 + 1)  # 0x200500000000000000001
F91 = GF(2**91 + 5 * 2**64 + 1)  # 0x80000050000000000000001
F125 = GF(2**125 + 266 * 2**64 + 1)  # 0x200000000000010a0000000000000001
F161 = GF(2**161 + 23 * 2**128 + 1)  # 0x20000001700000000000000000000000000000001
F253 = GF(2**253 + 2**199 + 1)  # 0x2000000000000080000000000000000000000000000000000000000000000001

# https://neuromancer.sk/std/bn/bn254#
F_PBN254 = GF(0x2523648240000001BA344D80000000086121000000000013A700000000000013)
F_QBN254 = GF(0x2523648240000001BA344D8000000007FF9F800000000010A10000000000000D)

# https://github.com/kendricktan/heiswap-dapp/blob/master/contracts/AltBn128.sol
F_QBN128 = GF(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001)

# https://neuromancer.sk/std/bls/BLS12-381#
F_PBLS12_381 = GF(0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab)
F_QBLS12_381 = GF(0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001)

# https://neuromancer.sk/std/other/Ed25519#
F_PED25519 = GF(0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed)
F_QED25519 = GF(0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed)

# Binary fields.
X = GF(2)['X'].gen()
Bin63 = GF(2**63, name='a', modulus=X**63 + X + 1)
Bin81 = GF(2**81, name='a', modulus=X**81 + X**4 + 1)
Bin91 = GF(2**91, name='a', modulus=X**91 + X**8 + X**5 + X + 1)
Bin127 = GF(2**127, name='a', modulus=X**127 + X + 1)
Bin161 = GF(2**161, name='a', modulus=X**161 + X**18 + 1)
Bin255 = GF(2**255, name='a', modulus=X**255 + X**5 + X**3 + X**2 + 1)

MPC_FIELDS = {
    'F61': F61,
    'F81': F81,
    'F91': F91,
    'F125': F125,
    'F161': F161,
    'F253': F253,
    'F_PBN254': F_PBN254,
    'F_QBN254': F_QBN254,
    'F_QBN128': F_QBN128,
    'F_PBLS12_381': F_PBLS12_381,
    'F_QBLS12_381': F_QBLS12_381,
    'F_PED25519': F_PED25519,
    'F_QED25519': F_QED25519,
    'Bin63': Bin63,
    'Bin81': Bin81,
    'Bin91': Bin91,
    'Bin127': Bin127,
    'Bin161': Bin161,
    'Bin255': Bin255,
}


def check_collision(hash_func, params, input1, input2):
    hash1 = hash_func(input1, params)
    hash2 = hash_func(input2, params)
    if params.field.characteristic() == 2:
        print('hash1:', [bin(h.integer_representation()) for h in hash1])
        print('hash2:', [bin(h.integer_representation()) for h in hash2])
    else:
        print('hash1:', hash1)
        print('hash2:', hash2)

    # Input length must be the same and the two inputs must be different for a
    # valid collision.
    print('Preconditions?', input1 != input2 and len(input1) == len(input2))
    print('Collision?', hash1 == hash2)


def sponge(permutation_func, inputs, params):
    """
    Applies the sponge construction to permutation_func.
    inputs should be a vector of field elements whose size is divisible by
    params.r.
    permutation_func should be a function which gets (state, params) where state
    is a vector of params.m field elements, and returns a vector of params.m
    field elements.
    """
    assert parent(inputs) == VectorSpace(params.field, len(inputs)), \
        'inputs must be a vector of field elements. Found: %r' % parent(inputs)

    assert len(inputs) % params.r == 0, \
        'Number of field elements must be divisible by %s. Found: %s' % (
            params.r, len(inputs))

    state = vector([params.field(0)] * params.m)

    for i in range(0, len(inputs), params.r):
        state[:params.r] += inputs[i:i+params.r]
        state = permutation_func(state, params)

    # We do not support more than r output elements, since this requires
    # additional invocations of permutation_func.
    assert params.output_size <= params.r
    return state[:params.output_size]


def generate_round_constant(fn_name, field, idx):
    """
    Returns a field element based on the result of sha256.
    The input to sha256 is the concatenation of the name of the hash function
    and an index.
    For example, the first element for MiMC will be computed using the value
    of sha256('MiMC0').
    """
    from hashlib import sha256
    val = int(sha256(('%s%d' % (fn_name, idx)).encode('utf8')).hexdigest(), 16)
    if field.is_prime_field():
        return field(val)
    else:
        return int2field(field, val % field.order())


def int2field_alt(field, val):
    """
    Converts val to an element of a binary field according to the binary
    representation of val.
    For example, 11=0b1011 is converted to 1*a^3 + 0*a^2 + 1*a + 1.
    """
    assert field.characteristic() == 2
    # assert 0 <= val < field.order(), \
    #     'Value %d out of range. Expected 0 <= val < %d.' % (val, field.order())

    # res = field(map(int, bin(val)[2:][::-1]))
    # res = field([x for x in map(int, bin(val)[2:][::-1])])
    res = []
    c = 1
    while c <= val:
        res.append(int((val & c) == c))
        c *= 2
    res = field(res)
    assert res.integer_representation() == val
    return res


def int2field(field, val):
    """
    Converts val to an element of a binary field according to the binary
    representation of val.
    For example, 11=0b1011 is converted to 1*a^3 + 0*a^2 + 1*a + 1.
    """
    assert field.characteristic() == 2
    assert 0 <= val < field.order(), \
        'Value %d out of range. Expected 0 <= val < %d.' % (val, field.order())

    # res = field(map(int, bin(val)[2:][::-1]))
    res = field(list(map(int, bin(val)[2:][::-1])))

    assert res.integer_representation() == val
    return res


def binary_vector(field, values):
    """
    Converts a list of integers to field elements using int2field.
    """
    return vector(field, [int2field(field, val) for val in values])


def binary_matrix(field, values):
    """
    Converts a list of lists of integers to field elements using int2field.
    """
    return matrix(field, [[int2field(field, val) for val in row]
                          for row in values])


def generate_mds_matrix(name, field, m):
    """
    Generates an MDS matrix of size m x m over the given field, with no
    eigenvalues in the field.
    Given two disjoint sets of size m: {x_1, ..., x_m}, {y_1, ..., y_m} we set
    A_{ij} = 1 / (x_i - y_j).
    """
    for attempt in range(100):
        x_values = [generate_round_constant(name + 'x', field, attempt * m + i)
                    for i in range(m)]
        y_values = [generate_round_constant(name + 'y', field, attempt * m + i)
                    for i in range(m)]
        # Make sure the values are distinct.
        assert len(set(x_values + y_values)) == 2 * m, \
            'The values of x_values and y_values are not distinct'
        mds = matrix([[1 / (x_values[i] - y_values[j]) for j in range(m)]
                      for i in range(m)])
        # Sanity check: check the determinant of the matrix.
        x_prod = product(
            [x_values[i] - x_values[j] for i in range(m) for j in range(i)])
        y_prod = product(
            [y_values[i] - y_values[j] for i in range(m) for j in range(i)])
        xy_prod = product(
            [x_values[i] - y_values[j] for i in range(m) for j in range(m)])
        expected_det = (1 if m % 4 < 2 else -1) * x_prod * y_prod / xy_prod
        det = mds.determinant()
        assert det != 0
        assert det == expected_det, \
            'Expected determinant %s. Found %s' % (expected_det, det)

        if len(mds.characteristic_polynomial().roots()) == 0:
            # There are no eigenvalues in the field.
            return mds
    raise Exception('No good MDS found')


def get_field_size(field):
    """Returns log2-size of the field so all elements can be represented in the given bitsize"""
    return int(math.ceil(math.log(int(field.cardinality()), 2)))


def is_binary_field(field):
    """Returns true if input field is binary, i.e., has characteristics 2"""
    return field.characteristic() == 2


def get_fieldizer(field):
    """Returns a function converting integer vectors to a field vectors"""
    return (lambda x: binary_vector(field, x)) if is_binary_field(field) \
        else (lambda x: vector(field, x))


def get_defieldizer(field):
    """Returns a function converting field elements to integers (input is a vector)"""
    return (lambda x: [int(z.integer_representation()) for z in x]) if is_binary_field(field) \
        else (lambda x: [int(z) for z in x])


def get_input_block_filler_field(field, uses_sponge=False, m=None, r=None):
    """Returns function padding input block to a block of required size. Returns a field vector.
        e.g., sponge requires |input| % r == 0, else is padded to |input| == m"""
    if uses_sponge:
        # |inp_vct| % params.r == 0
        return lambda inp_vct: inp_vct if len(inp_vct) % r == 0 else vector(list(inp_vct) + ([field(0)] * (r - len(inp_vct) % r)))

    else:
        # |inp_vct| == params.m
        return lambda inp_vct: inp_vct if len(inp_vct) == m else vector(list(inp_vct) + [field(0)] * (m - len(inp_vct)))


def get_input_block_filler(field, uses_sponge=False, m=None, r=None):
    """Returns function padding input block to a block of required size
    e.g., sponge requires |input| % r == 0, else is padded to |input| == m"""
    if uses_sponge:
        # |inp_vct| % params.r == 0
        return lambda inp_vct: inp_vct if len(inp_vct) % r == 0 else inp_vct + [0] * (r - len(inp_vct) % r)

    else:
        # |inp_vct| == params.m
        return lambda inp_vct: inp_vct if len(inp_vct) == m else inp_vct + [0] * (m - len(inp_vct))


def get_int_reader(islicer, endian='big'):
    """Returns a function reading bytes from input slicer, converting them to integers and yielding out"""
    def int_reader():
        for chunk in islicer.process():
            yield int.from_bytes(bytes(chunk), byteorder=endian)
    return int_reader
