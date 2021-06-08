import sys
import math
import logging

load('common.sage')
from base import InputSlicer, OutputSequencer, chunks


logger = logging.getLogger(__name__)
try:
    import coloredlogs
    coloredlogs.install(level=logging.INFO)
except:
    pass


class MarvellousParams(object):
    def __init__(self, field, r, c, num_rounds, reduced_rounds=None):
        self.field = field
        self.r = r
        self.c = c
        self.m = m = r + c
        self.output_size = c
        assert self.output_size <= r
        self.num_rounds = num_rounds
        self.red_rounds = reduced_rounds if reduced_rounds is not None else num_rounds
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
# Rescue_S45a = MarvellousParams(field=F91, r=2, c=1, num_rounds=10)
# Vision_S45a = MarvellousParams(field=Bin91, r=2, c=1, num_rounds=10)
# Rescue_S45b = MarvellousParams(field=F91, r=10, c=1, num_rounds=10)
# Vision_S45b = MarvellousParams(field=Bin91, r=10, c=1, num_rounds=10)
#
# Rescue_S80a = MarvellousParams(field=F81, r=2, c=2, num_rounds=10)
# Vision_S80a = MarvellousParams(field=Bin81, r=2, c=2, num_rounds=10)
# Rescue_S80b = MarvellousParams(field=F161, r=2, c=1, num_rounds=14)
# Vision_S80b = MarvellousParams(field=Bin161, r=2, c=1, num_rounds=10)
# Rescue_S80c = MarvellousParams(field=F161, r=10, c=1, num_rounds=10)
# Vision_S80c = MarvellousParams(field=Bin161, r=10, c=1, num_rounds=10)
#
# Rescue_S128a = MarvellousParams(field=F125, r=2, c=2, num_rounds=16)
# Vision_S128a = MarvellousParams(field=Bin127, r=2, c=2, num_rounds=12)
# Rescue_S128b = MarvellousParams(field=F253, r=2, c=1, num_rounds=22)
# Vision_S128b = MarvellousParams(field=Bin255, r=2, c=1, num_rounds=16)
# Rescue_S128c = MarvellousParams(field=F125, r=10, c=2, num_rounds=10)
# Vision_S128c = MarvellousParams(field=Bin127, r=10, c=2, num_rounds=10)
# Rescue_S128d = MarvellousParams(field=F61, r=8, c=4, num_rounds=10)
# Vision_S128d = MarvellousParams(field=Bin63, r=8, c=4, num_rounds=10)
# Rescue_S128e = MarvellousParams(field=F253, r=10, c=1, num_rounds=10)
# Vision_S128e = MarvellousParams(field=Bin255, r=10, c=1, num_rounds=10)
#
# Rescue_S256a = MarvellousParams(field=F125, r=4, c=4, num_rounds=16)
# Vision_S256a = MarvellousParams(field=Bin127, r=4, c=4, num_rounds=12)
# Rescue_S256b = MarvellousParams(field=F125, r=10, c=4, num_rounds=10)
# Vision_S256b = MarvellousParams(field=Bin127, r=10, c=4, num_rounds=10)


# Evaluate the block cipher.
def block_cipher(state, params):
    """
    Evaluates the block cipher with key=0 in forward direction.
    """
    state += params.round_constants[0]
    for r in range(2 * params.red_rounds):
        sbox = params.sbox0 if r % 2 == 0 else params.sbox1
        for i in range(params.m):
            state[i] = sbox(state[i])

        state = params.MDS * state + params.round_constants[r + 1]

    return state


def marvellous_hash(inputs, params):
    return sponge(block_cipher, inputs, params)


# Example for checking a collision in Rescue between the inputs (1, 2) and
# (3, 4):
# check_collision(
#     hash_func=marvellous_hash,
#     params=Rescue_S45a,
#     input1=vector(Rescue_S45a.field, [1, 2]),
#     input2=vector(Rescue_S45a.field, [3, 4]))
#
# # Example for checking a collision in Vision between the inputs (1, 2) and
# # (3, 4):
# check_collision(
#     hash_func=marvellous_hash,
#     params=Vision_S45a,
#     input1=binary_vector(Vision_S45a.field, [1, 2]),
#     input2=binary_vector(Vision_S45a.field, [3, 4]))
#

# TODO: test sponge AND block cipher separately.
# for a permutation only, c=0 is OK, r=1, test only one state value.


def main_vision():
    import argparse
    parser = argparse.ArgumentParser(description='Vision / Rescue')
    parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                        help='enables debug mode')

    parser.add_argument('--raw', dest='raw', action='store_const', const=True,
                        help='Use raw permutation function, without sponge')
    parser.add_argument('-f', dest='function',
                        help='Named function name to use')

    parser.add_argument('--field', dest='field',
                        help='Field f')
    parser.add_argument('--rate', dest='rate', type=int,
                        help='Rate r')
    parser.add_argument('--cap', dest='capacity', type=int,
                        help='Capacity c')
    parser.add_argument('--rf', dest='rounds_full', type=int,
                        help='Rf')
    parser.add_argument('-r', dest='rounds', type=int,
                        help='Rounds')

    parser.add_argument('--inp-endian', dest='inp_endian', default='big',
                        help='Input block size endian')
    parser.add_argument('--inp-block-size', dest='inp_block_size', type=int,
                        help='Input block size')
    parser.add_argument('--inp-block-count', dest='inp_block_count', type=int,
                        help='Block count per one hash operation')
    parser.add_argument('--out-block-size', dest='out_block_size', type=int,
                        help='Size of the output block, enables to trim the output, concat field elements precisely '
                             'or left-pad the field element (e.g., to full bytes)')
    parser.add_argument('--out-blocks', dest='out_blocks', type=int,
                        help='Number of output blocks to process')

    cparams = None

    logger.info('Initializing...')
    args = parser.parse_args()
    cc = args.function

    use_red = args.rounds is not None
    is_spec = [
        args.field is not None,
        args.rate is not None,
        args.capacity is not None,
        args.rounds is not None,
    ]
    use_spec = sum(is_spec) == len(is_spec)

    if cc is not None and sum(is_spec) > 0:
        raise ValueError('Cannot define named function and params specs, conflict')
    if not args.raw and sum(is_spec) > 0 and not use_spec:
        raise ValueError('When using specs, specify all parameters')

    sfield = (args.field or 'F_QBLS12_381').upper()
    fields = MPC_FIELDS
    field = (fields[sfield] if sfield in fields else None)
    i_r, i_c, i_rf, i_r = args.rate, args.capacity, args.rounds_full, args.rounds

    if cc is None or cc == '':
        # Reasonable default for unnamed ciphers
        cparams = MarvellousParams(field=field,
                                   r=i_r,
                                   c=i_c,
                                   num_rounds=i_rf,
                                   reduced_rounds=i_r)

    elif cc == 'Rescue_S128e':
        cparams = MarvellousParams(field=F253, r=i_r or 10, c=i_c or 1, num_rounds=i_rf or 10, reduced_rounds=i_r)

    elif cc == 'Rescue_S45a':
        cparams = MarvellousParams(field=F91, r=2, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S45a':
        cparams = MarvellousParams(field=Bin91, r=2, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S45b':
        cparams = MarvellousParams(field=F91, r=10, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S45b':
        cparams = MarvellousParams(field=Bin91, r=10, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S80a':
        cparams = MarvellousParams(field=F81, r=2, c=2, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S80a':
        cparams = MarvellousParams(field=Bin81, r=2, c=2, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S80b':
        cparams = MarvellousParams(field=F161, r=2, c=1, num_rounds=14, reduced_rounds=i_r)

    elif cc == 'Vision_S80b':
        cparams = MarvellousParams(field=Bin161, r=2, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S80c':
        cparams = MarvellousParams(field=F161, r=10, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S80c':
        cparams = MarvellousParams(field=Bin161, r=10, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S128a':
        cparams = MarvellousParams(field=F125, r=2, c=2, num_rounds=16, reduced_rounds=i_r)

    elif cc == 'Vision_S128a':
        cparams = MarvellousParams(field=Bin127, r=2, c=2, num_rounds=12, reduced_rounds=i_r)

    elif cc == 'Rescue_S128b':
        cparams = MarvellousParams(field=F253, r=2, c=1, num_rounds=22, reduced_rounds=i_r)

    elif cc == 'Vision_S128b':
        cparams = MarvellousParams(field=Bin255, r=2, c=1, num_rounds=16, reduced_rounds=i_r)

    elif cc == 'Rescue_S128c':
        cparams = MarvellousParams(field=F125, r=10, c=2, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S128c':
        cparams = MarvellousParams(field=Bin127, r=10, c=2, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S128d':
        cparams = MarvellousParams(field=F61, r=8, c=4, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S128d':
        cparams = MarvellousParams(field=Bin63, r=8, c=4, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S128e':
        cparams = MarvellousParams(field=F253, r=10, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S128e':
        cparams = MarvellousParams(field=Bin255, r=10, c=1, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Rescue_S256a':
        cparams = MarvellousParams(field=F125, r=4, c=4, num_rounds=16, reduced_rounds=i_r)

    elif cc == 'Vision_S256a':
        cparams = MarvellousParams(field=Bin127, r=4, c=4, num_rounds=12, reduced_rounds=i_r)

    elif cc == 'Rescue_S256b':
        cparams = MarvellousParams(field=F125, r=10, c=4, num_rounds=10, reduced_rounds=i_r)

    elif cc == 'Vision_S256b':
        cparams = MarvellousParams(field=Bin127, r=10, c=4, num_rounds=10, reduced_rounds=i_r)

    else:
        raise ValueError('Unknown named cipher %s' % (cc, ))

    if not cparams:
        raise ValueError('Function not specified')

    """
    raw: use permutation function hades_permutation with state size r + c (may decrease to 1+1) here.
    full: call sponge with given parameters. 
    """
    field_size = get_field_size(cparams.field)
    fieldizer = get_fieldizer(cparams.field)
    defieldizer = get_defieldizer(cparams.field)
    inp_filler = get_input_block_filler(cparams.field, uses_sponge=not args.raw, m=cparams.m, r=cparams.r)
    islicer = InputSlicer(stream=sys.stdin.buffer, isize=args.inp_block_size if args.inp_block_size else field_size)
    oseq = OutputSequencer(ostream=sys.stdout.buffer, fsize=field_size,
                           osize=args.out_block_size if args.out_block_size else field_size)

    int_reader = get_int_reader(islicer, args.inp_endian)
    num_inp_blocks = args.inp_block_count if args.inp_block_count else 1
    logger.info('Algorithms initialized, starting computation')

    for blocks in chunks(int_reader(), num_inp_blocks):
        inp_vct = inp_filler(blocks)
        inp_vct = fieldizer(inp_vct)
        # print(inp_vct)

        if args.raw:
            resb = block_cipher(inp_vct, cparams)
        else:
            resb = sponge(block_cipher, inp_vct, cparams)

        outb = resb[:args.out_blocks or 1]
        # print(outb)
        outb = defieldizer(outb)
        # print(outb)
        oseq.dump(outb)
    oseq.flush()


if __name__ == '__main__':
    main_vision()

