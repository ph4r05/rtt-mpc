# https://github.com/KULeuven-COSIC/Marvellous/blob/master/rescue_prime.sage
# https://eprint.iacr.org/2020/1143.pdf
# https://www.esat.kuleuven.be/cosic/sites/rescue/

import sys
import math
import logging

load('common.sage')
from base import InputSlicer, OutputSequencer, chunks
from shake256 import SHAKE256

logger = logging.getLogger(__name__)
try:
    import coloredlogs
    coloredlogs.install(level=logging.INFO)
except:
    pass


def get_round_constants(p, m, capacity, security_level, N):
    # generate pseudorandom bytes
    bytes_per_int = ceil(len(bin(p)[2:]) / 8) + 1
    num_bytes = bytes_per_int * 2 * m * N
    seed_string = "Rescue-XLIX(%i,%i,%i,%i)" % (p, m, capacity, security_level)
    byte_string = SHAKE256(bytes(seed_string, "ascii"), num_bytes)

    # process byte string in chunks
    round_constants = []
    Fp = FiniteField(p)
    for i in range(2 * m * N):
        chunk = byte_string[bytes_per_int * i: bytes_per_int * (i + 1)]
        integer = sum(256 ^ j * ZZ(chunk[j]) for j in range(len(chunk)))
        round_constants.append(Fp(integer % p))

    return round_constants


def get_number_of_rounds(p, m, capacity, security_level, alpha):
    # get number of rounds for Groebner basis attack
    rate = m - capacity
    dcon = lambda N: floor(0.5 * (alpha - 1) * m * (N - 1) + 2)
    v = lambda N: m * (N - 1) + rate
    target = 2 ^ security_level
    for l1 in range(1, 25):
        if binomial(v(l1) + dcon(l1), v(l1)) ^ 2 > target:
            break

    # set a minimum value for sanity and add 50%
    return ceil(1.5 * max(5, l1))


def get_alphas(p):
    for alpha in range(3, p):
        if gcd(alpha, p - 1) == 1:
            break
    g, alphainv, garbage = xgcd(alpha, p - 1)
    return (alpha, (alphainv % (p - 1)))


def get_mds_matrix(p, m):
    # get a primitive element
    Fp = FiniteField(p)
    g = Fp(2)
    while g.multiplicative_order() != p - 1:
        g = g + 1

    # get a systematic generator matrix for the code
    V = matrix([[g ^ (i * j) for j in range(0, 2 * m)] for i in range(0, m)])
    V_ech = V.echelon_form()

    # the MDS matrix is the transpose of the right half of this matrix
    MDS = V_ech[:, m:].transpose()
    return MDS


def rescue_prime_parameters(p, m, capacity, security_level):
    alpha, alphainv = get_alphas(p)
    N = get_number_of_rounds(p, m, capacity, security_level, alpha)
    MDS = get_mds_matrix(p, m)
    round_constants = get_round_constants(p, m, capacity, security_level, N)
    return p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants


def rescue_prime_wrapper(parameters, input_sequence):
    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    rate = m - capacity
    Fp = FiniteField(p)

    padded_input = input_sequence + [Fp(1)]
    while len(padded_input) % rate != 0:
        padded_input.append(Fp(0))

    return rescue_prime_hash(parameters, padded_input)


def rescue_prime_hash(parameters, input_sequence):
    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    rate = m - capacity
    Fp = FiniteField(p)

    assert len(input_sequence) % rate == 0

    # initialize state to all zeros
    state = matrix([[Fp(0)] for i in range(m)])

    # absorbing
    absorb_index = 0
    while absorb_index < len(input_sequence):
        for i in range(0, rate):
            state[i, 0] += input_sequence[absorb_index]
            absorb_index += 1
        state = rescue_XLIX_permutation(parameters, state)

    # squeezing
    output_sequence = []
    for i in range(0, rate):
        output_sequence.append(state[i, 0])

    return output_sequence


def rescue_XLIX_permutation(parameters, state):
    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    Fp = state[0, 0].parent()

    for i in range(N):
        # S-box
        for j in range(m):
            state[j, 0] = state[j, 0] ^ alpha
        # mds
        state = MDS * state
        # constants
        for j in range(m):
            state[j, 0] += round_constants[i * 2 * m + j]

        # inverse S-box
        for j in range(m):
            state[j, 0] = state[j, 0] ^ alphainv
        # mds
        state = MDS * state
        # constants
        for j in range(m):
            state[j, 0] += round_constants[i * 2 * m + m + j]

    return state


def get_number_of_rounds1(p, m, capacity, security_level, alpha):
    # get number of rounds for Groebner basis attack
    rate = m - capacity
    dcon = lambda N: floor(0.5 * (alpha - 1) * m * (N - 1) + 2)
    v = lambda N: m * (N - 1) + rate
    target = 2 ^ security_level
    for l1 in range(1, 25):
        if binomial(v(l1) + dcon(l1), v(l1)) ^ 2 > target:
            break

    # get number of rounds for differential attack
    l0 = 2 * security_level / (log(1.0 * p ^ (m + 1), 2.0) - log(1.0 * (alpha - 1) ^ (m + 1), 2.0))

    # take minimum of numbers, sanity factor, and add 50%
    return ceil(1.5 * max(5, l0, l1))


def rescue_prime_DEC(parameters, input_sequence, output_length):
    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    rate = m - capacity
    Fp = FiniteField(p)

    padded_input = input_sequence + [Fp(1)]
    while len(padded_input) % rate != 0:
        padded_input.append(Fp(0))

    return rescue_prime_sponge(parameters, padded_input, output_length)


def rescue_prime_sponge(parameters, input_sequence, output_length):
    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    rate = m - capacity
    Fp = FiniteField(p)

    assert len(input_sequence) % rate == 0

    # initialize state to all zeros
    state = matrix([[Fp(0)] for i in range(m)])

    # absorbing
    absorb_index = 0
    while absorb_index < len(input_sequence):
        for i in range(0, rate):
            state[i, 0] += input_sequence[absorb_index]
            absorb_index += 1
        state = rescue_XLIX_permutation(parameters, state)

    # squeezing
    output_sequence = []
    squeeze_index = 0
    while squeeze_index < output_length:
        for i in range(0, rate):
            output_sequence.append(state[i, 0])
            squeeze_index += 1
        if squeeze_index < output_length:
            state = rescue_XLIX_permutation(parameters, state)

    return output_sequence[:output_length]


class RescuePrimeParams(object):
    def __init__(self, field, m, capacity, security_level, num_rounds=None, reduced_rounds=None):
        """rescue_prime_parameters"""
        self.p = p = field.order()
        self.field = field
        self.m = m
        self.c = capacity
        self.r = m - capacity
        self.security_level = security_level
        self.num_rounds = num_rounds
        self.red_rounds = reduced_rounds
        assert self.r > 0

        self.alpha, self.alphainv = get_alphas(p)
        self.N = get_number_of_rounds(p, m, capacity, security_level, self.alpha)
        self.MDS = get_mds_matrix(p, m)
        self.round_constants = get_round_constants(p, m, capacity, security_level, self.N)
        self.params = p, m, capacity, security_level, self.alpha, self.alphainv, reduced_rounds or self.N, \
                      self.MDS, self.round_constants


def main_rescuep():
    import argparse
    parser = argparse.ArgumentParser(description='Rescue Prime')
    parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                        help='enables debug mode')

    parser.add_argument('--raw', dest='raw', action='store_const', const=True,
                        help='Use raw permutation function, without sponge')
    parser.add_argument('-f', dest='function',
                        help='Named function name to use')

    parser.add_argument('--field', dest='field',
                        help='Field f')
    parser.add_argument('--state', dest='state', type=int,
                        help='State size m')
    parser.add_argument('--cap', dest='capacity', type=int,
                        help='Capacity c')
    parser.add_argument('--sec', dest='sec', type=int,
                        help='Security level')
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
    parser.add_argument('--max-out', dest='max_out', type=int,
                        help='Maximum length in bits for output')

    cparams = None

    logger.info('Initializing...')
    args, unparsed = parser.parse_known_args()
    cc = args.function

    use_red = args.rounds is not None
    is_spec = [
        args.field is not None,
        args.state is not None,
        args.capacity is not None,
    ]
    use_spec = sum(is_spec) == len(is_spec)

    sfield = (args.field or 'F91').upper()
    fields = MPC_FIELDS
    field = (fields[sfield] if sfield in fields else None)
    i_s, i_c, i_sec, i_rr = args.state, args.capacity, args.sec, args.rounds

    if i_s is not None:
        cparams = RescuePrimeParams(field, i_s, i_c, i_sec, reduced_rounds=i_rr)

    elif cc == 'RescueP_S80a' or not cc:
        cparams = RescuePrimeParams(F91, 2, 1, 80, reduced_rounds=i_rr)  # N = 18

    elif cc == 'RescueP_S80b':
        cparams = RescuePrimeParams(F253, 2, 1, 80, reduced_rounds=i_rr)  # N = 18

    elif cc == 'RescueP_S80c' or not cc:
        cparams = RescuePrimeParams(F91, 4, 2, 80, reduced_rounds=i_rr)  # N = 9

    elif cc == 'RescueP_S80d':
        cparams = RescuePrimeParams(F253, 4, 2, 80, reduced_rounds=i_rr)  # N = 9

    elif cc == 'RescueP_128a':
        cparams = RescuePrimeParams(F91, 2, 1, 128, reduced_rounds=i_rr)  # N = 27

    elif cc == 'RescueP_128b':
        cparams = RescuePrimeParams(F253, 2, 1, 128, reduced_rounds=i_rr)  # N = 27

    elif cc == 'RescueP_128c':
        cparams = RescuePrimeParams(F91, 4, 2, 128, reduced_rounds=i_rr)  # N = 14

    elif cc == 'RescueP_128d':
        cparams = RescuePrimeParams(F253, 4, 2, 128, reduced_rounds=i_rr)  # N = 14

    else:
        raise ValueError('Unknown named cipher %s' % (cc, ))

    if not cparams:
        raise ValueError('Function not specified')

    """
    raw: use permutation function hades_permutation with state size r + c (may decrease to 1+1) here.
    full: call sponge with given parameters. 
    """
    max_out = args.max_out
    field_size = get_field_size(cparams.field)
    fieldizer = get_fieldizer(cparams.field)
    defieldizer = get_defieldizer(cparams.field)
    inp_filler = get_input_block_filler(cparams.field, uses_sponge=True, m=cparams.m, r=cparams.r)
    islicer = InputSlicer(stream=sys.stdin.buffer, isize=args.inp_block_size if args.inp_block_size else field_size)
    oseq = OutputSequencer(ostream=sys.stdout.buffer, fsize=field_size,
                           osize=args.out_block_size if args.out_block_size else field_size)

    int_reader = get_int_reader(islicer, args.inp_endian)
    num_inp_blocks = args.inp_block_count if args.inp_block_count else 1
    logger.info('Algorithms initialized, starting computation')

    for blocks in chunks(int_reader(), cparams.r):
        # inp_vct = inp_filler(blocks)
        # inp_vct = fieldizer(inp_vct)
        inp_vct = blocks
        if len(blocks) != cparams.r:
            break

        if args.raw:
            resb = rescue_prime_hash(cparams.params, inp_vct)
        else:
            resb = rescue_prime_wrapper(cparams.params, inp_vct)

        outb = resb[:args.out_blocks or 1]
        # print(outb)
        outb = defieldizer(outb)
        # print(outb)
        oseq.dump(outb)
        oseq.maybe_flush()

        if max_out is not None and oseq.bits_written >= max_out:
            break

    oseq.flush()


if __name__ == '__main__':
    main_rescuep()

