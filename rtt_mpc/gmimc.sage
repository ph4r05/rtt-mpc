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


class GMiMCParams(object):
    def __init__(self, field, r, c, num_rounds, red_rounds=None):
        self.field = field
        self.r = r
        self.c = c
        self.m = r + c
        self.red_rounds = red_rounds if red_rounds is not None else num_rounds
        self.round_constants = [generate_round_constant('GMiMC_erf', field, i)
                                for i in range(num_rounds)]

        self.output_size = c
        assert self.output_size <= r


# Parameter sets.
# S45a = GMiMCParams(field=F91, r=2, c=1, num_rounds=121)
# S45b = GMiMCParams(field=F91, r=10, c=1, num_rounds=137)
#
# S80a = GMiMCParams(field=F81, r=2, c=2, num_rounds=111)
# S80b = GMiMCParams(field=F161, r=2, c=1, num_rounds=210)
# S80c = GMiMCParams(field=F161, r=10, c=1, num_rounds=226)
#
# S128a = GMiMCParams(field=F125, r=2, c=2, num_rounds=166)
# S128b = GMiMCParams(field=F253, r=2, c=1, num_rounds=326)
# S128c = GMiMCParams(field=F125, r=10, c=2, num_rounds=182)
# S128d = GMiMCParams(field=F61, r=8, c=4, num_rounds=101)
# S128e = GMiMCParams(field=F253, r=10, c=1, num_rounds=342)
#
# S256a = GMiMCParams(field=F125, r=4, c=4, num_rounds=174)
# S256b = GMiMCParams(field=F125, r=10, c=4, num_rounds=186)


def erf_feistel_permutation(x, params):
    """
    https://eprint.iacr.org/2019/951.pdf
    GMiMC-erf = R^i_k(x_1, ..., x_t) = x2 XOR (x1 XOR c_i XOR k)**3, ..., xt XOR (x1 XOR c_i XOR k)**3, x1
     - k = 0 in this setup
     - when |x| == 2, it is a classical Feistel, R^i(x1, x2) = x2 XOR (x1 XOR c_i)**3, x1
    """
    for x_i, c_i in enumerate(params.round_constants):
        if x_i >= params.red_rounds:
            break
        mask = (x[0] + c_i) ** 3
        x = [x_j + mask for x_j in x[1:]] + [x[0]]
    return vector(params.field, x)


def GMiMC_erf(inputs, params):
    """Sponge calls: state = permutation_func(state, params)"""
    return sponge(erf_feistel_permutation, inputs, params)


# Example for checking a collision between the inputs [1, 2] and [3, 4]:
# check_collision(
#     hash_func=GMiMC_erf,
#     params=S45a,
#     input1=vector(S45a.field, [1, 2]),
#     input2=vector(S45a.field, [3, 4]))

"""
Testing permutation: call erf_feistel_permutation with at least 2 elements (same as Feistel)
Testing sponge: classical sponge, range(0, len(inputs), params.r), returns params.output_size blocks from the state
  - having full parameters as specified by paper yields wider state and better, sponge mixes the state |x|/r times.
  - assert |x| % r == 0
  - if only x0 is non-zero and |x| == r (minimal size), only one invocation of the permutation function will be made.
    Level of mixing depends on the permutation function, e.g., feistel should have ideally at least 2 rounds to avoid 
    naive non-randomness detections
"""


def main_gmimc():
    import argparse
    parser = argparse.ArgumentParser(description='GMiMC')
    parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                        help='enables debug mode')

    parser.add_argument('--raw', dest='raw', action='store_const', const=True,
                        help='Use raw permutation function, without sponge')
    parser.add_argument('-f', dest='function',
                        help='Named function name to use')
    parser.add_argument('-r', dest='rounds', type=int,
                        help='rounds')

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

    logger.info('Initializing...')
    cparams = None
    args = parser.parse_args()
    cc = args.function
    i_r, i_c, i_rr = args.rate, args.capacity, args.rounds

    if cc is None or cc == '' or cc == 'S128b':
        # Reasonable default for unnamed ciphers
        S128a = GMiMCParams(field=F125, r=i_r or 2, c=i_c or 2, num_rounds=166, red_rounds=i_rr)
        cparams = S128a

    elif cc == 'S45a':
        S45a = GMiMCParams(field=F91, r=i_r or 2, c=i_c or 1, num_rounds=121, red_rounds=i_rr)
        cparams = S45a

    elif cc == 'S45b':
        S45b = GMiMCParams(field=F91, r=i_r or 10, c=i_c or 1, num_rounds=137, red_rounds=i_rr)
        cparams = S45b

    elif cc == 'S80a':
        S80a = GMiMCParams(field=F81, r=i_r or 2, c=i_c or 2, num_rounds=111, red_rounds=i_rr)
        cparams = S80a

    elif cc == 'S80b':
        S80b = GMiMCParams(field=F161, r=i_r or 2, c=i_c or 1, num_rounds=210, red_rounds=i_rr)
        cparams = S80b

    elif cc == 'S80c':
        S80c = GMiMCParams(field=F161, r=i_r or 10, c=i_c or 1, num_rounds=226, red_rounds=i_rr)
        cparams = S80c

    elif cc == 'S128a':
        S128a = GMiMCParams(field=F125, r=i_r or 2, c=i_c or 2, num_rounds=166, red_rounds=i_rr)
        cparams = S128a

    elif cc == 'S128c':
        S128c = GMiMCParams(field=F125, r=i_r or 10, c=i_c or 2, num_rounds=182, red_rounds=i_rr)
        cparams = S128c

    elif cc == 'S128d':
        S128d = GMiMCParams(field=F61, r=i_r or 8, c=i_c or 4, num_rounds=101, red_rounds=i_rr)
        cparams = S128d

    elif cc == 'S128e':
        S128e = GMiMCParams(field=F253, r=i_r or 10, c=i_c or 1, num_rounds=342, red_rounds=i_rr)
        cparams = S128e

    elif cc == 'S256a':
        S256a = GMiMCParams(field=F125, r=i_r or 4, c=i_c or 4, num_rounds=174, red_rounds=i_rr)
        cparams = S256a

    elif cc == 'S256b':
        S256b = GMiMCParams(field=F125, r=i_r or 10, c=i_c or 4, num_rounds=186, red_rounds=i_rr)
        cparams = S256b

    else:
        raise ValueError('Unknown named cipher %s' % (cc, ))

    if not cparams:
        raise ValueError('Function not specified')

    if args.rounds is not None:
        logger.info("Round reduced to: %s" % (args.rounds,))
        cparams.red_rounds = args.rounds

    max_out = args.max_out
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
            resb = erf_feistel_permutation(inp_vct, cparams)
        else:
            resb = sponge(erf_feistel_permutation, inp_vct, cparams)

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
    main_gmimc()

