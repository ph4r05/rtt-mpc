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


class MiMCParams(object):
    def __init__(self, field, n_rounds, red_rounds=None):
        self.field = field
        self.r = 1
        self.c = 1
        self.m = 2
        self.output_size = 1
        self.red_rounds = red_rounds if red_rounds is not None else n_rounds
        self.round_constants = [field(0)] + \
            [generate_round_constant('MiMC', field, i)
             for i in range(n_rounds - 2)] + [field(0)]


# Parameter sets.
# S45 = MiMCParams(field=F91, n_rounds=116)
# S80 = MiMCParams(field=F161, n_rounds=204)
# S128 = MiMCParams(field=F253, n_rounds=320)


def feistel_permutation(inputs, params):
    x_left, x_right = inputs
    for x_i, c_i in enumerate(params.round_constants):
        if x_i >= params.red_rounds:
            break
        x_left, x_right = x_right + (x_left + c_i) ** 3, x_left

    return vector([x_left, x_right])


def MiMC(inputs, params):
    return sponge(feistel_permutation, inputs, params)


# Example for checking a collision between the inputs (1, 2) and (3, 4):
# check_collision(
#     hash_func=MiMC,
#     params=S45,
#     input1=vector(S45.field, [1, 2]),
#     input2=vector(S45.field, [3, 4]))


"""
Testing permutation: calling feistel_permutation with 2 state works (Feistel; L, R), returns 2 elements
Testing sponge: classical sponge, range(0, len(inputs), params.r), returns params.output_size == capacity blocks from the state
"""


def main_mimc():
    import argparse
    parser = argparse.ArgumentParser(description='MiMC')
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

    logger.info('Initializing...')
    cparams = None
    args = parser.parse_args()
    cc = args.function

    if cc is None or cc == '' or cc == 'S128':
        # Reasonable default for unnamed ciphers
        S128 = MiMCParams(field=F253, n_rounds=320)
        cparams = S128

    elif cc == 'S45':
        S45 = MiMCParams(field=F91, n_rounds=116)
        cparams = S45

    elif cc == 'S80':
        S80 = MiMCParams(field=F161, n_rounds=204)
        cparams = S80

    else:
        raise ValueError('Unknown named cipher %s' % (cc, ))

    if not cparams:
        raise ValueError('Function not specified')

    if args.rounds is not None:
        logger.info("Round reduced to: %s" % (args.rounds,))
        cparams.red_rounds = args.rounds

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
            resb = feistel_permutation(inp_vct, cparams)
        else:
            resb = sponge(feistel_permutation, inp_vct, cparams)

        outb = resb[:args.out_blocks or 1]
        # print(outb)
        outb = defieldizer(outb)
        # print(outb)
        oseq.dump(outb)
    oseq.flush()


if __name__ == '__main__':
    main_mimc()
