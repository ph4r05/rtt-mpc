"""
!! NOT FINISHED !!
Works over fields, in the Sage
"""

import math
import time
import sys
import logging
from typing import BinaryIO, Optional
from .base import InputSlicer
logger = logging.getLogger(__name__)


class FieldProcessor:
    """
    Takes a hash function over fields, applies it to the input stream, returns output.

    Input:
     - For slicing input stream to chunks, InputSlicer is used.
     - FieldProcessor passes given number of elements from the InputSlicer (e.g., for Posseidon c + r = 5 elements)
       to the hash function for processing. In this current version we will be using just one invocation.
     - Challenge: setup input in such a way the field application does not cause input unwanted duplication
       e.g., if HW ctr is bigger than moduli, blocks geq to moduli will be reduced and probably mapped to already
       existing input block, causing duplication in output. This yields to unwanted redundancy that can be caught
       by randomness tests, causing false positive.

    Output:
     - Given number of output blocks is passed to the OutputSequencer to dump it to the stdout.
     - Challenge: with prime fields, bit space "above" the prime is not used. A reasonable strategy has to be used
       to overcome this issue, otherwise it is caught by randomness tests, as they work over the whole bits.

       Naively, only lower x bits can be taken from the output. It has to be checked whether the distribution in
       such strategy is uniform.

       Spreader uses more sophisticated strategies to map prime range to an uniform distribution over the whole bit-set.

    !! NOT FINISHED !!
    """
    def __init__(self):
        self.is_binary = False
        self.bin_size = None
        self.mod = None

        self.input_bytes = True  # bytes vs bits level

        self.isize = None  # input block size
        self.input_blocks = None  # number of blocks per function invocation

        self.osize = None
        self.max_len = None
        self.max_out = None

    def process(self):
        mod_size = int(math.ceil(math.log2(self.mod))) if self.mod else None

        if not self.isize and self.mod:
            self.isize = mod_size
        if not mod_size and self.isize:
            mod_size = self.isize
        if not self.osize and self.isize:
            self.osize = self.isize
        if not self.osize and mod_size:
            self.osize = mod_size

        osize_b = int(math.ceil(self.osize / 8.))
        isize_b = int(math.ceil(self.isize / 8.))

        read_multiplier = 8 / tgcd(self.isize, 8)
        read_chunk_base = int(self.isize * read_multiplier)
        read_chunk = read_chunk_base * max(1, 65_536 // read_chunk_base)  # expand a bit
        max_len = self.max_len
        max_out = self.max_out
        cur_len = 0
        cur_out = 0

        if not self.input_bytes:
            from bitarray import bitarray
            b = bitarray(endian='big')
            bout = bitarray(endian='big')
            b_filler = bitarray(isize_b * 8 - self.isize, endian='big')
            b_filler.setall(0)
            btmp = bitarray(endian='big')

        osize_mask = (2 ** (osize_b * 8)) - 1
        nrejects = 0
        noverflows = 0
        time_start = time.time()
        logger.info("Generating data")

        while True:
            data = sys.stdin.buffer.read(read_chunk)

            if not data:
                break

            # Manage output size constrain in bits
            cblen = len(data) * 8
            last_chunk_sure = False
            if max_len is not None and cur_len + cblen > max_len:
                rest = max_len - cur_len
                data = data[:rest // 8]
                cblen = rest
                last_chunk_sure = True

            cur_len += cblen
            elems = cblen // self.isize

            if cblen % self.isize != 0 and not last_chunk_sure:
                logger.warning('Read bits not aligned, %s vs isize %s, mod: %s. May happen for the last chunk.'
                               % (cblen, self.isize, cblen % self.isize))

            b.clear()
            b.frombytes(data)

            # Parse on ints
            for i in range(elems):
                cbits = b_filler + b[i * isize: (i + 1) * isize]

                cbytes = cbits.tobytes()
                celem = int.from_bytes(bytes=cbytes, byteorder='big')
                spreaded = spread_func(celem)
                if spreaded is None:
                    nrejects += 1
                    continue
                if spreaded > osize_mask:
                    noverflows += 1

                oelem = int(spreaded) & osize_mask
                oelem_b = oelem.to_bytes(osize_b, 'big')
                btmp.clear()
                btmp.frombytes(oelem_b)
                bout += btmp[osize_b * 8 - osize:]
                cur_out += osize
                if max_out is not None and cur_out >= max_out:
                    break

            finishing = data is None \
                        or (max_len is not None and max_len <= cur_len) \
                        or (max_out is not None and max_out <= cur_out)
            if (len(bout) % 8 == 0 and len(bout) >= 2048) or finishing:
                tout = bout.tobytes()
                if self.args.ohex:
                    tout = binascii.hexlify(tout)

                output_fh.write(tout)
                bout = bitarray(endian='big')

            if finishing:
                output_fh.flush()
                break
        time_elapsed = time.time() - time_start
        logger.info("Number of rejects: %s, overflows: %s, time: %s s" % (nrejects, noverflows, time_elapsed,))
        if self.args.ofile:
            output_fh.close()
