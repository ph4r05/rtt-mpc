import math
import time
import sys
import logging
from typing import BinaryIO, Optional

logger = logging.getLogger(__name__)


# Input processing strategies:
#  Feeding stdin bytes to the primitive, returning some portion
#  1. Full bytes, `r` field elements per invocation (or its multiple), `c` output field elements.
#      - Field element width defined in bytes. fl0 | fl1,
#      - HW counter, separation to more field elements if using prime field?
#        HW only in one, the rest is zero-padded? yes.
#  2. Bit-level slicing. More complicated, using bitarray library
#
#  params:
#      - # of input blocks
#      - block size in bytes
#
# Problem if more data is hashed - how to split on bits? 1 bit below moduli? Not implemented now, as we use only
# CTR / HW anyway.


def tgcd(x, y):
    while y:
        x, y = y, x % y
    return x


class InputSlicer:
    """
    Slices input stream to bytes of size isize-bits.
    If isize % 8 == 0, direct operation on bytes is used, otherwise bit-level slicing is used with bitarray.
    """
    def __init__(self, isize=256, max_len=None, stream: Optional[BinaryIO] = None, reader=None):
        self.isize = isize  # input block size to produce, in bits
        self.max_len = max_len  # max amount to read in bits
        self.stream = stream
        self.reader = reader

    @property
    def input_bytes(self):
        return (self.isize % 8) == 0

    def process(self):
        isize_b = int(math.ceil(self.isize / 8.))

        read_multiplier = 8 / tgcd(self.isize, 8)
        read_chunk_base = int(self.isize * read_multiplier)
        read_chunk = read_chunk_base * max(1, 65_536 // read_chunk_base)  # expand a bit for optimized read
        cur_len = 0

        b, bout, b_filler, btmp = None, None, None, None
        if not self.input_bytes:
            from bitarray import bitarray
            b = bitarray(endian='big')
            b_filler = bitarray(isize_b * 8 - self.isize, endian='big')
            b_filler.setall(0)

        while True:
            data = self.stream.read(read_chunk) if self.stream else self.reader(read_chunk)
            if not data:
                break

            # Manage output size constrain in bits
            data_bits = len(data) * 8
            last_chunk_sure = False
            if self.max_len is not None and cur_len + data_bits > self.max_len:
                rest = self.max_len - cur_len
                data = data[:rest // 8]
                data_bits = rest
                last_chunk_sure = True

            cur_len += data_bits
            elems = data_bits // self.isize

            if data_bits % self.isize != 0 and not last_chunk_sure:
                logger.warning('Read bits not aligned, %s vs isize %s, mod: %s. May happen for the last chunk.'
                               % (data_bits, self.isize, data_bits % self.isize))

            if self.input_bytes:
                for i in range(elems):
                    yield data[i * (self.isize // 8): (i + 1) * (self.isize // 8)]
                continue

            # bit-level precision branch
            assert not self.input_bytes
            b.clear()
            b.frombytes(data)

            # Parse on ints
            for i in range(elems):
                cbits = b_filler + b[i * self.isize: (i + 1) * self.isize]
                cbytes = cbits.tobytes()
                yield cbytes


class BitCutter:
    """
    Simple bit cutter, slices given segment of data stream out.
    Does not slice continuous bit stream to chunks, use InputSlicer for this purpose.

    Situation we want to solve:
    - Input read from STDIN via InputSlicer is a HW counter generated as 256b. We want to take just 252 bits.
      InputSlicer has isize 256, but we need to slice it down

    Note: we won't be needing it right now as it is better to generate HW counter on log2(prime)//8 bytes
    (full byte so it is leq than moduli size), to avoid blocks colision (should HW 1 occur in bits "above" moduli)
    """
    def __init__(self, isize=256, osize=256):
        self.isize = isize  # input block size, in bits
        self.osize = osize  # output block size, in bits
        raise SystemError('Not implemented yet')


class OutputSequencer:
    """
    Takes array of bytes, sequences them to blocks. Designed for use with hash functions over prime fields.
    Hash function outputs [f0, f1, ..., fn], we now want to map this to byte stream.

    if osize <  fsize, field element is trimmed after conversion to bits.
       if endian=='big', fe is serialized in big-endian (msb lowest), trimming happens from left, so the highest
       bits are trimmed

    if osize == fsize, field element is appended precisely on the whole bit-width
       e.g., if osize == 253, all field elements are concatenated, 8 elements spans 253*8 = 2024b, the smallest
       number of fe that are byte-aligned

    if osize <  fsize, field element is padded with given number of zero bits
       (ideal for post processing, e.g., spreading).
       To ensure consistency with padding, elements are left-padded, i.e., the most significant bits are added
       so the integer value of the fe is not modified in big-endian encoding.

    """
    def __init__(self, fsize=256, osize=256, ostream: Optional[BinaryIO] = None, writer=None, endian='big'):
        self.ostream = ostream  # type: Optional[BinaryIO]
        self.writer = writer
        self.endian = endian
        self.fsize = fsize  # field element size in bits
        self.fsize_b = int(math.ceil(self.fsize / 8.))  # full size in bytes (ceiled)
        self.osize = osize  # bit size of the output per element
        self.osize_b = int(math.ceil(self.osize / 8.))  # full size in bytes (ceiled)

        self.b = None
        self.btmp = None
        self.bout = None
        self.filler = b"" if self.osize_b <= self.fsize_b else bytes([0 for _ in range(self.osize_b - self.fsize_b)])
        self.do_padd = self.osize_b > self.fsize_b

        if not self.whole_bytes:
            from bitarray import bitarray
            self.b = bitarray(endian=self.endian)
            self.bout = bitarray(endian=self.endian)
            self.btmp = bitarray(endian=self.endian)
            self.bfill = bitarray(self.osize - self.fsize, endian=self.endian) if self.osize > self.fsize \
                else bitarray(endian=self.endian)
            self.bfill.setall(0)
            self.do_padd = self.osize > self.fsize

    @property
    def whole_bytes(self):
        return (self.osize % 8) == 0

    def dump(self, output, flush=False):
        for celem in output:
            cb = int(celem).to_bytes(self.fsize_b, byteorder=self.endian)
            if self.whole_bytes:
                if self.do_padd:
                    self.write(self.filler)
                self.write(cb[:self.osize_b])

            else:
                self.btmp.clear()
                self.btmp.frombytes(cb)
                if self.osize == self.fsize:
                    self.bout += self.btmp
                elif self.osize < self.fsize:
                    self.bout += self.btmp[self.osize_b * 8 - self.osize:]
                else:
                    self.bout += self.bfill
                    self.bout += self.btmp

                if (len(self.bout) % 8 == 0 and len(self.bout) >= 2048) or flush:
                    self._flush()

    def flush(self):
        if self.whole_bytes:
            return
        self._flush()

    def _flush(self):
        if self.whole_bytes:
            return

        from bitarray import bitarray
        tout = self.bout.tobytes()
        if tout:
            self.write(tout)

        self.bout = bitarray(endian=self.endian)

    def write(self, chunk):
        if self.ostream is not None:
            self.ostream.write(chunk)

        elif self.writer is not None:
            self.writer(chunk)


def chunks(items, size):
    if size == 0:
        raise ValueError('Invalid value')
    if size == 1:
        for x in items:
            yield [x]
        return

    buff = []
    for x in items:
        buff.append(x)
        if len(buff) >= size:
            yield buff
            buff = []
    if buff:
        yield buff

