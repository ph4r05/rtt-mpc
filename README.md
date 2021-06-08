## MPC-friendly crypto primitives

### Requirements

- `bitarray`

### Sage modules

Preparse, sage -> py:
```bash
sage -preparse starkad_poseidon.sage
```

Move
```bash
mv starkad_poseidon.sage.py starkad_poseidon_sage.py
```

Import in sage
```bash
from starkad_poseidon_sage import *
test()
```

### OSX Installation

```bash
xcode-select --install
export LDFLAGS="-L/usr/local/opt/zlib/lib -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
export CPPFLAGS="-I/usr/local/opt/zlib/include"
export PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"
export SYSTEM_VERSION_COMPAT=1

sysctl -n hw.ncpu
export MAKE='make -j12'

./sage -i openssl
./sage -f python3
./sage -pip install bitarray
```
