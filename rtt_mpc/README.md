## MPC-friendly crypto primitives

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
