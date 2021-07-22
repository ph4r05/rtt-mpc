
Generate jobs from console

```python
from experiments import generator as g
import json, os
os.chdir('/tmp/ggen')
rr=g.gen_lowmc(eprefix='testmpc02-')
g.write_submit(rr)
```
