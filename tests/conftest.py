import os
import sys
from pathlib import Path

# Ein BLAS-Thread: die Rechnungen sind klein, viele Threads bremsen nur (in Containern mit wenigen Kernen, auf geteilten Rechnern). Die Variablen müssen vor dem ersten Import von NumPy stehen.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
