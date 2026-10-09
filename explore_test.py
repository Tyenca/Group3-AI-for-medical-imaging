import sys
from pathlib import Path
import numpy as np
import nibabel as nib

files = sorted(Path(sys.argv[1]).rglob("*.nii.gz"))
print(len(files), "files")
for f in files:
    img = nib.load(f)
    sp = tuple(round(float(s), 3) for s in img.header.get_zooms()[:3])
    data = np.asanyarray(img.dataobj)
    print(f.name, img.shape, sp, "HU", int(data.min()), int(data.max()), data.dtype)