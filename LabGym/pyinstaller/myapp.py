# LabGym/pyinstaller/myapp.py


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import multiprocessing
import re
import sys
import os

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Importing app entry after the start method has been set
    from LabGym.__main__ import main

    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
