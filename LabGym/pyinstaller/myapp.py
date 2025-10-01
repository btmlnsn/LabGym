# LabGym/pyinstaller/myapp.py


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import multiprocessing
import re
import sys
import os

# Multiprocessing start method is spawn before imports; child processes won't inherit main process's state
if sys.platform == "darwin":
    multiprocessing.set_start_method('spawn', force=True)

# free_zupport() on macOS prevents double-launch issues
if sys.platform == "darwin":
    multiprocessing.freeze_support()


# Once the multiprocessing is established, the main function can be imported and executed
from LabGym.__main__ import main

if __name__ == "__main__":
    # argv[0] would be nicer if launched via a script wrapper
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
