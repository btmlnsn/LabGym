# LabGym/pyinstaller/myapp.py


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sys
from LabGym.__main__ import main

if __name__ == "__main__":
    # make argv[0] nicer if launched via a script wrapper
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
