'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext.

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''

import nox
import sys

nox.options.error_on_missing_interpreters=True


@nox.session(python=['3.9','3.10'],reuse_venv=True)
def tests(session:nox.Session):
    # prefer wheels
    session.env["PIP_PREFER_BINARY"]="1"
    session.install("-U", "pip", "setuptools", "wheel")

    # Linux: preintall a wxPython wheel to avoid building from source
    if sys.platform.startswith("linux"):
        session.run(
            "python",
            "-m",
            "pip",
            "install",
            "-f",
            "http://wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04",
            'wxPython==4.2.1',
        )

    # package an Python dependencies installed, and test dependencies
    session.install("-e", ".")
    session.install("pytest")
    session.run("pytest", "-q")


@nox.session(reuse_venv=True)
def docs(session:nox.Session):
    session.install("-U", "pip", "setuptools", "wheel")
    session.install('-r','docs/requirements.txt')
    session.run('make','-C','docs','clean',external=True)
    session.run('sphinx-autobuild','docs','docs/_build/html')
