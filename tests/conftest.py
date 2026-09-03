'''
Put this checkout first on ``sys.path``.

The test modules append the repository root themselves, but *append* loses to
a copy of ``pybmd`` installed non-editably in site-packages; inserting at the
front here guarantees the suite exercises the source tree it sits in.
'''
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if sys.path[0] != REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)
