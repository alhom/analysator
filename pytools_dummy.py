# 
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2018 University of Helsinki
# 
# For details of usage, see the COPYING file and read the "Rules of the Road"
# at http://www.physics.helsinki.fi/vlasiator/
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# 

from os import path as __path
import warnings
import sys
import logging




# Slurp and exec the analysator.py file here to get all functionalities under the pytools alias if needed
if 'analysator' not in sys.modules.keys():
   warnings.warn("Please update your import command to `import analysator as pt`. `pytools.py` has been renamed as `analysator.py` for consistency and eventual package publication. `import pytools` and `import pytools as pt` will work via the dirty hack here in `pytools.py` until some time in the future (v.1 release/package publication?).")
   logging.info("Importing analysator.py to pytools")
   root = __path.dirname(__file__)
   with open(__path.join(root,'_analysator.py'),'r') as f:
      source = f.read()
      exec(source)
      f.close()
      del f
else:
   logging.info("Analysator imported already. Pointing pytools to analysator.")
   sys.modules['pytools'] = sys.modules['analysator']