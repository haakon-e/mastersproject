# Without conda:
## Preparation
1. Start with a clean ubuntu (latest version) (possibly minimal installation).

Next, assert that `python -V` is not installed, and that `python3 -V` is `3.6.8` or similar.

Next, some basic installs:
```
sudo apt update && yes|sudo apt upgrade && sudo apt install -y python3-pip git wget
sudo apt autoremove
```

Some dependecies of PorePy, not found on pip/pip3:

gmsh for meshing:
```
cd ~
wget http://gmsh.info/bin/Linux/gmsh-4.3.0-Linux64.tgz
sudo tar xvf gmsh-4.3.0-Linux64.tgz
sudo rm gmsh-4.3.0-Linux64.tgz
echo "config={\"gmsh_path\":\"~/gmsh-4.3.0-Linux64/bin/gmsh\"}" > porepy_config.py
```
If you install in a different folder than `~`, adjust the path in `porepy_config.py` accordingly.

polyhedron:
```
cd ~
wget https://raw.githubusercontent.com/keileg/polyhedron/master/polyhedron.py
mv polyhedron.py robust_point_in_polyhedron.py
export PYTHONPATH=~/
```

## Install PorePy
```
clone git clone https://github.com/pmgbergen/porepy.git
cd porepy
pip3 install -r requirements-dev.txt
```

To install PorePy locally on the user (i.e. in a folder you know you will have rwx access), use the --user option.
If you develop PorePy and wish that `import porepy` should always refer to whatever source code is located in `~/porepy/src`, use the `-e` option
```
pip3 install --user -e .
```

Test the installation with:
```
cd ~/porepy
python setup.py test
```
You should get 934 passed with 80% total cover.
If you get errors on matplotlib, try this:
```
sudo apt install python3.6-tk
```

# With conda

Download conda: 
```
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
```
You should verify sha256sum. Run `sha256sum Anaconda3-2019.07-Linux-x86_64.sh` and check against the appropriate file on:
https://docs.anaconda.com/anaconda/install/hashes/all/

For `Anaconda3-2019.07-Linux-x86_64.sh` it should be
```
69581cf739365ec7fb95608eef694ba959d7d33b36eb961953f2b82cb25bdf5a
```

Next, run `bash Anaconda3-2019.07-Linux-x86_64.sh`, read the terms, answer `yes` when prompted, and press enter on default location.
(should be `/home/<user>/anaconda3`)
When prompted to "initialize Anaconda3 by running conda init", type `yes`. Close and re-open your current shell for changes to be effective.
If you type `conda list` you'll see lots of packages pre-installed with Anaconda. Many of them will be useful, like jupyter notebook.

Setup a new environment:
```
conda create -y --name pore python=3.7 jupyter
conda activate pore
```
Note that when inside an environment (prompt prefixed by (base) or (pore)), you can use `python` and `pip` instead of `python3` and `pip3` since conda knows we refer to the version of python and pip installed in the conda environment, not python 2.x which is what the computer thinks we refer to when typing `python`.

## Add porepy and its dependecies to the environment

navigate to `cd ~/porepy`, then run
```
pip install -r requirements-dev.txt
pip install -e .
```

Results from building in conda:
import of porepy fails:
```
(pore) haakon@haakon-VirtualBox:~/porepy$ ipython
Python 3.7.4 (default, Aug 13 2019, 20:35:49) 
Type 'copyright', 'credits' or 'license' for more information
IPython 7.7.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import porepy as pp                                                                      
Segmentation fault (core dumped)
(pore) haakon@haakon-VirtualBox:~/porepy$
```

Also, when running `python setup.py test`:
```
(pore) haakon@haakon-VirtualBox:~/porepy$ python setup.py test
running pytest
running egg_info
writing src/porepy.egg-info/PKG-INFO
writing dependency_links to src/porepy.egg-info/dependency_links.txt
writing requirements to src/porepy.egg-info/requires.txt
writing top-level names to src/porepy.egg-info/top_level.txt
reading manifest file 'src/porepy.egg-info/SOURCES.txt'
reading manifest template 'MANIFEST.in'
warning: no files found matching 'Licence.md'
writing manifest file 'src/porepy.egg-info/SOURCES.txt'
running build_ext
====================================== test session starts ======================================
platform linux -- Python 3.7.4, pytest-4.5.0, py-1.8.0, pluggy-0.12.0
rootdir: /home/haakon/porepy, inifile: setup.cfg
plugins: cov-2.6.1
collecting ... Segmentation fault (core dumped)
(pore) haakon@haakon-VirtualBox:~/porepy$
```

However, (with the setup as above), when i exit the conda environment, I get:
```
(pore) haakon@haakon-VirtualBox:~/porepy$ conda deactivate
(base) haakon@haakon-VirtualBox:~/porepy$ conda deactivate
haakon@haakon-VirtualBox:~/porepy$ python3
Python 3.6.8 (default, Jan 14 2019, 11:02:34) 
[GCC 8.0.1 20180414 (experimental) [trunk revision 259383]] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import porepy as pp
/home/haakon/porepy/src/porepy/grids/partition.py:19: UserWarning: Could not import pymetis. Some functions will not work as    intended
  intended"
>>> 
```
