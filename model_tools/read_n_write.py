import os
import re
import hdnntools as hdn

import scipy
from scipy import stats
import statistics

import numpy as np
import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from pandas.plotting import table


def read_xyz (file):           #XYZ file reader for RXN
    import numpy as np
    xyz = []
    typ = []
    Na  = []
    ct = []
    fd = open(file, 'r').read()
    #rb = re.compile('(\d+?)\s*(.*?)\n((?:[A-Z][a-z]?.+?(?:\n|$))+)')
    rb = re.compile('(\d*)\n(.*)\n(((?:[A-Z][a-z]?.+?(?:\n|$))+))')
    ra = re.compile('([A-Z][a-z]?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s*?(?:\n|$)')
    s = rb.findall(fd)
    Nc = len(s)
    if Nc == 0:
        raise ValueError('No coordinates found in file. Check formatting of '+file+'.')
    for i in s:
        X=[]
        T=[]
        ct.append(i[1])
        c = ra.findall(i[2])
        Na.append(len(c))
        for j in c:
            T.append(j[0])
            X.append(j[1])
            X.append(j[2])
            X.append(j[3])
        X=np.array(X, dtype=np.float32)
        X=X.reshape(len(T),3)
        xyz.append(X)
        typ.append(T)
    return xyz,typ,Na,ct



def write_xyz(fn, X, S, cmt='', aw='w'):
    f = open(fn, aw)
    for i in range(len(X)):
        N = len(S[i])
        f.write(str(N)+'\n' + cmt + '\n')
        for j in range(N):
            x=X[i][j][0]
            y=X[i][j][1]
            z=X[i][j][2]
            f.write(S[i][j] + ' ' + "{:.7f}".format(x) + ' ' + "{:.7f}".format(y) + ' ' + "{:.7f}".format(z) + '\n')
    f.close()






def read_fh(f):
    string = open(f, 'r').read()
    regex=".*\n\d*\n([\S\s]*)"
    coor=re.findall(regex, string)
    return coor





def get_gau_E(f, method="HF"):
    """
    get energy from gaussian log file (in hartrees)
    """
    me=''
    for i in method:
        me+=i
        me+="\s*"
    string = open(f, 'r').read()
    regex=re.compile("\W\s*"+me+"=([\S\s]{1,15})\s*\\\\")
    Estr=regex.findall(string)[-1]
#    print(Estr)
    Espl=Estr.split()
    E=''
    for i in Espl:
        E+=i
    return float(E)


def get_Orca_E(f):
    """
    get energy from ORCA output file (in hartrees)
    """
    string = open(f, 'r').read()
    regex=re.compile("FINAL SINGLE POINT ENERGY\s*([-]?\d*\.\d*)")
    E=regex.findall(string)[-1]
    return float(E)

def get_Orca_E_array(f):
    """
    same as get_Orca_E but if multiple energies in one file
    """
    string = open(f, 'r').read()
    regex=re.compile("FINAL SINGLE POINT ENERGY\s*([-]?\d*\.\d*)")
    E=regex.findall(string)
    return np.array(E,dtype=float)



def make_gau_com(f, theory='wb97x/6-31g*', s='0', m='1', mod=[], calc=[], title='Title', chk=False, mem=''):
    """
    makes a gaussian input file
    f=xyz or fh file
    s=charge
    m=spin multiplicity
    mod=list of restraints (bonds, angles, or dihedrals to freeze or scan)
    calc=list of keywords
    """
    f_name, f_ext = os.path.splitext(f)
    name=os.path.basename(f_name)
    w = open(name + '.com', 'w')
    w.write(mem)
    if chk==True:
        w.write('%')
        w.write('Chk=%s.chk' '\n' %name)
    w.write('# %s ' %theory)
    for i in calc:
        w.write('%s ' %i)
    w.write('\n\n')
    w.write('%s\n\n' %title)
    w.write('%s %s\n' %(s, m))
    if f_ext=='.fh':
        coor=read_fh(f)
        w.write('%s' '\n' %coor[0])
        for i in range(len(mod)):
            w.write('%s\n' %mod[i])
    else:
        mol = read(f)
        X=mol.get_positions()
        S=mol.get_chemical_symbols()
        for i in range (len(S)):
            x=X[i][0]
            y=X[i][1]
            z=X[i][2]
            w.write('%s %f %f %f' '\n' %(S[i], x, y, z))
    w.write('\n')
    w.close()



