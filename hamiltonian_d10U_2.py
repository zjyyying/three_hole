'''
Functions for constructing individual parts of the Hamiltonian. The 
matrices still need to be multiplied with the appropriate coupling 
constants t_pd, t_pp, etc..

If adding into diagonal Nd atoms (with only s-like orbital as approximation)
then only dxy hops with it
'''
import os
import time
import parameters as pam
import lattice as lat
import variational_space as vs 
import utility as util
import numpy as np
import scipy.sparse as sps

directions_to_vecs = {'UR': (1,1,0),\
                      'UL': (-1,1,0),\
                      'DL': (-1,-1,0),\
                      'DR': (1,-1,0),\
                      'L': (-1,0,0),\
                      'R': (1,0,0),\
                      'U': (0,1,0),\
                      'D': (0,-1,0),\
                      'T': (0,0,1),\
                      'B': (0,0,-1),\
                      'L2': (-2,0,0),\
                      'R2': (2,0,0),\
                      'U2': (0,2,0),\
                      'D2': (0,-2,0),\
                      'T2': (0,0,2),\
                      'B2': (0,0,-2),\
                      'pzL': (-1,0,1),\
                      'pzR': (1,0,1),\
                      'pzU': (0,1,1),\
                      'pzD': (0,-1,1),\
                      'mzL': (-1,0,-1),\
                      'mzR': (1,0,-1),\
                      'mzU': (0,1,-1),\
                      'mzD': (0,-1,-1)}
tpp_nn_hop_dir = ['UR','UL','DL','DR']

def set_tpd_tpp_tNiH_tpH(Norb,tpd,tpp,pds,pdp,pps,ppp,tNiH,tpH):
    # dxz and dyz has no tpd hopping
    if pam.Norb==8:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'px'    : ['L','R'],\
                          'py'    : ['U','D']}
    elif pam.Norb==10 or pam.Norb==11:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D']}
    elif pam.Norb==12:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'dxz'   : ['L','R'],\
                          'dyz'   : ['U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'pz1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D'],\
                          'pz2'   : ['U','D']}
    if pam.Norb==8 or pam.Norb==10 or pam.Norb==11 or pam.Norb==12:
        tNiH_nn_hop_dir = {'d3z2r2': ['T','B'],\
                            'H'    : ['T','B']}
        tpH_nn_hop_dir = {'px': ['pzL','pzR','mzL','mzR'],\
                          'py': ['pzU','pzD','mzU','mzD'],\
                           'H': ['pzL','pzR','mzL','mzR','pzU','pzD','mzU','mzD']}
        
    if pam.Norb==8:
        tpd_orbs = {'d3z2r2','dx2y2','px','py'}
    elif pam.Norb==10:
        tpd_orbs = {'d3z2r2','dx2y2','dxy','px1','py1','px2','py2'}
    elif pam.Norb==12:
        tpd_orbs = {'d3z2r2','dx2y2','dxy','dxz','dyz','px1','py1','pz1','px2','py2','pz2'}
        
    if pam.Norb==8 or pam.Norb==10 or pam.Norb==12:
        tNiH_orbs = {'d3z2r2','H'}
        tpH_orbs  = {'px','py','H'}
        
    # hole language: sign convention followed from Fig 1 in H.Eskes's PRB 1990 paper
    #                or PRB 2016: Characterizing the three-orbital Hubbard model...
    # Or see George's email on Aug.19, 2021:
    # dx2-y2 hoping to the O px in the positive x direction should be positive for holes and for O in the minus x
    # directions should be negative for holes, i.e. the hoping integral should be minus sign of overlap integral
    # between two neighboring atoms. 
    if pam.Norb==8:
        # d3z2r2 has +,-,+ sign structure so that it is negative in x-y plane
        tpd_nn_hop_fac = {('d3z2r2','L','px'): -tpd/np.sqrt(3),\
                          ('d3z2r2','R','px'):  tpd/np.sqrt(3),\
                          ('d3z2r2','U','py'):  tpd/np.sqrt(3),\
                          ('d3z2r2','D','py'): -tpd/np.sqrt(3),\
                          ('dx2y2','L','px'):   tpd,\
                          ('dx2y2','R','px'):  -tpd,\
                          ('dx2y2','U','py'):   tpd,\
                          ('dx2y2','D','py'):  -tpd,\
                          # below just inverse dir of the above one by one
                          ('px','R','d3z2r2'): -tpd/np.sqrt(3),\
                          ('px','L','d3z2r2'):  tpd/np.sqrt(3),\
                          ('py','D','d3z2r2'):  tpd/np.sqrt(3),\
                          ('py','U','d3z2r2'): -tpd/np.sqrt(3),\
                          ('px','R','dx2y2'):   tpd,\
                          ('px','L','dx2y2'):  -tpd,\
                          ('py','D','dx2y2'):   tpd,\
                          ('py','U','dx2y2'):  -tpd}
    elif pam.Norb==10:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'): -pds/2.0,\
                          ('d3z2r2','R','px1'):  pds/2.0,\
                          ('d3z2r2','U','py2'):  pds/2.0,\
                          ('d3z2r2','D','py2'): -pds/2.0,\
                          ('dx2y2','L','px1'):   pds*c,\
                          ('dx2y2','R','px1'):  -pds*c,\
                          ('dx2y2','U','py2'):   pds*c,\
                          ('dx2y2','D','py2'):  -pds*c,\
                          ('dxy','L','py1'):  -pdp,\
                          ('dxy','R','py1'):   pdp,\
                          ('dxy','U','px2'):   pdp,\
                          ('dxy','D','px2'):  -pdp,\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'): -pds/2.0,\
                          ('px1','L','d3z2r2'):  pds/2.0,\
                          ('py2','D','d3z2r2'):  pds/2.0,\
                          ('py2','U','d3z2r2'): -pds/2.0,\
                          ('px1','R','dx2y2'):   pds*c,\
                          ('px1','L','dx2y2'):  -pds*c,\
                          ('py2','D','dx2y2'):   pds*c,\
                          ('py2','U','dx2y2'):  -pds*c,\
                          ('py1','R','dxy'):  -pdp,\
                          ('py1','L','dxy'):   pdp,\
                          ('px2','D','dxy'):   pdp,\
                          ('px2','U','dxy'):  -pdp}
    elif pam.Norb==12:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'): -pds/2.0,\
                          ('d3z2r2','R','px1'):  pds/2.0,\
                          ('d3z2r2','U','py2'):  pds/2.0,\
                          ('d3z2r2','D','py2'): -pds/2.0,\
                          ('dx2y2','L','px1'):   pds*c,\
                          ('dx2y2','R','px1'):  -pds*c,\
                          ('dx2y2','U','py2'):   pds*c,\
                          ('dx2y2','D','py2'):  -pds*c,\
                          ('dxy','L','py1'):  -pdp,\
                          ('dxy','R','py1'):   pdp,\
                          ('dxy','U','px2'):   pdp,\
                          ('dxy','D','px2'):  -pdp,\
                          ('dxz','L','pz1'):  -pdp,\
                          ('dxz','R','pz1'):   pdp,\
                          ('dyz','U','pz2'):   pdp,\
                          ('dyz','D','pz2'):  -pdp,\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'): -pds/2.0,\
                          ('px1','L','d3z2r2'):  pds/2.0,\
                          ('py2','D','d3z2r2'):  pds/2.0,\
                          ('py2','U','d3z2r2'): -pds/2.0,\
                          ('px1','R','dx2y2'):   pds*c,\
                          ('px1','L','dx2y2'):  -pds*c,\
                          ('py2','D','dx2y2'):   pds*c,\
                          ('py2','U','dx2y2'):  -pds*c,\
                          ('py1','R','dxy'):  -pdp,\
                          ('py1','L','dxy'):   pdp,\
                          ('px2','D','dxy'):   pdp,\
                          ('px2','U','dxy'):  -pdp,\
                          ('pz1','R','dxz'):  -pdp,\
                          ('pz1','L','dxz'):   pdp,\
                          ('pz2','D','dyz'):   pdp,\
                          ('pz2','U','dyz'):  -pdp}
    ########################## tpp below ##############################
    if pam.Norb==8:
        tpp_nn_hop_fac = {('UR','px','py'): -tpp,\
                          ('UL','px','py'):  tpp,\
                          ('DL','px','py'): -tpp,\
                          ('DR','px','py'):  tpp}
    elif pam.Norb==10:
        tpp_nn_hop_fac = {('UR','px1','px2'):  0.5*(ppp-pps),\
                          ('UL','px1','px2'):  0.5*(ppp-pps),\
                          ('DL','px1','px2'):  0.5*(ppp-pps),\
                          ('DR','px1','px2'):  0.5*(ppp-pps),\
                          ('UR','py1','py2'):  0.5*(ppp-pps),\
                          ('UL','py1','py2'):  0.5*(ppp-pps),\
                          ('DL','py1','py2'):  0.5*(ppp-pps),\
                          ('DR','py1','py2'):  0.5*(ppp-pps),\
                          ('UR','px1','py2'): -0.5*(ppp+pps),\
                          ('UL','px1','py2'):  0.5*(ppp+pps),\
                          ('DL','px1','py2'): -0.5*(ppp+pps),\
                          ('DR','px1','py2'):  0.5*(ppp+pps),\
                          ('UR','px2','py1'): -0.5*(ppp+pps),\
                          ('UL','px2','py1'):  0.5*(ppp+pps),\
                          ('DL','px2','py1'): -0.5*(ppp+pps),\
                          ('DR','px2','py1'):  0.5*(ppp+pps)}
    elif pam.Norb==12:
        tpp_nn_hop_fac = {('UR','px1','px2'):  0.5*(ppp-pps),\
                          ('UL','px1','px2'):  0.5*(ppp-pps),\
                          ('DL','px1','px2'):  0.5*(ppp-pps),\
                          ('DR','px1','px2'):  0.5*(ppp-pps),\
                          ('UR','py1','py2'):  0.5*(ppp-pps),\
                          ('UL','py1','py2'):  0.5*(ppp-pps),\
                          ('DL','py1','py2'):  0.5*(ppp-pps),\
                          ('DR','py1','py2'):  0.5*(ppp-pps),\
                          ('UR','px1','py2'): -0.5*(ppp+pps),\
                          ('UL','px1','py2'):  0.5*(ppp+pps),\
                          ('DL','px1','py2'): -0.5*(ppp+pps),\
                          ('DR','px1','py2'):  0.5*(ppp+pps),\
                          ('UR','px2','py1'): -0.5*(ppp+pps),\
                          ('UL','px2','py1'):  0.5*(ppp+pps),\
                          ('DL','px2','py1'): -0.5*(ppp+pps),\
                          ('DR','px2','py1'):  0.5*(ppp+pps),\
                          ('UR','pz1','pz2'):  ppp,\
                          ('UL','pz1','pz2'):  ppp,\
                          ('DL','pz1','pz2'):  ppp,\
                          ('DR','pz1','pz2'):  ppp}
        
    ########################## tNiH below ##############################
    # H only hops to Ni's d3z2-r2
    if pam.Norb==8 or pam.Norb==10 or pam.Norb==12:
        tNiH_nn_hop_fac = {('d3z2r2','T','H'): tNiH,\
                           ('d3z2r2','B','H'): tNiH,\
                           ('H','T','d3z2r2'): tNiH,\
                           ('H','B','d3z2r2'): tNiH}
        tpH_nn_hop_fac = {('H','pzL','px'):  tpH,\
                          ('H','pzR','px'): -tpH,\
                          ('H','mzL','px'):  tpH,\
                          ('H','mzR','px'): -tpH,\
                          ('H','pzU','py'): -tpH,\
                          ('H','pzD','py'):  tpH,\
                          ('H','mzU','py'): -tpH,\
                          ('H','mzD','py'):  tpH,\
                          ('px','pzL','H'): -tpH,\
                          ('px','pzR','H'):  tpH,\
                          ('px','mzL','H'): -tpH,\
                          ('px','mzR','H'):  tpH,\
                          ('py','pzU','H'):  tpH,\
                          ('py','pzD','H'): -tpH,\
                          ('py','mzU','H'):  tpH,\
                          ('py','mzD','H'): -tpH}
        
    return tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac, \
           tNiH_nn_hop_dir, tNiH_orbs, tNiH_nn_hop_fac, tpH_nn_hop_dir, tpH_orbs, tpH_nn_hop_fac
        
    
def get_interaction_mat(A, sym):
    '''
    Get d-d Coulomb and exchange interaction matrix
    total_spin based on lat.spin_int: up:1 and dn:0
    
    Rotating by 90 degrees, x goes to y and indeed y goes to -x so that this basically interchanges 
    spatial wave functions of two holes and can introduce - sign (for example (dxz, dyz)).
    But one has to look at what such a rotation does to the Slater determinant state of two holes.
    Remember the triplet state is (|up,down> +|down,up>)/sqrt2 so in interchanging the holes 
    the spin part remains positive so the spatial part must be negative. 
    For the singlet state interchanging the electrons the spin part changes sign so the spatial part can stay unchanged.
    
    Triplets cannot occur for two holes in the same spatial wave function while only singlets only can
    But both singlets and triplets can occur if the two holes have orthogonal spatial wave functions 
    and they will differ in energy by the exchange terms
    
    ee denotes xz,xz or xz,yz depends on which integral <ab|1/r_12|cd> is nonzero, see handwritten notes
    
    AorB_sym = +-1 is used to label if the state is (e1e1+e2e2)/sqrt(2) or (e1e1-e2e2)/sqrt(2)
    For syms (in fact, all syms except 1A1 and 1B1) without the above two states, AorB_sym is set to be 0
    
    Here different from all other codes, change A by A/2 to decrease the d8 energy to some extent
    the remaining A/2 is shifted to d10 states, see George's email on Jun.21, 2021
    and also set_edepeH subroutine
    '''
    B = pam.B
    C = pam.C
    
    # not useful if treat 1A1 and 1B1 as correct ee states as (exex +- eyey)/sqrt(2)
    if sym=='1AB1':
        fac = np.sqrt(6)
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 4,\
                       ('d3z2r2','dx2y2') : 5}
        interaction_mat = [[A/2.+4.*B+3.*C,  4.*B+C,       4.*B+C,           B+C,           B+C,       0], \
                           [4.*B+C,       A/2.+4.*B+3.*C,  C,             3.*B+C,        3.*B+C,       0], \
                           [4.*B+C,       C,            A/2.+4.*B+3.*C,   3.*B+C,        3.*B+C,       0], \
                           [B+C,          3.*B+C,       3.*B+C,        A/2.+4.*B+3.*C,   3.*B+C,       B*fac], \
                           [B+C,          3.*B+C,       3.*B+C,        3.*B+C,        A/2.+4.*B+3.*C, -B*fac], \
                           [0,            0,            0,              B*fac,         -B*fac,      A/2.+2.*C]]
    elif sym=='1A1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 1
        fac = np.sqrt(2)
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 3}
        interaction_mat = [[A/2.+4.*B+3.*C,  4.*B+C,       4.*B+C,        fac*(B+C)], \
                           [4.*B+C,       A/2.+4.*B+3.*C,  C,             fac*(3.*B+C)], \
                           [4.*B+C,       C,            A/2.+4.*B+3.*C,   fac*(3.*B+C)], \
                           [fac*(B+C),    fac*(3.*B+C), fac*(3.*B+C),  A/2.+7.*B+4.*C]]
    elif sym=='1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dx2y2'): 0,\
                       ('dxz','dxz')     : 1,\
                       ('dyz','dyz')     : 1}
        interaction_mat = [[A/2.+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A/2.+B+2.*C]]
    elif sym=='1A2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0}
        interaction_mat = [[A/2.+4.*B+2.*C]]
    elif sym=='3A2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0,\
                       ('dxz','dyz')  : 1}
        interaction_mat = [[A/2.+4.*B,   6.*B], \
                           [6.*B,     A/2.-5.*B]]
    elif sym=='3B1':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('d3z2r2','dx2y2'): 0}
        interaction_mat = [[A/2.-8.*B]]
    elif sym=='1B2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxy'): 0,\
                       ('dxz','dyz')   : 1}
        interaction_mat = [[A/2.+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A/2.+B+2.*C]]
    elif sym=='3B2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = -1
        state_order = {('d3z2r2','dxy'): 0}
        interaction_mat = [[A/2.-8.*B]]
    elif sym=='1E':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}    
        interaction_mat = [[A/2.+3.*B+2.*C,  0,           -B*fac,      0,          0,        -B*fac], \
                           [0,            A/2.+3.*B+2.*C,  0,          B*fac,     -B*fac,     0], \
                           [-B*fac,       0,            A/2.+B+2.*C,   0,          0,        -3.*B], \
                           [0,            B*fac,        0,          A/2.+B+2.*C,   3.*B,      0 ], \
                           [0,           -B*fac,        0,          3.*B,       A/2.+B+2.*C,  0], \
                           [-B*fac,       0,           -3.*B,       0,          0,         A/2.+B+2.*C]]
    elif sym=='3E':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}        
        interaction_mat = [[A/2.+B,         0,         -3.*B*fac,    0,          0,        -3.*B*fac], \
                           [0,           A/2.+B,        0,           3.*B*fac,  -3.*B*fac,  0], \
                           [-3.*B*fac,   0,          A/2.-5.*B,      0,          0,         3.*B], \
                           [0,           3.*B*fac,   0,           A/2.-5.*B,    -3.*B,      0 ], \
                           [0,          -3.*B*fac,   0,          -3.*B,       A/2.-5.*B,    0], \
                           [-3.*B*fac,   0,          3.*B,        0,          0,         A/2.-5.*B]]
        
    return state_order, interaction_mat, Stot, Sz_set, AorB_sym

def set_matrix_element(row,col,data,new_state,col_index,VS,element):
    '''
    Helper function that is used to set elements of a matrix using the
    sps coo format.

    Parameters
    ----------
    row: python list containing row indices
    col: python list containing column indices
    data: python list containing non-zero matrix elements
    col_index: column index that is to be appended to col
    new_state: new state corresponding to the row index that is to be
        appended.
    VS: VariationalSpace class from the module variationalSpace
    element: (complex) matrix element that is to be appended to data.

    Returns
    -------
    None, but appends values to row, col, data.
    '''
    row_index = VS.get_index(new_state)
    if row_index != None:
        data.append(element)
        row.append(row_index)
        col.append(col_index)

def create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac):
    '''
    Create nearest neighbor (NN) pd hopping part of the Hamiltonian
    Only hole can hop with tpd

    Parameters
    ----------
    VS: VariationalSpace class from the module variationalSpace
    
    Returns
    -------
    matrix: (sps coo format) t_pd hopping part of the Hamiltonian without 
        the prefactor t_pd.
    
    Note from the sps documentation
    -------------------------------
    By default when converting to CSR or CSC format, duplicate (i,j)
    entries will be summed together
    '''    
    print ("start create_tpd_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    tpd_keys = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # double check which cost some time, might not necessary
        assert VS.get_uid(start_state) == VS.lookup_tbl[i]
        
        # only hole hops with tpd
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']
        x3, y3, z3 = start_state['hole3_coord']

        # hole 1 hops: some d-orbitals might have no tpd
        if orb1 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb1]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                if orbs1 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1+vx,y1+vy,x2,y2,x3,y3):
                    continue

                # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                for o1 in orbs1:
                    if o1 not in tpd_orbs:
                        continue
                        
                    # consider Pauli principle
                    slabel = [s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    # debug:
#                     row_index = VS.get_index(new_state)
#                     print(row_index)
                    
#                     ts1 = new_state['hole1_spin']
#                     ts2 = new_state['hole2_spin']
#                     ts3 = new_state['hole3_spin']
#                     torb1 = new_state['hole1_orb']
#                     torb2 = new_state['hole2_orb']
#                     torb3 = new_state['hole3_orb']
#                     tx1, ty1, tz1 = new_state['hole1_coord']
#                     tx2, ty2, tz2 = new_state['hole2_coord']
#                     tx3, ty3, tz3 = new_state['hole3_coord']
                    #if row_index == None:
                        #print ('state after tpd', ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3)

                    o12 = tuple([orb1, dir_, o1])
                    if o12 in tpd_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)

        # hole 2 hops; some d-orbitals might have no tpd
        if orb2 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb2]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)
                if orbs2 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1,y1,x2+vx,y2+vy,x3,y3):
                    continue

                for o2 in orbs2:
                    if o2 not in tpd_orbs:
                        continue
                        
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb2, dir_, o2])
                    if o12 in tpd_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)
                        
        # hole 3 hops; some d-orbitals might have no tpd
        if orb3 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb3]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs3 = lat.get_unit_cell_rep(x3+vx, y3+vy, z3+vz)
                if orbs3 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1,y1,x2,y2,x3+vx,y3+vy):
                    continue

                for o3 in orbs3:
                    if o3 not in tpd_orbs:
                        continue
                        
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,o3,x3+vx,y3+vy,z3+vz]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb3, dir_, o3])
                    if o12 in tpd_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out

def create_tNiH_nn_matrix(VS, tNiH_nn_hop_dir, tNiH_orbs, tNiH_nn_hop_fac):
    '''
    Create nearest neighbor (NN) Ni to H hopping part of the Hamiltonian
    Note that d3z2r2 has +,-,+ sign structure so that it is - in x-y plane and dz2-H hopping integral is positive
    so that for electrons the hopping should be negative
    '''    
    print ("start create_tNiH_nn_matrix")
    print ("===========================")
    
    dim = VS.dim
    tNiH_keys = tNiH_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']
        x3, y3, z3 = start_state['hole3_coord']
            
        # hole 1 hops
        if orb1 in tNiH_orbs:
            for dir_ in tNiH_nn_hop_dir[orb1]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                if orbs1 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1+vx,y1+vy,x2,y2,x3,y3):
                    continue

                # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                for o1 in orbs1:
                    if o1 not in tNiH_orbs:
                        continue
                    
                    # consider Pauli principle
                    slabel = [s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb1, dir_, o1])
                    if o12 in tNiH_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tNiH_nn_hop_fac[o12]*ph)

        # hole 2 hops
        if orb2 in tNiH_orbs:
            for dir_ in tNiH_nn_hop_dir[orb2]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)
                if orbs2 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1,y1,x2+vx,y2+vy,x3,y3):
                    continue

                for o2 in orbs2:
                    if o2 not in tNiH_orbs:
                        continue
                        
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb2, dir_, o2])
                    if o12 in tNiH_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tNiH_nn_hop_fac[o12]*ph)
                        
        # hole 3 hops
        if orb3 in tNiH_orbs:
            for dir_ in tNiH_nn_hop_dir[orb3]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs3 = lat.get_unit_cell_rep(x3+vx, y3+vy, z3+vz)
                if orbs3 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1,y1,x2,y2,x3+vx,y3+vy):
                    continue

                for o3 in orbs3:
                    if o3 not in tNiH_orbs:
                        continue
                        
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,o3,x3+vx,y3+vy,z3+vz]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb3, dir_, o3])
                    if o12 in tNiH_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tNiH_nn_hop_fac[o12]*ph)

                            
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    #idx = list(col).index(None)
    #print idx, row[idx],col[idx],data[idx]
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out

def create_tpH_nn_matrix(VS, tpH_nn_hop_dir, tpH_orbs, tpH_nn_hop_fac):
    '''
    Create nearest neighbor O to H hopping part of the Hamiltonian
    '''    
    print ("start create_tpH_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    tpH_keys = tpH_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']
        x3, y3, z3 = start_state['hole3_coord']
            
        # hole 1 hops
        if orb1 in tpH_orbs:
            for dir_ in tpH_nn_hop_dir[orb1]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                if orbs1 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1+vx,y1+vy,x2,y2,x3,y3):
                    continue

                # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                for o1 in orbs1:
                    if o1 not in tpH_orbs:
                        continue
                    
                    # consider Pauli principle
                    slabel = [s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb1, dir_, o1])
                    if o12 in tpH_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpH_nn_hop_fac[o12]*ph)

        # hole 2 hops
        if orb2 in tpH_orbs:
            for dir_ in tpH_nn_hop_dir[orb2]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)
                if orbs2 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1,y1,x2+vx,y2+vy,x3,y3):
                    continue

                for o2 in orbs2:
                    if o2 not in tpH_orbs:
                        continue
                        
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb2, dir_, o2])
                    if o12 in tpH_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpH_nn_hop_fac[o12]*ph)
                        
        # hole 3 hops
        if orb3 in tpH_orbs:
            for dir_ in tpH_nn_hop_dir[orb3]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs3 = lat.get_unit_cell_rep(x3+vx, y3+vy, z3+vz)
                if orbs3 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1,y1,x2,y2,x3+vx,y3+vy):
                    continue

                for o3 in orbs3:
                    if o3 not in tpH_orbs:
                        continue
                        
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,o3,x3+vx,y3+vy,z3+vz]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb3, dir_, o3])
                    if o12 in tpH_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpH_nn_hop_fac[o12]*ph)

                            
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    #idx = list(col).index(None)
    #print idx, row[idx],col[idx],data[idx]
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out


def create_tpp_nn_matrix(VS,tpp_nn_hop_fac): 
    '''
    similar to comments in create_tpp_nn_matrix
    '''   
    print ("start create_tpp_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    tpp_orbs = tpp_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # only hole hops with tpp
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']
        x3, y3, z3 = start_state['hole3_coord']
            
        # hole1 hops: only p-orbitals has t_pp 
        if orb1 in pam.O_orbs: 
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)

                if orbs1!=pam.O1_orbs and orbs1!=pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition1(x1+vx,y1+vy,x2,y2,x3,y3):
                    continue

                # consider t_pp for all cases; when one hole hops, the other hole should not change orb
                for o1 in orbs1:
                    # consider Pauli principle
                    slabel = [s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb1, dir_, o1])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

        # hole 2 hops, only p-orbitals has t_pp 
        if orb2 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)

                if orbs2!=pam.O1_orbs and orbs2!=pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition1(x1,y1,x2+vx,y2+vy,x3,y3): 
                    continue

                for o2 in orbs2:
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz,s3,orb3,x3,y3,z3]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb2, dir_, o2])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)
                        
        # hole 3 hops, only p-orbitals has t_pp 
        if orb3 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs3 = lat.get_unit_cell_rep(x3+vx, y3+vy, z3+vz)

                if orbs3!=pam.O1_orbs and orbs3!=pam.O2_orbs:
                    continue
                    
                if not vs.check_in_vs_condition1(x1,y1,x2,y2,x3+vx,y3+vy): 
                    continue

                for o3 in orbs3:
                    # consider Pauli principle
                    slabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,o3,x3+vx,y3+vy,z3+vz]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_three_hole_state(slabel)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb3, dir_, o3])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        #print(s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,o3,x3,y3,z3)
                        #print('hops to ',s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,o3,x3+vx,y3+vy,z3+vz)
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out


def create_edepeH_diag_matrix(VS,A,ep):
    '''
    Create diagonal part of the site energies
    '''    
    t1 = time.time()
    print ("start create_edepeH_diag_matrix")
    print ("Separate U into d8 and d10     ")
    print ("================================")
    dim = VS.dim
    data = []
    row = []
    col = []
    idxs = np.zeros(dim)

    for i in range(0,dim):
        diag_el = 0.
        state = VS.get_state(VS.lookup_tbl[i])

        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        orb3 = state['hole3_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']

        if orb1 in pam.Ni_orbs:
            diag_el += pam.ed[orb1]
        elif orb1 in pam.O_orbs:
            diag_el += ep
        elif orb1 in pam.H_orbs:
            diag_el += pam.eH

        if orb2 in pam.Ni_orbs:
            diag_el += pam.ed[orb2]
        elif orb2 in pam.O_orbs:
            diag_el += ep
        elif orb2 in pam.H_orbs:
            diag_el += pam.eH
            
        if orb3 in pam.Ni_orbs:
            diag_el += pam.ed[orb3]
        elif orb3 in pam.O_orbs:
            diag_el += ep
        elif orb3 in pam.H_orbs:
            diag_el += pam.eH
            
        '''
        below for finding states consisting of d10 to add A/2
        找到d10
        '''    
        # obtain which orbs are d and p
        # dxs stores x coordinate of d orb for the case of 2 Ni
        nNi, nO, nH, dorbs, dxs, porbs, Horbs = util.get_statistic_3orb(orb1,orb2,orb3,x1,x2,x3)

        # d7 is not allowed in VS
        # only need to consider cases of different nNi
        if nNi==2 and dxs[0]==dxs[1]:
            # there must be one Ni with d8 and the other d10 state:
            diag_el += A/2.

        elif nNi==1: 
            # there must be one Ni with d9 and the other d10 state
            diag_el += A/2.

        elif nNi==0: 
            # both two Ni with d10
            diag_el += A

        data.append(diag_el); row.append(i); col.append(i)
        #print i, diag_el


    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    #print min(data)
    #print len(row), len(col)
    
   # for ii in range(0,len(row)):
   #     if data[ii]==0:
   #         print ii
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    #print("---create_edepeH_diag_matrix %s seconds ---" % (time.time() - t1))
    
    return out
    

def get_double_occu_list(VS):
    '''
    Get the list of states that two holes are both d or p-orbitals
    idx, hole3state, dp_orb, dp_pos record detailed info of states
    '''
    dim = VS.dim
    d_list = []; p_list = []
    idx = []; hole3_part = []; double_part = []
    
    for i in range(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']

        # find out which two holes are on Ni
        # idx is to label which hole is not on Ni
        if (x1, y1, z1)==(x2, y2, z2):
            if o1 in pam.Ni_orbs and o2 in pam.Ni_orbs:
                d_list.append(i)
                idx.append(3); hole3_part.append([s3, o3, x3, y3, z3])
                double_part.append([s1,o1,x1,y1,z1,s2,o2,x2,y2,z2])
            elif o1 in pam.O_orbs and o2 in pam.O_orbs:
                p_list.append(i)

        elif (x1, y1, z1)==(x3, y3, z3):
            if o1 in pam.Ni_orbs and o3 in pam.Ni_orbs:
                d_list.append(i)
                idx.append(2); hole3_part.append([s2, o2, x2, y2, z2])
                double_part.append([s1,o1,x1,y1,z1,s3,o3,x3,y3,z3])
            elif o1 in pam.O_orbs and o3 in pam.O_orbs:
                p_list.append(i)

        elif (x2, y2, z2)==(x3, y3, z3):
            if o2 in pam.Ni_orbs and o3 in pam.Ni_orbs:
                d_list.append(i)
                idx.append(1); hole3_part.append([s1, o1, x1, y1, z1])
                double_part.append([s2,o2,x2,y2,z2,s3,o3,x3,y3,z3])
            elif o2 in pam.O_orbs and o3 in pam.O_orbs:
                p_list.append(i)

    print ("len(d_list)", len(d_list))
    print ("len(p_list)", len(p_list))
    
    return d_list, p_list, double_part, idx, hole3_part



def create_interaction_matrix_ALL_syms(VS,d_double, p_double, double_part, idx, hole3_part,S_val, Sz_val, AorB_sym, A, Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets including all symmetries
    
    Loop over all d_double states, find the corresponding sym channel; 
    the other loop over all d_double states, if it has same sym channel and S, Sz
    enter into the matrix element
    
    There are some complications or constraints due to three holes and one Nd electron:
    From H_matrix_reducing_VS file, to set up interaction between states i and j:
    1. i and j belong to the same type, same order of orbitals to label the state (idxi==idxj below)
    2. i and j's spins are same; or L and s should also have same spin
    3. Positions of L and Nd-electron should also be the same
    
    Generate the states interacting with i for a particular sym instead of finding it
    '''    
    t1 = time.time()
    print ("start create_interaction_matrix_ALL_syms (three holes)")
    print ("======================================================")
    
    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []
    
    channels = ['1A1','1A2','3A2','1B1','3B1','1E','3E','1B2','3B2']

    if os.path.isfile('./test.txt'):
        os.remove('./test.txt')
    f = open('./test.txt','w',1) 
    
    for sym in channels:
        state_order, interaction_mat, Stot, Sz_set, AorB = get_interaction_mat(A, sym)
        sym_orbs = state_order.keys()
        #print ("orbitals in sym ", sym, "= ", sym_orbs)

        for i, double_id in enumerate(d_double):
            count = []  # temporarily store states interacting with i to avoid double count
            
            s1 = double_part[i][0]
            o1 = double_part[i][1]
            s2 = double_part[i][5]
            o2 = double_part[i][6]
            dpos = double_part[i][2:5]

            o12 = sorted([o1,o2])
            o12 = tuple(o12)
            
            # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
            S12  = S_val[double_id]
            Sz12 = Sz_val[double_id]

            # continue only if (o1,o2) is within desired sym
            if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
                continue

            if (o1==o2=='dxz' or o1==o2=='dyz') and AorB_sym[double_id]!=AorB:
                continue

            # get the corresponding index in sym for setting up matrix element
            idx1 = state_order[o12]
            
#             if double_id==1705:
#                 print(S12,Sz12,AorB_sym[double_id],idx1,AorB)
            
            '''
            Below: generate len(sym_orbs) states that interact with i for a particular sym
            '''
            for idx2, o34 in enumerate(sym_orbs):
                # ('dyz','dyz') is degenerate with ('dxz','dxz') for D4h 
                if o34==('dyz','dyz'):
                    idx2 -= 1
                    
                # Because VS's make_state_canonical follows the rule of up, dn order
                # then the state like ['up', 'dxy', 0, 0, 0, 'dn', 'dx2y2', 0, 0, 0]'s
                # order is opposite to (dx2y2,dxy) order in interteration_mat
                # Here be careful with o34's order that can be opposite to o12 !!
                
                for s1 in ('up','dn'):
                    for s2 in ('up','dn'):
                        if idx[i]==3:
                            slabel = [s1,o34[0]]+dpos + [s2,o34[1]]+dpos + hole3_part[i]
                        if idx[i]==2:
                            slabel = [s1,o34[0]]+dpos + hole3_part[i] + [s2,o34[1]]+dpos 
                        if idx[i]==1:
                            slabel = hole3_part[i] + [s1,o34[0]]+dpos + [s2,o34[1]]+dpos 

                        if not vs.check_Pauli(slabel):
                            continue
                        
                        tmp_state = vs.create_three_hole_state(slabel)
                        new_state,_ = vs.make_state_canonical(tmp_state)
                        j = VS.get_index(new_state)
                        
#                             # debug:
#                             if double_id==1705:
#                                 tse = new_state['e_spin']
#                                 ts1 = new_state['hole1_spin']
#                                 ts2 = new_state['hole2_spin']
#                                 ts3 = new_state['hole3_spin']
#                                 torbe = new_state['e_orb']
#                                 torb1 = new_state['hole1_orb']
#                                 torb2 = new_state['hole2_orb']
#                                 torb3 = new_state['hole3_orb']
#                                 txe, tye, tze = new_state['e_coord']
#                                 tx1, ty1, tz1 = new_state['hole1_coord']
#                                 tx2, ty2, tz2 = new_state['hole2_coord']
#                                 tx3, ty3, tz3 = new_state['hole3_coord']
#                                 print (j,tse,torbe,txe,tye,tze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3)

                        if j!=None and j not in count:
                            S34  = S_val[j]
                            Sz34 = Sz_val[j]
    
                            if not (S34==S12 and Sz34==Sz12):
                                continue

                            if o34==('dxz','dxz') or o34==('dyz','dyz'):
                                if AorB_sym[j]!=AorB:
                                    continue

                            # debug
#                             if j>=double_id:
#                                 if double_id==1705:
#                                     print (double_id,o12[0],o12[1],S12,Sz12," ",j,o34[0],o34[1],S34,Sz34," ", \
#                                            interaction_mat[idx1][idx2])

#                                 f.write('{:.6e}\t{:.6e}\t{:.6e}\n'.format(double_id, j, interaction_mat[idx1][idx2]))

                            val = interaction_mat[idx1][idx2]
                            data.append(val); row.append(double_id); col.append(j)
                            count.append(j)
                    
    # Create Upp matrix for p-orbital multiplets
    if Upp!=0:
        for i in p_double:
            data.append(Upp); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    print("---create_interaction_matrix_ALL_syms %s seconds ---" % (time.time() - t1))
    
    return out
