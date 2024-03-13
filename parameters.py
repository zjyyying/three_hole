import math
import numpy as np
M_PI = math.pi

Mc = 6

# Note that Ni-d and O-p orbitals use hole language
# while Nd orbs use electron language
# ed = {'d3z2r2': 0.0,\
#       'dx2y2' : 0.0,\
#       'dxy'   : 0.0,\
#       'dxz'   : 0.0,\
#       'dyz'   : 0.0}

ed = {'d3z2r2': 1.97,\
      'dx2y2' : 0.0,\
      'dxy'   : 1.53,\
      'dxz'   : 1.6,\
      'dyz'   : 1.6}

# after adding H
ed = {'d3z2r2': 3.3,\
      'dx2y2' : 0.0,\
      'dxy'   : 1.55,\
      'dxz'   : 1.9,\
      'dyz'   : 1.9}

eps = np.arange(4.7, 4.71, 1.0)
eH  = 4.8 

As = np.arange(6.0, 6.01, 2.0)
B = 0.15
C = 0.58
#As = np.arange(100, 100.1, 1.0)
# As = np.arange(0.0, 0.01, 1.0)
# B = 0
# C = 0

# Note: tpd and tpp are only amplitude signs are considered separately in hamiltonian.py
# Slater Koster integrals and the overlaps between px and d_x^2-y^2 is sqrt(3) bigger than between px and d_3z^2-r^2 
# These two are proportional to tpd,sigma, whereas the hopping between py and d_xy is proportional to tpd,pi

# IMPORTANT: keep all hoppings below positive to avoid confusion
#            hopping signs are considered in dispersion separately
Norb = 8
if Norb==4 or Norb==8:
    #tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    tpds = np.linspace(1.3, 1.3, num=1, endpoint=True) #[0.25]
#     tpds = [0.000]
    tpps = [0.5]
    #tNiH = 1.63
    tNiHs = np.linspace(4, 4.1, num=1, endpoint=True)
#     tNiHs = [0.000]
    tpH  = 0.58
elif Norb==10 or Norb==12:    
    # pdp = sqrt(3)/4*pds so that tpd(b2)=tpd(b1)/2: see Eskes's thesis and 1990 paper
    # the values of pds and pdp between papers have factor of 2 difference
    # here use Eskes's thesis Page 4
    # also note that tpd ~ pds*sqrt(3)/2
    vals = np.linspace(1.3, 1.3, num=1, endpoint=True)
    pdss = np.asarray(vals)*2./np.sqrt(3)
    pdps = np.asarray(pdss)*np.sqrt(3)/4.
    #pdss = [0.01]
    #pdps = [0.01]
    tNiH = 1.13
    tpH  = 0.58
    #tNiH = 0.01
    tHH = 0.23
    tHH_p = 0.44  # hopping between two planes
    #------------------------------------------------------------------------------
    # note that tpp ~ (pps+ppp)/2
    # because 3 or 7 orbital bandwidth is 8*tpp while 9 orbital has 4*(pps+ppp)
    pps = 0.9
    ppp = 0.2
    #pps = 0.00001
    #ppp = 0.00001

wmin = -8; wmax = 20
eta = 0.1
Lanczos_maxiter = 600

# restriction on variational space
reduce_VS = 0
if_H0_rotate_byU = 1

basis_change_type = 'd_double' #  'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_compute_Aw = 0
if if_compute_Aw==1:
    if_find_lowpeak = 0
    if if_find_lowpeak==1:
        peak_mode = 'lowest_peak' # 'lowest_peak' or 'highest_peak' or 'lowest_peak_intensity'
        if_write_lowpeak_ep_tpd = 1
    if_write_Aw = 1
    if_savefig_Aw = 1
    if_compute_Aw_dd_total = 0

if_get_ground_state = 1
if if_get_ground_state==1:
    # see issue https://github.com/scipy/scipy/issues/5612
    Neval = 10

Ni_orbs = ['dx2y2','dxy','dxz','dyz','d3z2r2']
#Ni_orbs = ['dx2y2','d3z2r2']
H_orbs = ['H']
    
if Norb==4 or Norb==8:
    O1_orbs  = ['px']
    O2_orbs  = ['py']
elif Norb==10:
    O1_orbs  = ['px1','py1']
    O2_orbs  = ['px2','py2']
elif Norb==11:
    O1_orbs  = ['px1','py1']
    O2_orbs  = ['px2','py2']
elif Norb==12:
    O1_orbs  = ['px1','py1','pz1']
    O2_orbs  = ['px2','py2','pz2']
O_orbs = O1_orbs + O2_orbs
# sort the list to facilliate the setup of interaction matrix elements
Ni_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
O_orbs.sort()
print ("Ni_orbs = ", Ni_orbs)
print ("O1_orbs = ",  O1_orbs)
print ("O2_orbs = ",  O2_orbs)
print ("H_orbs = ",  H_orbs)
orbs = Ni_orbs + O_orbs + H_orbs
#assert(len(orbs)==Norb)

Upps = [0]
symmetries = ['1A1','1B1','3B1','1A2','3A2','1E','3E']
print ("compute A(w) for symmetries = ",symmetries)

