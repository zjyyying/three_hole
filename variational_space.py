'''
Contains a class for the variational space for the NiO2 layer
and functions used to represent states as dictionaries.
Distance (L1-norm) between any two particles(holes) cannot > cutoff Mc
'''
import parameters as pam
import lattice as lat
import bisect
import utility as util
import numpy as np

def create_three_hole_state(slabel):
    '''
    Creates a dictionary representing the two-hole state
    '''
    s1 = slabel[0];  orb1 = slabel[1];  x1 = slabel[2];  y1 = slabel[3];  z1 = slabel[4];
    s2 = slabel[5];  orb2 = slabel[6];  x2 = slabel[7];  y2 = slabel[8];  z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14]
    
    assert not (((x3,y3,z3))==(x1,y1,z1) and s3==s1 and orb3==orb1)
    assert not (((x3,y3,z3))==(x2,y2,z2) and s3==s2 and orb3==orb2)
    assert not (((x1,y1,z1))==(x2,y2,z2) and s1==s2 and orb1==orb2)
   
    state = {'type': 'three_hole',\
             'hole1_spin' : slabel[0],  \
             'hole1_orb'  : slabel[1],\
             'hole1_coord': (slabel[2],slabel[3],slabel[4]),\
             'hole2_spin' : slabel[5],  \
             'hole2_orb'  : slabel[6],\
             'hole2_coord': (slabel[7],slabel[8],slabel[9]),\
             'hole3_spin' : slabel[10],  \
             'hole3_orb'  : slabel[11],\
             'hole3_coord': (slabel[12],slabel[13],slabel[14])}
    
    return state

    
def reorder_state(slabel):
    '''
    reorder the s, orb, coord's labeling a state to prepare for generating its canonical state
    Useful for three hole case especially !!!
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    # default
    state_label = slabel
    phase = 1.0
    
    if z1==z2:
        if (x2,y2)<(x1,y1):
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            phase = -1.0

        # note that z1 can differ from z2 in the presence of H orbital
        elif (x1,y1)==(x2,y2):           
            if s1==s2:
                o12 = list(sorted([orb1,orb2]))
                if o12[0]==orb2:
                    state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
                    phase = -1.0  
            elif s1=='dn' and s2=='up':
                state_label = ['up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1]
                phase = -1.0
                
    elif z2>z1:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        phase = -1.0
            
    return state_label, phase
                
    
def make_state_canonical(state):
    '''
    1. There are a few cases to avoid having duplicate states.
    The sign change due to anticommuting creation operators should be 
    taken into account so that phase below has a negative sign
    =============================================================
    Case 1: 
    Note here is different from Mirko's version for only same spin !!
    Now whenever hole2 is on left of hole 1, switch them and
    order the hole coordinates in such a way that the coordinates 
    of the left creation operator are lexicographically
    smaller than those of the right.
    =============================================================
    Case 2: 
    If two holes locate on the same (x,y) sites (even if including apical pz with z=1)
    a) same spin state: 
      up, dxy,    (0,0), up, dx2-y2, (0,0)
    = up, dx2-y2, (0,0), up, dxy,    (0,0)
    need sort orbital order
    b) opposite spin state:
    only keep spin1 = up state
    
    Different from periodic lattice, the phase simply needs to be 1 or -1
    
    2. Besides, see emails with Mirko on Mar.1, 2018:
    Suppose Tpd|state_i> = |state_j> = phase*|canonical_state_j>, then 
    tpd = <state_j | Tpd | state_i> 
        = conj(phase)* <canonical_state_j | Tpp | state_i>
    
    so <canonical_state_j | Tpp | state_i> = tpd/conj(phase)
                                           = tpd*phase
    
    Because conj(phase) = 1/phase, *phase and /phase in setting tpd and tpp seem to give same results
    But need to change * or / in both tpd and tpp functions
    
    Similar for tpp
    '''
    
    # default:
    canonical_state = state
    phase = 1.0
    
    s1 = state['hole1_spin']
    s2 = state['hole2_spin']
    s3 = state['hole3_spin']
    orb1 = state['hole1_orb']
    orb2 = state['hole2_orb']
    orb3 = state['hole3_orb']
    x1, y1, z1 = state['hole1_coord']
    x2, y2, z2 = state['hole2_coord']
    x3, y3, z3 = state['hole3_coord']

    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12,ph = reorder_state(tlabel)
    phase *= ph

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23, ph = reorder_state(tlabel)
    phase *= ph

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp, ph = reorder_state(tlabel)
    phase *= ph

    slabel = tmp+tmp23[5:10]
    canonical_state = create_three_hole_state(slabel)

    return canonical_state, phase

def calc_manhattan_dist(x1,y1,x2,y2):
    '''
    Calculate the Manhattan distance (L1-norm) between two vectors
    (x1,y1) and (x2,y2).
    '''
    out = abs(x1-x2) + abs(y1-y2)
    return out

def check_in_vs_condition(x1,y1,x2,y2):
    '''
    Restrictions: the distance between one hole and Ni-site (0,0)
    and two-hole distance less than cutoff Mc
    '''     
    if calc_manhattan_dist(x1,y1,0,0) > pam.Mc or \
        calc_manhattan_dist(x2,y2,0,0) > pam.Mc or \
        calc_manhattan_dist(x1,y1,x2,y2) > 2*pam.Mc:
        return False
    else:
        return True
    
def check_in_vs_condition1(x1,y1,x2,y2,x3,y3):
    '''
    Restrictions: the distance between one hole and Ni-site (0,0)
    and two-hole distance less than cutoff Mc
    Below use the middle of two Ni is at (1,0)
    '''     
    if calc_manhattan_dist(x1,y1,0,0) > pam.Mc or \
        calc_manhattan_dist(x2,y2,0,0) > pam.Mc or \
        calc_manhattan_dist(x3,y3,0,0) > pam.Mc or \
        calc_manhattan_dist(x1,y1,x2,y2) > 2*pam.Mc or \
        calc_manhattan_dist(x1,y1,x3,y3) > 2*pam.Mc or \
        calc_manhattan_dist(x2,y2,x3,y3) > 2*pam.Mc:
        return False
    else:
        return True

def check_Pauli(slabel):
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
    
    if (s1==s2 and orb1==orb2 and x1==x2 and y1==y2 and z1==z2) or \
        (s1==s3 and orb1==orb3 and x1==x3 and y1==y3 and z1==z3) or \
        (s3==s2 and orb3==orb2 and x3==x2 and y3==y2 and z3==z2):
        return False 
    else:
        return True
    
        
class VariationalSpace:
    '''
    Distance (L1-norm) between any two particles must not exceed a
    cutoff denoted by Mc. 

    Attributes
    ----------
    Mc: Cutoff for the hole-hole 
    lookup_tbl: sorted python list containing the unique identifiers 
        (uid) for all the states in the variational space. A uid is an
        integer which can be mapped to a state (see docsting of get_uid
        and get_state).
    dim: number of states in the variational space, i.e. length of
        lookup_tbl
    filter_func: a function that is passed to create additional 
        restrictions on the variational space. Default is None, 
        which means that no additional restrictions are implemented. 
        filter_func takes exactly one parameter which is a dictionary representing a state.

    Methods
    -------
    __init__
    create_lookup_table
    get_uid
    get_state
    get_index
    '''

    def __init__(self,Mc,filter_func=None):
        self.Mc = Mc
        if filter_func == None:
            self.filter_func = lambda x: True
        else:
            self.filter_func = filter_func
        self.lookup_tbl = self.create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print ("VS.dim = ", self.dim)
        #self.print_VS()

    def print_VS(self):
        for i in range(0,self.dim):
            state = self.get_state(self.lookup_tbl[i])
            
            if state['type'] == 'three_hole':
                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                s3 = state['hole3_spin']
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                orb3 = state['hole3_orb']
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']
                x3, y3, z3 = state['hole3_coord']
                print ('state', i, s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3)
                
                
    def create_lookup_tbl(self):
        '''
        Create a sorted lookup table containing the unique identifiers 
        (uid) of all the states in the variational space.
        
        Manhattan distance between a hole and the Ni-site (0,0) does not exceed Mc
        Then the hole-hole distance cannot be larger than 2*Mc

        Returns
        -------
        lookup_tbl: sorted python list.
        '''
        Mc = self.Mc
        lookup_tbl = []

        # two_hole: hole can only be on Ni or O 
        # hole1:
        for vx in range(-Mc,Mc+1):
            Bv = Mc - abs(vx)
            for vy in range(-Bv,Bv+1):
                for vz in [-1,0,1]:
                    orb1s = lat.get_unit_cell_rep(vx,vy,vz)
                    if orb1s==['NotOnSublattice']:
                        continue

                    # hole2:
                    for wx in range(-Mc,Mc+1):
                        Bw = Mc - abs(wx)
                        for wy in range(-Bw,Bw+1):
                            for wz in [-1,0,1]:
                                orb2s = lat.get_unit_cell_rep(wx,wy,wz)
                                if orb2s==['NotOnSublattice']:
                                    continue
                                    
                                # hole3:
                                for ux in range(-Mc,Mc+1):
                                    Bt = Mc - abs(ux)
                                    for uy in range(-Bt,Bt+1):
                                        for uz in [-1,0,1]:
                                            orb3s = lat.get_unit_cell_rep(ux,uy,uz)
                                            if orb3s==['NotOnSublattice']:
                                                continue

                                            if not check_in_vs_condition1(ux,uy,vx,vy,wx,wy):
                                                continue

                                            for orb1 in orb1s:
                                                for orb2 in orb2s:
                                                    for orb3 in orb3s:
                                                        for s1 in ['up','dn']:
                                                            for s2 in ['up','dn']:   
                                                                for s3 in ['up','dn']: 
                                                                    # neglect d7 state !!
                                                                    if orb1 in pam.Ni_orbs and \
                                                                        orb2 in pam.Ni_orbs and \
                                                                        orb3 in pam.Ni_orbs and \
                                                                        ux==vx==wx==0 or ux==vx==wx==2: #ux uy uz vx vyvz wxwywz是跳跃后的空穴的轨道 对应坐标
                                                                        continue
                                                                        
                                                                    # At most, there are only 2 holes in H
                                                                    # might not necessary since different from d orbitals
                                                                    # H only has one orbital
#                                                                     if orb1 in pam.H_orbs and \
#                                                                         orb2 in pam.H_orbs and \
#                                                                         orb3 in pam.H_orbs and \
#                                                                         uz==vz==wz:
#                                                                         continue
                                                                        
                                                                    # assume one hole is up 
                                                                    if pam.reduce_VS==1:
                                                                        sss = sorted([s1,s2,s3])
                                                                        if sss!=['dn','up','up']:
                                                                            continue

                                                                    # consider Pauli principle
                                                                    slabel = [s1,orb1,vx,vy,vz,\
                                                                              s2,orb2,wx,wy,wz,\
                                                                              s3,orb3,ux,uy,uz]
                                                                    if not check_Pauli(slabel):
                                                                        continue 

                                                                    state = create_three_hole_state(slabel)
                                                                    canonical_state,_ = make_state_canonical(state)

                                                                    if self.filter_func(canonical_state):
                                                                        uid = self.get_uid(canonical_state)
                                                                        lookup_tbl.append(uid)


        lookup_tbl = list(set(lookup_tbl)) # remove duplicates
        lookup_tbl.sort()
        #print "\n lookup_tbl:\n", lookup_tbl
        return lookup_tbl
            
    def check_in_vs(self,state):
        '''
        Check if a given state is in VS

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.
        Mc: integer cutoff for the Manhattan distance.

        Returns
        -------
        Boolean: True or False
        '''
        assert(self.filter_func(state) in [True,False])
        
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        orb3 = state['hole3_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']

        if check_in_vs_condition1(x1,y1,x2,y2,x3,y3):
            return True
        else:
            return False


    def get_uid(self,state):
        '''
        Every state in the variational space is associated with a unique
        identifier (uid) which is an integer number.
        
        Rule for setting uid (example below but showing ideas):
        Assuming that i1, i2 can take the values -1 and +1. Make sure that uid is always larger or equal to 0. 
        So add the offset +1 as i1+1. Now the largest value that (i1+1) can take is (1+1)=2. 
        Therefore the coefficient in front of (i2+1) should be 3. This ensures that when (i2+1) is larger than 0, 
        it will be multiplied by 3 and the result will be larger than any possible value of (i1+1). 
        The coefficient in front of (o1+1) needs to be larger than the largest possible value of (i1+1) + 3*(i2+1). 
        This means that the coefficient in front of (o1+1) must be larger than (1+1) + 3*(1+1) = 8, 
        so you can choose 9 and you get (i1+1) + 3*(i2+1) + 9*(o1+1) and so on ....

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.

        Returns
        -------
        uid (integer) or None if the state is not in the variational space.
        '''
        # Need to check if the state is in the VS, because after hopping the state can be outside of VS
        if not self.check_in_vs(state):
            return None
        
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
        B4 = B1*B3
        B5 = B1*B4
        B6 = B1*B5
        B7 = B1*B6
        N2 = N*N
        N3 = N2*N
        N4 = N3*N
        
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        orb3 = state['hole3_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']

        i1 = lat.spin_int[s1]
        i2 = lat.spin_int[s2]
        i3 = lat.spin_int[s3]
        o1 = lat.orb_int[orb1]
        o2 = lat.orb_int[orb2]
        o3 = lat.orb_int[orb3]

        # note the final +off1 to avoid mixing with no_eh state labeled as 0
        uid = i1 + 2*i2 +4*i3 + 8*(z1+1) + 24*(z2+1) + 72*(z3+1) \
            + 216*o1 +216*N*o2 +216*N2*o3 \
            + 216*N3*( (y1+s) + (x1+s)*B1 + (y2+s)*B2 + (x2+s)*B3 + (y3+s)*B4 + (x3+s)*B5) 

        # check if uid maps back to the original state, namely uid's uniqueness
        tstate = self.get_state(uid)
        ts1 = tstate['hole1_spin']
        ts2 = tstate['hole2_spin']
        ts3 = tstate['hole3_spin']
        torb1 = tstate['hole1_orb']
        torb2 = tstate['hole2_orb']
        torb3 = tstate['hole3_orb']
        tx1, ty1, tz1 = tstate['hole1_coord']
        tx2, ty2, tz2 = tstate['hole2_coord']
        tx3, ty3, tz3 = tstate['hole3_coord']
        assert((s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3)== \
               (ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3))

        return uid
#--------------------------------------------
    def get_state(self,uid):
        '''
        Given a unique identifier, return the corresponding state. 
        ''' 
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
        B4 = B1*B3
        B5 = B1*B4
        B6 = B1*B5
        B7 = B1*B6
        N2 = N*N
        N3 = N2*N
        N4 = N3*N
        
        uid_ = uid 
        x3 = int(uid_/(216*N3*B5)) - s
        uid_ = uid_ % (216*N3*B5)
        y3 = int(uid_/(216*N3*B4)) - s
        uid_ = uid_ % (216*N3*B4)
        x2 = int(uid_/(216*N3*B3)) - s
        uid_ = uid_ % (216*N3*B3)
        y2 = int(uid_/(216*N3*B2)) - s
        uid_ = uid_ % (216*N3*B2)
        x1 = int(uid_/(216*N3*B1)) - s
        uid_ = uid_ % (216*N3*B1)
        y1 = int(uid_/(216*N3)) - s
        uid_ = uid_ % (216*N3)
        o3 = int(uid_/(216*N2)) 
        uid_ = uid_ % (216*N2)
        o2 = int(uid_/(216*N)) 
        uid_ = uid_ % (216*N)
        o1 = int(uid_/216) 
        uid_ = uid_ % 216
        z3 = int(uid_/72) - 1
        uid_ = uid_ % 72
        z2 = int(uid_/24) - 1
        uid_ = uid_ % 24
        z1 = int(uid_/8) - 1 
        uid_ = uid_ % 8
        i3 = int(uid_/4)
        uid_ = uid_ % 4
        i2 = int(uid_/2)
        i1 = uid_ % 2

        orb1 = lat.int_orb[o1]
        orb2 = lat.int_orb[o2]
        orb3 = lat.int_orb[o3]
        s1 = lat.int_spin[i1]
        s2 = lat.int_spin[i2]
        s3 = lat.int_spin[i3]

        slabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
        state = create_three_hole_state(slabel)

        return state

    def get_index(self,state):
        '''
        Return the index under which the state is stored in the lookup
        table.  These indices are consecutive and can be used to
        index, e.g. the Hamiltonian matrix

        Parameters
        ----------
        state: dictionary representing a stat

        Returns
        -------
        index: integer such that lookup_tbl[index] = get_uid(state,Mc).
            If the state is not in the variational space None is returned.
        '''
        uid = self.get_uid(state)
        if uid == None:
            return None
        else:
            index = bisect.bisect_left(self.lookup_tbl,uid)
            if self.lookup_tbl[index] == uid:
                return index
            else:
                return None
