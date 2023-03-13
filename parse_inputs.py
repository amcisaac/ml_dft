import os
import numpy as np

# works in theory but final matrices are too large/takes too long
# to do: break into chunks, then append together

atomic_numbers = {'H': 1.0,'C': 6.0,'N': 7.0, 'O': 8.0, 'F': 9.0}

filelist=os.listdir('./data/') #[0:80000]
#filelist = ['dsgdb9nsd_000001.xyz','dsgdb9nsd_000002.xyz']
Nfiles = len(filelist)

Max_Natoms = 0
Ms = []
#Ms = np.zeros((Nfiles))
for k,file in enumerate(filelist):
    try:
        f = open('./data/'+file,'r')
        filelines = f.readlines()
        f.close()
    except UnicodeDecodeError:
        pass

    Nlines = len(filelines)
    Natoms = Nlines - 5
    if Natoms > Max_Natoms: Max_Natoms = Natoms
    i_atom_start = 2
    i_atom_end = 2+Natoms

    atomlines = filelines[i_atom_start:i_atom_end]
    atomlines_clean = []
    for line in atomlines:
        if '*^' in line:
            #print(file)
            #print(line)
            line = line.replace('*^','e')
            #print(line)
        linesp = line.split()
        atom = linesp[0]

        r = np.array([float(linesp[1]),float(linesp[2]),float(linesp[3])])


        atomlines_clean.append([atom,r])

    Mi = np.zeros((Natoms,Natoms))
    for i in range(Natoms):
        Zi = atomic_numbers[atomlines_clean[i][0]]
        Ri = atomlines_clean[i][1]
        #print(Zi, atomlines_clean[i][0])
        for j in range(Natoms):
            Zj = atomic_numbers[atomlines_clean[j][0]]
            Rj = atomlines_clean[j][1]
            if i == j:
                Mi[i][j] = 0.5*Zi**(2.4)
            else:
                Mi[i][j] = Zi * Zj / np.linalg.norm(Ri - Rj)

    Ms.append(Mi)


Ms_padded = []
M_eig_padded = []
for M in Ms:
    n,m = M.shape
    Npad = Max_Natoms - n
    M_pad = np.pad(M,[(0,Npad),(0,Npad)],mode='constant')
    Ms_padded.append(M_pad)

    evals = np.flip(np.array(sorted(np.linalg.eigvals(M_pad),key=np.linalg.norm)),axis=0)
    M_eig_padded.append(evals)


M_array = np.array(Ms_padded) # dimensions of M_array are: N_files x Max_Natoms x Max_Natoms
#print(M_array)
np.save('all_M',M_array)

M_eig_array = np.array(M_eig_padded)
#print(M_eig_array)
np.save('all_eig',M_eig_array)
