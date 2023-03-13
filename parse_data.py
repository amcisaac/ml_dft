import os

# parses dataset into a data file
# this file only parses the energetic data (e.g. not the inputs)

filelist=os.listdir('./data/')[0:80000]
outputfile = open('energy_data_sxlg.csv','w')

header = ','.join(['SMILES','rotational constant A','rotational constant B', \
    'rotational constant C','dipole moment','polarizability','HOMO energy',\
    'LUMO energy','HL gap','spatial extent','ZPE','U 0','U 298','H 298', \
    'G 298','Cv 298','\n'])
outputfile.write(header)
for file in filelist:
    try:
        f = open('./data/'+file,'r')
        filelines = f.readlines()
        f.close()
    except UnicodeDecodeError:
        print(file)
    smile = filelines[-2].split()[0] # SMILES string
    A,B,C,mu,pol,homo,lumo,hl_gap,spat_ext,zpe,U0,U298,H298,G298,Cv298 = filelines[1].split()[2:]
    outputfile.write(','.join([smile,A,B,C,mu,pol,homo,lumo,hl_gap,spat_ext,zpe,U0,U298,H298,G298,Cv298,'\n']))

outputfile.close()
