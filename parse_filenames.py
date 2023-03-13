import os

# parses dataset into a data file
# this file only parses the energetic data (e.g. not the inputs)

filelist=os.listdir('./data/')
outputfile = open('smiles_files.csv','w')

header = ','.join(['SMILES','input','\n'])
outputfile.write(header)
for file in filelist:
    try:
        f = open('./data/'+file,'r')
        filelines = f.readlines()
        f.close()
    except UnicodeDecodeError:
        print(file)
    smile = filelines[-2].split()[0] # SMILES string

    outputfile.write(','.join([smile,file,'\n']))

outputfile.close()
