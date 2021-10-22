import random
import re
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles

PATH = "/home/mol1/data/compound/"

with open(PATH + "CID-SMILES", "r") as f:
    compound = f.readlines()

random.seed(1)
GOAL = 1000000
sample = random.sample(compound, 2000000)
sample_count = 0
CID_data = []

def replace_halogen(string):
	"""Regex to replace Br,Cl,Sn,Na with single letters"""
	br = re.compile('Br')
	cl = re.compile('Cl')
	sn = re.compile('Sn')
	na = re.compile('Na')
	string = br.sub('R', string)
	string = cl.sub('L', string)
	string = sn.sub('X', string)
	string = na.sub('A', string)
	return string

voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', \
							'(', 'n', '2', 'c', 'F', ')', '=', 'N', '3', 'S', '/', 's', '-', '+', 'o', 'P', \
							 'R', '\\', 'L', '#', 'X', '6', 'B', '7', '4', 'I', '5', 'i', 'p', '8', '9', '%', '0', '.', ':', 'A']

# extract datasets in compound
for c in sample:
    if sample_count == GOAL: break

    c = c[c.find('\t')+1:len(c)-1]
    if len(c) > 150: continue

    try:
        SMILES = MolToSmiles(MolFromSmiles(c), isomericSmiles = False)
        SMILES = replace_halogen(SMILES)
        temp = True
        for k in SMILES:
            if k not in voca_list:
                temp =False
                break
        if temp:
            CID_data.append(c)
            sample_count += 1
    except:
        print("error")

length = len(CID_data)
train = CID_data[:int(length * 0.8)]

#write train-dataset file
with open(PATH + 'Smiles_train.txt', 'w') as f:
    for data in train:
        f.write("%s\n" % data)