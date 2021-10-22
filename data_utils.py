import torch
import random
import sentencepiece as spm
import re
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

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

# BERT-CHEM dataset
class BERTChemModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab: spm.SentencePieceProcessor, sep_id: str='[SEP]', cls_id: str='[CLS]',
                mask_id: str='[MASK]', pad_id: str="[PAD]", seq_len: int=512, mask_frac: float=0.15, p: float=0.5):
        """Initiate language modeling dataset.
        Arguments:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
            p (float): probability for NSP. defaut 0.5
        """
        super(BERTChemModelingDataset, self).__init__()
        self.vocab = vocab
        self.data = data
        self.seq_len = seq_len
        self.data_len = len(data)
        self.sep_id = vocab.piece_to_id(sep_id)
        self.cls_id = vocab.piece_to_id(cls_id)
        self.mask_id = vocab.piece_to_id(mask_id)
        self.pad_id = vocab.piece_to_id(pad_id)
        self.p = p
        self.mask_frac = mask_frac

    def getTanimoto(self, firstIdx, secondIdx):
        first = self.data[firstIdx][self.data[firstIdx].find('\t')+1:len(self.data[firstIdx])-1]
        second = self.data[secondIdx][self.data[secondIdx].find('\t')+1:len(self.data[secondIdx])-1]
        SMILES = [MolFromSmiles(first), MolFromSmiles(second)]

        fps = [MACCSkeys.GenMACCSKeys(x) for x in SMILES]
        tanimoto = DataStructs.FingerprintSimilarity(fps[0], fps[1])
        return tanimoto

    def __getitem__(self, i):
        seq1_re = MolToSmiles(MolFromSmiles(self.data[i]), isomericSmiles = False)
        seq1_re = replace_halogen(seq1_re)
        seq1 = self.vocab.encode_as_ids(seq1_re.strip())

        seq2_idx = random.randrange(0, self.data_len)
        while seq2_idx == i:
            seq2_idx = random.randrange(0, self.data_len)

        # decide SSP with using tanimoto
        if self.getTanimoto(i, seq2_idx) > 0.5:
            SSP = torch.tensor(1)
        else:
            SSP = torch.tensor(0)

        seq2_re = MolToSmiles(MolFromSmiles(self.data[seq2_idx]), isomericSmiles = False)
        seq2_re = replace_halogen(seq2_re)
        seq2 = self.vocab.encode_as_ids(seq2_re.strip())

        if len(seq1) + len(seq2) >= self.seq_len - 3: # except 1 [CLS] and 2 [SEP]
            idx = self.seq_len - 3 - len(seq1)
            seq2 = seq2[:idx]
        
        # sentence embedding: 0 for A, 1 for B
        mlm_target = torch.tensor([self.cls_id] + seq1 + [self.sep_id] + seq2 + [self.sep_id] + [self.pad_id] * (self.seq_len - 3 - len(seq1) - len(seq2))).long().contiguous()
        #sent_emb = torch.ones((mlm_target.size(0)))
        #_idx = len(seq1) + 2
        #sent_emb[:_idx] = 0
        
        def masking(data):
            data = torch.tensor(data).long().contiguous()
            data_len = data.size(0)
            ones_num = int(data_len * self.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]
            data = data.masked_fill(lm_mask.bool(), self.mask_id)

            return data

        mlm_train = torch.cat([torch.tensor([self.cls_id]), masking(seq1), torch.tensor([self.sep_id]), masking(seq1), torch.tensor([self.sep_id])]).long().contiguous()
        mlm_train = torch.cat([mlm_train, torch.tensor([self.pad_id] * (512 - mlm_train.size(0)))]).long().contiguous()

        # mlm_train, mlm_target, sentence embedding, NSP target
        return mlm_train, mlm_target, SSP #is_next -> SSP
        # return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab

    def decode(self, x):
        return self.vocab.DecodeIds(x)