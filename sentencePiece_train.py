import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=Smiles_train.txt --model_prefix=m --vocab_size=2000 --model_type=char')

sp = spm.SentencePieceProcessor()
sp.load("m.model")