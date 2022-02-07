import sys
from Bio import SeqIO

types = ['test','valid']
file_ins = ['new_'+t+'_hit' for t in types]
for file_in in file_ins:
    fp = open(file_in+'.data','w')
    pos = 0
    neg = 0
    for record in SeqIO.parse(file_in, 'fasta'):

        ids = str(record.description).strip().split()
        data = '\t'.join(ids[1:])
        if int(ids[-1])==0:
            neg+=1
        else:
            pos+=1
        seq = str(record.seq)

        seq1 = seq[:101]
        seq2 = seq[101:]
        fp.write(f'{seq1}\t{seq2}\t{data}\n')
    print(pos)
    print(neg)
