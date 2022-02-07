import sys

types = ['train','test','valid']
file_ins = ['new_'+t+'.data' for t in types]
for file_in in file_ins:
    fpo = open(file_in+'.fa','w')

    count =0
    for line in open(file_in):
        lines = line.strip().split()
        seq = lines[0] + lines[1]
        annot = '\t'.join(lines[2:])
        fpo.write(f'>{count}\t{annot}\n{seq}\n')