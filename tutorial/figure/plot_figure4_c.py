from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle as pkl

np.random.seed(1234)
df = {'interaction':[],'type':[],'chromosome':[]}

## 设定需要分析的染色体序号
chrs= 'chr1'

annotation = '../../ref/gencode.v38.chr_patch_hapl_scaff.annotation.gff3'

# 提取该染色体上所有基因的数据
# 上述注释数据已经按照染色体进行排序
flag = False
gene2pos = {}

for line in open(annotation):
    line = line.strip().split()
    if line[0] == chrs:
        flag = True
        if line[1] == 'HAVANA' and line[2]== 'gene':
            # get id
            annot = line[-1].split(';')[0].split('=')[1].split('.')[0]
            start = line[3]
            end = line[4]
            strand = line[6]
            gene2pos[annot] = (int(start),int(end),strand)
    else:
        if flag:
            break


# 先获取genename2idx
# rna_seq data
features = 'data/GSM4006843_HUVEC_control_scRNA_1_features.tsv'
gene2idx = {}
for idx, line in enumerate(open(features).readlines()):
    line = line.strip().split()
    if line[0] in gene2pos:
        gene2idx[line[0]] = idx
idx2gene = {value:key for key,value in gene2idx.items()}


# 在获取每个基因的表达量count
rnaseq = 'data/GSM4006843_HUVEC_control_scRNA_1_matrix.mtx'
fp = open(rnaseq)
# skip first three line
fp.readline()
fp.readline()
fp.readline()

gene2expression = {}
for line in fp:
    line = line.strip().split()
    idx = int(line[0]) -1 # convert to zero-start
    if idx in idx2gene:
        gene = idx2gene[idx]
        if gene not in gene2expression:
            gene2expression[gene] = 0
        gene2expression[gene]+=1

normal_inner_gene2inter = pkl.load(open('result/normal_inner_gene2inter.bin','rb'))
normal_intra_gene2inter = pkl.load(open('result/normal_intra_gene2inter.bin','rb'))

exp_inner_gene2inter = pkl.load(open('result/exp_inner_gene2inter.bin','rb'))
exp_intra_gene2inter = pkl.load(open('result/exp_intra_gene2inter.bin','rb'))

marker_genes = ['TMSB4X','ISG15','UBE2S','H2AFZ','CKS1B','TUBA1B','PTTG1','HMGB2','IFI27','FTH1','CCL2','HLA-B',
               'LTB','CXCL8','LTB','IFI27','SERPINE1','RPS2','PGF','TMSB4X','PTMA','SOX18']
highly_genes = pkl.load(open('result/highly_genes.bin','rb'))

#marker_genes 转换成ENSMBL_ID
# 先获取genename2idx
features = 'data/GSM4006843_HUVEC_control_scRNA_1_features.tsv'
name2gene = {}
for line in open(features):
    lines= line.strip().split()
    name2gene[lines[1]] = lines[0]

# 这里有个问题，是否需要考虑strand带来的影响呢？
def getGene(mid, strand):
    for gene_id, value in gene2pos.items():
        gstart, gend, gstrand = value
        if strand ==  gstrand:
            if mid <= gend and mid>=gstart:
                return gene_id
    return ""

data = []
for gene in marker_genes:
    if gene not in name2gene:
        continue
    ids = name2gene[gene]
    if ids in normal_inner_gene2inter and ids in exp_inner_gene2inter:
        if ids in normal_intra_gene2inter and ids in exp_intra_gene2inter:
        #data.append((normal_inner_gene2inter[ids],exp_inner_gene2inter[ids]))
            a = normal_inner_gene2inter[ids]/(normal_inner_gene2inter[ids]+normal_intra_gene2inter[ids])
            b = exp_inner_gene2inter[ids]/(exp_inner_gene2inter[ids]+exp_intra_gene2inter[ids])
            data.append((a,b))


allgenes = list(gene2pos.keys())
np.random.shuffle(allgenes)
#sampled = allgenes[:7]

random_data = []
count = 0
for ids in allgenes:
    if ids in normal_inner_gene2inter and ids in exp_inner_gene2inter:
        if ids in normal_intra_gene2inter and ids in exp_intra_gene2inter:
            count +=1
            #random_data.append((normal_inner_gene2inter[ids],exp_inner_gene2inter[ids]))
            a = normal_inner_gene2inter[ids]/(normal_inner_gene2inter[ids]+normal_intra_gene2inter[ids])
            b = exp_inner_gene2inter[ids]/(exp_inner_gene2inter[ids]+exp_intra_gene2inter[ids])
            random_data.append((a,b))
    if count==len(data):
        break

random_data = np.array(random_data)
data = np.array(data)

dic = {'':[],'interaction ratio':[],'type':[]}

for d in data:
    dic[''].append('highly differentiated')
    dic['interaction ratio'].append(d[0])
    dic['type'].append('normal cell')


    dic[''].append('highly differentiated')
    dic['interaction ratio'].append(d[1])
    dic['type'].append('abnormal cell')


for d in random_data:
    dic[''].append('nonhighly differentiated')
    dic['interaction ratio'].append(d[0])
    dic['type'].append('normal cell')


    dic[''].append('nonhighly differentiated')
    dic['interaction ratio'].append(d[1])
    dic['type'].append('abnormal cell')


import pandas as pd
df = pd.DataFrame(dic)

ax = sns.boxplot(x="type", y="interaction ratio", hue="",
                 data=df, palette="Set3")

plt.legend(loc='upper right')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('interaction ratio',fontsize=13)
plt.savefig('result/gene_expression_ratio.png')
plt.savefig('result/gene_expression_ratio.pdf')