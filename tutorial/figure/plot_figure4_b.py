import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl

start = 1
end = 6
ndf = pkl.load(open(f'result/inter_gene_expression_{start}_{end}.bin','rb'))

plt.figure(figsize=(13,9))
sns.boxplot(x=ndf['chromosome'],y=ndf['interaction'],hue= ndf['type'])
#sns.stripplot(x=ndf['type'],y=ndf['interaction'],hue = ndf['chromosome'], color='skyblue')
plt.legend(loc='upper right',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("")
plt.ylabel("interaction",fontsize=15)
plt.savefig(f'result/inter_expression_box_{start}_{end}.png')
plt.savefig(f'result/inter_expression_{start}_{end}.pdf')

# test
from scipy.stats import wilcoxon, ranksums

for i in range(start, end):
    chrs = ndf[ndf['chromosome']==f'chr{i}']
    high = chrs[chrs['type']=='highly expressed']
    low = chrs[chrs['type']=='low expressed']
    print(f'chr{i}: ', ranksums(high['interaction'], low['interaction']))