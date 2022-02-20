from matplotlib import pyplot as plt
import seaborn as sns
import pickle as pkl
import numpy as np
#new_x
ndf = pkl.load(open('result/inter_expression_normal.bin','rb'))
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)#
g=sns.relplot(x=ndf['expression'].values,y=ndf['interaction'].values)

plt.xlim(2,10.5)
plt.ylim(2,12.5)
plt.xlabel('expression',fontsize=13)
plt.ylabel('interaction',fontsize=13)

plt.savefig('result/inter_expression_normal_corr.pdf')
plt.savefig('result/inter_expression_normal_corr.png')

