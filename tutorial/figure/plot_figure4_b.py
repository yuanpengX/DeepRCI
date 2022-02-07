import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl

ndf = pkl.load(open('result/inter_expression_normal.bin','rb'))

sns.boxplot(x=ndf['chromosome'],y=ndf['interaction'],hue= ndf['type'])
#sns.stripplot(x=ndf['type'],y=ndf['interaction'],hue = ndf['chromosome'], color='skyblue')
plt.legend(loc='upper right')

plt.savefig('result/inter_expression_normal_box.pdf')
plt.savefig('result/inter_expression_normal_box.png')


plt.clf()

sns.set_theme(style="darkgrid")
sns.jointplot(x=ndf['expression'].values,y=ndf['interaction'].values,xlim=(2,9),kind='reg',color="m", )
plt.savefig('result/inter_expression_normal_corr.pdf')
plt.savefig('result/inter_expression_normal_corr.png')
