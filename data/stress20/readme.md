# Dataset processing instructions

DeepRCI requires three types of data:

* Sequence data
* HiC data
* ATAC-seq data

*If you want to generate training data from the iMARGI data, you also need a reference genome*
* GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
## Convert Normed Hi-C and ATAC-seq to matrice

```
ipython
%run data_generation.ipynb
```

## remove redundant
```
python data2fa.py
bash cd-hit-est.sh
python fa2data.py
```
