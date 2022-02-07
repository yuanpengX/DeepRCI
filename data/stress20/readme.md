# Dataset processing instructions

DeepRCI requires three types of data:

* Sequence data
* HiC data
* ATAC-seq data

*If you want to generate trainning data from the iMARGI data, you also need reference genome*
* GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
## Convert Normed Hi-C and ATAC-seq to mactrice

```
ipython
%run data_generation.ipynb
```

## remove redudant
```
python data2fa.py
bash cd-hit-est.sh
python fa2data.py
```
