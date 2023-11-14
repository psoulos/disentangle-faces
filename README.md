# Disentangled deep generative models reveal coding principles of the human face processing network

This is the code accompanying the paper "Disentangled deep generative models reveal coding principles of the human face
processing network" by Paul Soulos and Leyla Isik.

## Data
The data is available on [OSF](https://osf.io/pcyrf/). Please unzip the archive, there is a README in the archive that
explains the data structure.

## Code

The matlab code requires [Fieldtrip](https://github.com/fieldtrip/fieldtrip/tree/master) for processing the fMRI data. 

### Encoding Performance by ROI (Figure 3)
Encoding performance values are generated by `correlate_betas.py` and `correlate_betas_vgg.py` using the argument
`--localizer=roi` . The results can be visualized using `notebooks/data plots.ipynb`.

### Whole brain encoding performance (Figure 4)
Encoding performance values are generated by `correlate_betas.py` and `correlate_betas_vgg.py` using the argument
`--localizer=score` . The resulting correlation mat file can be converted to nifti using 
`convert_whole_brain_correlation_mat_to_nifti.m` and viewed using Freesurfer.


### ROI preference mapping (Figure 5)
The ROI preference map data is generated using `preference_mapping_roi.m`. The results can be visualized using 
`notebooks/encoding feature performance.ipynb`.

### Facial identity decoding (Figure 6)
The identity decoding accuracies are generated by `identity_decoding_whole_brain_xhat.m`. The results can be visualized 
using `notebooks/identity decoding.ipynb`.

### Whole brain encoding performance (Figure S1 and S2)
See the section titled "Whole brain encoding performance (Figure 4)".

### STS preference mapping (Figure S3)
The ROI preference map data is generated using `preference_mapping_roi.m`. The results can be visualized using 
`notebooks/encoding feature performance.ipynb`.

### Whole brain preference mapping (Figure S4)
The nifti files for the whole brain preference mapping are generated by `whole_brain_preference_mapping.m`. This nifti
file can be viewed using Freesurfer.

