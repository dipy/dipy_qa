#! /bin/bash

cd ../data/sherby/subj_1/

mkdir out_work

# Create brain mask
dipy_median_otsu dwi.nii.gz --out_dir out_work/

# Create stopping criteria
dipy_dti_metrics dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

# Create peaks
dipy_reconst_csd dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

cd out_work/

# Create tracks
dipy_det_track __dwi_dwi_dwi_brain_mask_peaks.npz __dwi_dwi_dwi_brain_mask_fa.nii.gz
