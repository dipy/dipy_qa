#! /bin/bash

cd ../data/sherby/Sherby/subj_1/

mkdir out_work

# Create brain mask
dipy_median_otsu dwi.nii.gz --out_dir out_work/

# Create stopping criteria
dipy_reconst_dti dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

# Create peaks
dipy_reconst_csd dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

# Create seeding mask (this creates seed_mask in new folder)
#dipy_mask out_work/fa.nii.gz 0.4 --out_dir out_work/ --out_mask seed_mask.nii.gz
# To avoid that we ommit the --out_dir option
dipy_mask out_work/fa.nii.gz 0.4 --out_mask seed_mask.nii.gz

cd out_work/

# Create tracks using peaks
dipy_det_track peaks.npz fa.nii.gz seed_mask.nii.gz --out_tractogram 'tracks_from_peaks.trk'

# Create tracks using sh cone
dipy_det_track peaks.npz fa.nii.gz seed_mask.nii.gz --out_tractogram 'tracks_from_sh.trk' --use_sh
