#! /bin/bash

# uncomment if data directory not available
# mkdir ../data

cd ../data/sherby/Sherby/subj_1/

mkdir out_work

# Create brain mask
dipy_median_otsu dwi.nii.gz --out_dir out_work/

# Create stopping criteria
dipy_fit_dti dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

# Create peaks
dipy_fit_csd dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

# Create seeding mask (this creates seed_mask in new folder)
#dipy_mask out_work/fa.nii.gz 0.4 --out_dir out_work/ --out_mask seed_mask.nii.gz
# To avoid that we ommit the --out_dir option
dipy_mask out_work/fa.nii.gz 0.4 --out_mask seed_mask.nii.gz

cd out_work/

# Create tracks using peaks
dipy_track_det peaks.pam5 fa.nii.gz seed_mask.nii.gz --out_tractogram 'tracks_from_peaks.trk'

# Create tracks using sh cone
dipy_track_det peaks.pam5 fa.nii.gz seed_mask.nii.gz --out_tractogram 'tracks_from_sh.trk' --use_sh

# dipy_slr ~/.dipy/bundle_atlas_hcp842/Atlas_in_MNI_Space_16_bundles/whole_brain/whole_brain_MNI.trk csa_track.trk --mix_names
# dipy_recobundles whole_brain_MNI_csa_track__moved.trk  "/home/elef/.dipy/bundle_atlas_hcp842/Atlas_in_MNI_Space_16_bundles/bundles/*.trk" --mix_names --refine --out_dir rrbs

