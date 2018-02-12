import os
import numpy as np
from time import time
from dipy.data import (read_stanford_labels, default_sphere,
                       read_stanford_pve_maps)
from dipy.direction import ProbabilisticDirectionGetter
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.tracking.local import LocalTracking, ParticleFilteringTracking
from dipy.tracking import utils
from dipy.viz import window, actor, ui
from dipy.viz.colormap import line_colors
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.dti import TensorModel
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from dipy.tracking.local import CmcTissueClassifier
from dipy.tracking.streamline import Streamlines
import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from dipy.reconst.shm import sph_harm_lookup
from dipy.io.peaks import save_peaks


def show_odfs_and_fa(fa, pam, mask, affine, sphere, ftmp='odf.mmap',
                     basis_type=None, norm_odfs=True, scale_odfs=0.5):
    
    renderer = window.Renderer()
    renderer.background((1, 1, 1))
    
    slice_actor = actor.slicer(fa) #, value_range)
    
    odf_shape = fa.shape + (sphere.vertices.shape[0],)
    odfs = np.memmap(ftmp, dtype=np.float32, mode='w+',
                     shape=odf_shape)
    
    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m, n = sph_harm_basis(8, sphere.theta, sphere.phi)

    odfs[:] = np.dot(pam.shm_coeff.astype('f4'), B.T.astype('f4'))
    
    odf_slicer = actor.odf_slicer(odfs, mask=mask,
                                  sphere=sphere, scale=scale_odfs,
                                  norm=norm_odfs, colormap='magma')
        
    renderer.add(odf_slicer)
    renderer.add(slice_actor)
    
    show_m = window.ShowManager(renderer, size=(2000, 1000))
    show_m.initialize()
    
    """
    We'll start by creating the panel and adding it to the ``ShowManager``
    """
    
    label_position = ui.TextBlock2D(text='Position:')
    label_value = ui.TextBlock2D(text='Value:')
    
    result_position = ui.TextBlock2D(text='')
    result_value = ui.TextBlock2D(text='')
    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)
    
    def change_slice_z(i_ren, obj, slider):
        z = int(np.round(slider.value))
        slice_actor.display(z=z)
        odf_slicer.display(z=z)
        show_m.iren.force_render()
    
    line_slider_z.add_callback(line_slider_z.slider_disk,
                               "LeftButtonReleaseEvent",
                               change_slice_z)
    
    panel_picking = ui.Panel2D(center=(200, 120),
                               size=(250, 225),
                               color=(0, 0, 0),
                               opacity=0.75,
                               align="left")
    
    # panel_picking.add_element(label_position, 'relative', (0.1, 0.55))
    # panel_picking.add_element(label_value, 'relative', (0.1, 0.25))
    
    # panel_picking.add_element(result_position, 'relative', (0.45, 0.55))
    # panel_picking.add_element(result_value, 'relative', (0.45, 0.25))
    
    panel_picking.add_element(line_slider_z, 'relative', (0.5, 0.9))
    
    show_m.ren.add(panel_picking)
    
    def left_click_callback(obj, ev):
        """Get the value of the clicked voxel and show it in the panel."""
        event_pos = show_m.iren.GetEventPosition()
    
        obj.picker.Pick(event_pos[0],
                        event_pos[1],
                        0,
                        show_m.ren)
    
        i, j, k = obj.picker.GetPointIJK()
        print(i,j,k)
        result_position.message = '({}, {}, {})'.format(str(i), str(j), str(k))
        result_value.message = '%.3f' % fa[i, j, k]
    
    slice_actor.SetInterpolate(True)
    slice_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1.0)
        
    show_m.start()
    
    odfs._mmap.close()
    del odfs
    os.remove(ftmp)


def show_image(data, affine=None):
    renderer = window.Renderer()
    slicer = actor.slicer(data, affine)
    renderer.add(slicer)
    window.show(renderer)
    
    
def show_lines(streamlines, affine=None):
    renderer = window.Renderer()
    lines = actor.line(streamlines, affine)
    renderer.add(lines)
    window.show(renderer)
    
    
def show_peaks(pam):
    renderer = window.Renderer()
    peaks = actor.peak_slicer(pam.peak_dirs, colors=None)
    renderer.add(peaks)
    window.show(renderer)
    
    
def show_seeds(seeds):
    renderer = window.Renderer()
    points = actor.point(seeds, colors=np.random.rand(*seeds.shape))
    renderer.add(points)
    window.show(renderer)
    

def save_trk(fname, streamlines, affine, vox_size=None, shape=None, header=None):
    """ Saves tractogram files (*.trk)

    Parameters
    ----------
    fname : str
        output trk filename
    streamlines : list of 2D arrays or generator
        Each 2D array represents a sequence of 3D points (points, 3).
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
    vox_size : array_like (3,), optional
        The sizes of the voxels in the reference image (default: None)
    shape : array, shape (dim,), optional
        The shape of the reference image (default: None)
    header : dict, optional
        Metadata associated to the tractogram file(*.trk). (default: None)

    """
    if vox_size and shape:
        if not isinstance(header, dict):
            header = {}
        header[Field.VOXEL_TO_RASMM] = affine.copy()
        header[Field.VOXEL_SIZES] = vox_size
        header[Field.DIMENSIONS] = shape
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(affine))

    tractogram = nib.streamlines.Tractogram(streamlines)
    tractogram.affine_to_rasmm = affine
    trk_file = nib.streamlines.TrkFile(tractogram, header=header)
    nib.streamlines.save(trk_file, fname)


parallel = False # Be careful our current parallelization needs a lot of memory
seed_density = 1
step_size = 0.2

fbvals = 'bvals'
fbvecs = 'bvecs'
fbmask = '100307_brain_mask.nii.gz'
fdwi = '100307_dwi.nii.gz'
fpve = '100307_pve.nii.gz'

ffa = 'fa.nii.gz'
fdet = 'det.trk'
fpft = 'pft.trk'
fpam5 = 'peaks.pam5'

rec_model = 'GQI2'

sphere = get_sphere('repulsion724')

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

data, affine, vox_size = load_nifti(fdwi, return_voxsize=True)
mask, _ = load_nifti(fbmask)
pve, _, img, vox_size = load_nifti(
        fpve, return_img=True,
        return_voxsize=True)

shape = data.shape[:3]

if rec_model == 'GQI2':
    # Try with SFM, SHore, MAPL, MSMTCSD
    model = GeneralizedQSamplingModel(gtab,
                                      sampling_length=1.2)
if rec_model == 'SFM':
    # Ariel please add here your best SFM calls and parameters
    pass

if rec_model == 'MAPL':
    # Brent look at Aman's PR and call here
    pass
    
if rec_model == 'CSD':
    # Elef add CSD version and add MTMSCSD when is ready.
    pass

pam = peaks_from_model(model, data, sphere,
                       relative_peak_threshold=.8,
                       min_separation_angle=45,
                       mask=mask, parallel=parallel)




    
ten_model = TensorModel(gtab)
fa = ten_model.fit(data, mask).fa
save_nifti(ffa, fa, affine)                  

save_peaks(fpam5, pam,  affine)

show_odfs_and_fa(fa, pam, mask, None, sphere, ftmp='odf.mmap',
                 basis_type=None)
                  
pve_csf, pve_gm, pve_wm = pve[..., 0], pve[..., 1], pve[..., 2]

cmc_classifier = CmcTissueClassifier.from_pve(
        pve_wm,
        pve_gm,
        pve_csf,
        step_size=step_size,
        average_voxel_size=np.average(vox_size))

seed_mask = np.zeros(mask.shape)
seed_mask[mask > 0] = 1
seed_mask[pve_wm < 0.5] = 0
seeds = utils.seeds_from_mask(seed_mask,
                              density=1,
                              affine=affine)

det_streamline_generator = LocalTracking(pam,
                                         cmc_classifier,
                                         seeds,
                                         affine,
                                         step_size=step_size)

# The line below is failing not sure why
# detstreamlines = Streamlines(det_streamline_generator)

detstreamlines = list(det_streamline_generator)
detstreamlines = Streamlines(detstreamlines)
save_trk('det.trk', detstreamlines, affine=np.eye(4),
         vox_size=vox_size, shape=shape)

dg = ProbabilisticDirectionGetter.from_shcoeff(pam.shm_coeff,
                                               max_angle=20.,
                                               sphere=sphere)

# Particle Filtering Tractography
pft_streamline_generator = ParticleFilteringTracking(dg,
                                                     cmc_classifier,
                                                     seeds,
                                                     affine,
                                                     max_cross=1,
                                                     step_size=step_size,
                                                     maxlen=1000,
                                                     pft_back_tracking_dist=2,
                                                     pft_front_tracking_dist=1,
                                                     particle_count=15,
                                                     return_all=False)
# The line below is failing not sure why
# pftstreamlines = Streamlines(pft_streamline_generator)

pftstreamlines = list(pft_streamline_generator)
pftstreamlines = Streamlines(pftstreamlines)

save_trk('pft.trk', pftstreamlines, affine=np.eye(4),
         vox_size=vox_size, shape=shape)
#