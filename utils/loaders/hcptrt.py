"""Data loading utilities for the HCP Test-Retest (hcptrt) fMRI dataset.

This module provides the HCPTRTLoader class for loading preprocessed fMRI
surface data (CIFTI dtseries), confound regressors, brain masks, and
parcellation data from the CNeuroMod hcptrt dataset.

Dataset structure assumed:
    <fmri_dir>/
        sub-<subject>/
            ses-<session>/
                func/
                    sub-<subject>_ses-<session>_task-<task>_run-<run>_space-fsLR_den-91k_bold.dtseries.nii
                    sub-<subject>_ses-<session>_task-<task>_run-<run>_desc-confounds_timeseries.tsv
                    sub-<subject>_ses-<session>_task-<task>_run-<run>_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
                    sub-<subject>_ses-<session>_task-<task>_run-<run>_space-T1w_desc-aparcaseg_dseg.nii.gz

Parcellation is handled via hcp_utils, which ships pre-built label arrays
for the fsLR 91k grayordinate space — no external atlas files needed.

Available parcellations (pass as `parcellation` argument):
    "mmp"        Glasser 360 — HCP multimodal parcellation, 180 regions/hemisphere
    "yeo7"       Yeo 7 large-scale functional networks
    "yeo17"      Yeo 17 functional networks
    "ca_parcels" Cole-Anticevic whole-brain parcellation (cortex + subcortex)
    "ca_network" Cole-Anticevic functional networks

Notes
-----
- Surface BOLD is stored as CIFTI dtseries with shape (T, 91282),
  where 91282 = ~59k cortical grayordinates + ~32k subcortical voxels.
- Confound denoising is applied inside load_fmri_responses() when
  denoise=True (the default). Call denoise_bold() directly for manual use.
- hcp_utils parcellation arrays are bundled with the package — no
  internet connection or file downloads are needed at runtime.
  Only your local dtseries.nii files are loaded from disk.

Install dependencies:
    pip install nibabel numpy pandas tqdm hcp_utils
"""

import os
import re
import glob
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import nibabel as nib
import hcp_utils as hcp

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_tsv(file_path: str) -> pd.DataFrame:
    """Read a TSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, sep="\t")


def _load_cifti_surface(path: str) -> Tuple[np.ndarray, nib.Cifti2Image]:
    """Load a CIFTI dtseries file.

    Parameters
    ----------
    path : str
        Path to ``*_bold.dtseries.nii`` file.

    Returns
    -------
    data : np.ndarray, shape (T, 91282)
        BOLD timeseries — T timepoints x 91282 grayordinates.
    img : nib.Cifti2Image
        The loaded CIFTI image (carries header / axis metadata).
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # (T, G)
    return data, img


# ---------------------------------------------------------------------------
# MMP area → lobe lookup
# ---------------------------------------------------------------------------
# Maps each Glasser MMP area name (without L_/R_ prefix) to a coarse lobe label.
# Based on the 22 cortical lobes in Glasser et al. 2016, collapsed to 8 broad lobes,
# plus subcortical structures.
MMP_LOBE: Dict[str, str] = {
    # Primary & early visual (occipital)
    "V1":"occipital","V2":"occipital","V3":"occipital","V4":"occipital",
    "V6":"occipital","V6A":"occipital","V7":"occipital","V8":"occipital",
    "V3A":"occipital","V3B":"occipital","V3CD":"occipital",
    "V4t":"occipital","LO1":"occipital","LO2":"occipital","LO3":"occipital",
    "VMV1":"occipital","VMV2":"occipital","VMV3":"occipital",
    "POS1":"occipital","POS2":"occipital","DVT":"occipital",
    # Dorsal visual / parietal
    "MST":"parietal","MT":"parietal","FST":"parietal","PH":"parietal",
    "V6A":"parietal","IPS1":"parietal","MIP":"parietal","VIP":"parietal",
    "LIPd":"parietal","LIPv":"parietal","AIP":"parietal",
    "IP0":"parietal","IP1":"parietal","IP2":"parietal",
    "7AL":"parietal","7Am":"parietal","7PC":"parietal","7PL":"parietal","7Pm":"parietal","7m":"parietal",
    "5L":"parietal","5m":"parietal","5mv":"parietal",
    "PCV":"parietal","PGi":"parietal","PGp":"parietal","PGs":"parietal",
    "PF":"parietal","PFm":"parietal","PFt":"parietal","PFop":"parietal","PFcm":"parietal",
    "TPOJ1":"parietal","TPOJ2":"parietal","TPOJ3":"parietal",
    "RSC":"parietal",
    # Somatomotor
    "1":"somatomotor","2":"somatomotor","3a":"somatomotor","3b":"somatomotor","4":"somatomotor",
    "6a":"somatomotor","6d":"somatomotor","6ma":"somatomotor","6mp":"somatomotor",
    "6r":"somatomotor","6v":"somatomotor",
    "FEF":"somatomotor","PEF":"somatomotor","55b":"somatomotor",
    "SCEF":"somatomotor","MI":"somatomotor",
    "43":"somatomotor","OP4":"somatomotor","OP1":"somatomotor","OP2-3":"somatomotor",
    # Insular / opercular
    "Ig":"insular","PI":"insular","PoI1":"insular","PoI2":"insular",
    "AAIC":"insular","AVI":"insular","FOP1":"insular","FOP2":"insular",
    "FOP3":"insular","FOP4":"insular","FOP5":"insular","RI":"insular",
    "Pir":"insular","52":"insular",
    # Auditory / temporal
    "A1":"temporal","A4":"temporal","A5":"temporal",
    "LBelt":"temporal","MBelt":"temporal","PBelt":"temporal",
    "PSL":"temporal","SFL":"temporal","STGa":"temporal",
    "STSda":"temporal","STSdp":"temporal","STSva":"temporal","STSvp":"temporal",
    "STV":"temporal","TA2":"temporal",
    "TE1a":"temporal","TE1m":"temporal","TE1p":"temporal",
    "TE2a":"temporal","TE2p":"temporal",
    "TF":"temporal","TGd":"temporal","TGv":"temporal",
    "PHA1":"temporal","PHA2":"temporal","PHA3":"temporal","PHT":"temporal",
    "FFC":"temporal","PIT":"temporal","VVC":"temporal",
    "H":"temporal","EC":"temporal","PeEc":"temporal","PreS":"temporal","ProS":"temporal",
    # Prefrontal
    "8Ad":"prefrontal","8Av":"prefrontal","8BL":"prefrontal","8BM":"prefrontal","8C":"prefrontal",
    "9a":"prefrontal","9m":"prefrontal","9p":"prefrontal",
    "9-46d":"prefrontal","a9-46v":"prefrontal","p9-46v":"prefrontal",
    "10d":"prefrontal","10pp":"prefrontal","10r":"prefrontal","10v":"prefrontal",
    "a10p":"prefrontal","p10p":"prefrontal",
    "46":"prefrontal","IFJa":"prefrontal","IFJp":"prefrontal","IFSa":"prefrontal","IFSp":"prefrontal",
    "44":"prefrontal","45":"prefrontal","47l":"prefrontal","47m":"prefrontal","47s":"prefrontal",
    "a47r":"prefrontal","p47r":"prefrontal","i6-8":"prefrontal","s6-8":"prefrontal",
    # Cingulate / medial
    "23c":"cingulate","23d":"cingulate","24dd":"cingulate","24dv":"cingulate",
    "a24":"cingulate","a24pr":"cingulate","p24":"cingulate","p24pr":"cingulate",
    "25":"cingulate","33pr":"cingulate","a32pr":"cingulate","p32":"cingulate",
    "p32pr":"cingulate","d23ab":"cingulate","v23ab":"cingulate",
    "d32":"cingulate","s32":"cingulate","31a":"cingulate","31pd":"cingulate","31pv":"cingulate",
    # Orbitofrontal
    "OFC":"orbitofrontal","11l":"orbitofrontal","13l":"orbitofrontal",
    "pOFC":"orbitofrontal",
    # Subcortical
    "accumbens_left":"subcortical","accumbens_right":"subcortical",
    "amygdala_left":"subcortical","amygdala_right":"subcortical",
    "brainStem":"subcortical",
    "caudate_left":"subcortical","caudate_right":"subcortical",
    "cerebellum_left":"subcortical","cerebellum_right":"subcortical",
    "diencephalon_left":"subcortical","diencephalon_right":"subcortical",
    "hippocampus_left":"subcortical","hippocampus_right":"subcortical",
    "pallidum_left":"subcortical","pallidum_right":"subcortical",
    "putamen_left":"subcortical","putamen_right":"subcortical",
    "thalamus_left":"subcortical","thalamus_right":"subcortical",
}

# ---------------------------------------------------------------------------
# Available hcp_utils parcellations
# ---------------------------------------------------------------------------

# Maps user-facing string names to the corresponding hcp_utils parcel objects.
# Each object exposes:
#   .map_all         — integer label array, shape (91282,)
#   .labels          — list of region name strings
#   .nontrivial_ids  — array of non-background parcel IDs
HCP_PARCELLATIONS: Dict[str, Any] = {
    "mmp":        hcp.mmp,         # Glasser 360 multimodal parcellation
    "yeo7":       hcp.yeo7,        # Yeo 7 functional networks
    "yeo17":      hcp.yeo17,       # Yeo 17 functional networks
    "ca_parcels": hcp.ca_parcels,  # Cole-Anticevic whole-brain parcels
    "ca_network": hcp.ca_network,  # Cole-Anticevic functional networks
}

# Default confound columns for 24-parameter + WM/CSF denoising
DEFAULT_CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x",   "rot_y",   "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1",   "rot_y_derivative1",   "rot_z_derivative1",
    "white_matter", "csf",
    "framewise_displacement",
]


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class HCPTRTLoader:
    """Data loader for the CNeuroMod HCP Test-Retest (hcptrt) dataset.

    Provides methods for loading surface BOLD timeseries, confound
    regressors, brain masks, segmentation files, and parcellated
    region timeseries — mirroring the interface of AlgonautsLoader.

    Parcellation is handled via hcp_utils, which ships label arrays
    for the fsLR 91k grayordinate space. No external atlas files are
    needed — only your local dtseries.nii files are loaded from disk.

    Parameters
    ----------
    fmri_dir : str
        Root directory of the preprocessed hcptrt dataset, i.e. the
        path that contains ``sub-01/``, ``sub-03/``, etc.
        Example: ``"data/cneuromod.processed/fmriprep/hcptrt"``
    subjects : list of str, optional
        Subject IDs to operate on. Defaults to the three open-access
        subjects ``["sub-01", "sub-03", "sub-05"]``.
    tr : float, optional
        Repetition time in seconds. Default is 1.49 (hcptrt acquisition).
    hrf_delay : int, optional
        Number of TRs to shift for HRF peak. Default is 5.
    parcellation : str, optional
        Which parcellation to use for region averaging. One of:
        ``"mmp"`` (Glasser 360, default), ``"yeo7"``, ``"yeo17"``,
        ``"ca_parcels"``, ``"ca_network"``.

    Examples
    --------
    >>> loader = HCPTRTLoader(
    ...     fmri_dir="data/cneuromod.processed/fmriprep/hcptrt",
    ...     parcellation="mmp"   # Glasser 360, no file download needed
    ... )
    >>> # Single run — denoised surface BOLD, shape (T, 91282)
    >>> bold = loader.load_fmri_responses("sub-01", "ses-001", "motor", run=1)
    >>> # Single run — parcellated to Glasser regions, shape (T, 360)
    >>> bold_parc = loader.load_fmri_responses(
    ...     "sub-01", "ses-001", "motor", run=1, parcellate=True
    ... )
    """

    # Tasks available in hcptrt
    TASKS = [
        "emotion", "gambling", "language",
        "motor", "relational", "social", "wm"
    ]

    def __init__(
        self,
        fmri_dir: str,
        subjects: Optional[List[str]] = None,
        tr: float = 1.49,
        hrf_delay: int = 5,
        parcellation: str = "mmp",
    ):
        self.fmri_dir = fmri_dir
        self.subjects = subjects or ["sub-01", "sub-03", "sub-05"]
        self.tr = tr
        self.hrf_delay = hrf_delay

        self._init_parcellation(parcellation)

    # ------------------------------------------------------------------
    # Parcellation setup
    # ------------------------------------------------------------------

    def _init_parcellation(self, parcellation: str):
        """Set the active parcellation from hcp_utils.

        Parameters
        ----------
        parcellation : str
            One of the keys in ``HCP_PARCELLATIONS``.

        Raises
        ------
        ValueError
            If the parcellation name is not recognised.
        """
        if parcellation not in HCP_PARCELLATIONS:
            raise ValueError(
                f"Unknown parcellation '{parcellation}'. "
                f"Choose from: {list(HCP_PARCELLATIONS.keys())}"
            )
        self.parcellation_name = parcellation
        self._parcel_obj = HCP_PARCELLATIONS[parcellation]

        # Integer label array, shape (91282,) — one label per grayordinate.
        # Label 0 = background / unlabelled grayordinates.
        self._parcel_labels: np.ndarray = self._parcel_obj.map_all

        # Region name strings keyed by integer parcel ID.
        # hcp_utils exposes .labels as dict {int_id: str_name}, so we store it
        # directly rather than converting to a list (which would give keys, not names).
        raw_labels = self._parcel_obj.labels
        if isinstance(raw_labels, dict):
            self._parcel_names: Dict[int, str] = {int(k): str(v) for k, v in raw_labels.items()}
        else:
            # Fallback: plain list — build a 0-indexed dict
            self._parcel_names: Dict[int, str] = {i: str(v) for i, v in enumerate(raw_labels)}

        n = len(self._parcel_obj.nontrivial_ids)
        print(f"Parcellation : {parcellation} — {n} regions (hcp_utils, no download needed)")

    def set_parcellation(self, parcellation: str):
        """Switch to a different parcellation at any time.

        Parameters
        ----------
        parcellation : str
            One of ``"mmp"``, ``"yeo7"``, ``"yeo17"``,
            ``"ca_parcels"``, ``"ca_network"``.

        Examples
        --------
        >>> loader.set_parcellation("yeo7")
        """
        self._init_parcellation(parcellation)

    def get_parcel_names(self) -> List[str]:
        """Return region names for the active parcellation.

        Returns
        -------
        list of str
            One name per non-background parcel, in parcel-ID order.
        """
        return [
            self._parcel_names[int(pid)]
            for pid in self._parcel_obj.nontrivial_ids
        ]

    def get_atlas_info(self) -> Dict[str, Any]:
        """Return structured metadata for all parcels in the active parcellation.

        Analogous to ``AlgonautsLoader.load_atlas_for_subject()`` — provides
        parcel names, hemisphere, region labels, and the mean MNI (x, y, z)
        centroid of each parcel's grayordinates.

        Unlike AlgonautsLoader which loads a per-subject NIfTI atlas file,
        here coordinates come from hcp_utils which stores the MNI position of
        every grayordinate in the 91k fsLR surface space. No subject-specific
        file is needed — the surface geometry is the same across subjects.

        Returns
        -------
        dict with keys:
            ``"parcel_ids"``    : np.ndarray (n_parcels,)
            ``"parcel_names"``  : list of str (n_parcels,)
            ``"parcel_coords"`` : np.ndarray (n_parcels, 3) — MNI centroid per parcel
            ``"parcel_desc"``   : dict mapping int parcel_id →
                                    {
                                      "name"       : str,
                                      "hemisphere" : str  ("LH", "RH", or "unknown"),
                                      "region"     : str,
                                      "coords"     : np.ndarray (3,) MNI centroid
                                    }
        """
        parcel_ids   = self._parcel_obj.nontrivial_ids   # (n_parcels,)
        parcel_names = self.get_parcel_names()

        # Build a (91282, 3) coordinate array from the midthickness surface meshes.
        # hcp.struct entries are slice objects (indices), not coordinate arrays.
        # Cortical grayordinates (0..59411): positions come from the midthickness
        # vertex coordinates, filtered to the grayordinate vertices via grayl/grayr.
        # Subcortical grayordinates (59412..91281): no surface mesh in hcp_utils,
        # so we fill with NaN and use nanmean for centroids.
        all_coords = np.full((91282, 3), np.nan, dtype=np.float32)

        # Left cortex: grayl gives the grayordinate row indices in the 91282 array
        # AND the vertex indices into the midthickness mesh simultaneously, because
        # hcp_utils aligns them: grayl[i] is both the 91k index and the vertex index.
        verts_l = hcp.mesh.midthickness_left[0]    # (n_verts_L, 3)
        idx_l   = hcp.vertex_info.grayl             # 1-D integer array
        all_coords[idx_l] = verts_l[idx_l]

        # Right cortex (grayr indexes into the right-hemisphere mesh)
        verts_r = hcp.mesh.midthickness_right[0]   # (n_verts_R, 3)
        idx_r   = hcp.vertex_info.grayr             # 1-D integer array
        # grayr values are 91k-space indices; right-mesh vertex index = idx_r - len(idx_l)
        # but hcp_utils stores them 0-based relative to each hemisphere's mesh:
        all_coords[idx_r] = verts_r[idx_r - idx_r[0]]

        labels = self._parcel_labels                 # (91282,)

        parcel_coords = np.zeros((len(parcel_ids), 3), dtype=np.float32)
        parcel_desc   = {}

        for i, pid in enumerate(parcel_ids):
            mask     = labels == pid
            centroid = np.nanmean(all_coords[mask], axis=0)   # NaN-safe for subcortical
            if np.any(np.isnan(centroid)):
                centroid = np.zeros(3, dtype=np.float32)
            parcel_coords[i] = centroid

            name  = parcel_names[i]
            parts = name.split("_")

            # Parse hemisphere and region name from label string.
            # 1. Glasser/MMP style:  "L_V1_ROI"  or  "R_46_ROI"
            if parts[0] in ("L", "R"):
                hemi     = "LH" if parts[0] == "L" else "RH"
                # area_key for lobe lookup: strip "ROI" if it's there
                area_key = "_".join(parts[1:-1]) if parts[-1] == "ROI" else "_".join(parts[1:])
            
            # 2. Yeo/Schaefer style: "7Networks_LH_Vis_1"
            elif len(parts) > 1 and parts[1] in ("LH", "RH"):
                hemi     = parts[1]
                area_key = "_".join(parts[2:])
            
            # 3. Subcortical / Other: "Hippocampus_Left" or "thalamus-right"
            else:
                name_lower = name.lower()
                if "left" in name_lower:
                    hemi   = "LH"
                    # For subcortical, area_key stays as name to match MMP_LOBE dict
                    area_key = name
                elif "right" in name_lower:
                    hemi   = "RH"
                    area_key = name
                else:
                    hemi   = "unknown"
                    area_key = name

            # Map area → lobe using MMP_LOBE; fall back to area_key if unknown
            region = MMP_LOBE.get(area_key, area_key)
            
            # If the lookup worked, we have a clean lobe name. 
            # If not, and it was a subcortical with "left"/"right", 
            # we might want to clean the region name for display.
            if region == area_key:
                # Cleanup "left"/"right" from display name if it wasn't in the dict
                region = re.sub(r'[_-]?left[_-]?', '', region, flags=re.IGNORECASE)
                region = re.sub(r'[_-]?right[_-]?', '', region, flags=re.IGNORECASE)
                region = region.strip("-_ ")

            parcel_desc[int(pid)] = {
                "name":       name,
                "hemisphere": hemi,
                "region":     region,
                "coords":     centroid,
            }

        return {
            "parcel_ids":    parcel_ids,
            "parcel_names":  parcel_names,
            "parcel_coords": parcel_coords,
            "parcel_desc":   parcel_desc,
        }

    def get_parcel_label_from_coords(
        self, coords: np.ndarray, tolerance_mm: float = 10.0
    ) -> Optional[str]:
        """Return the parcel name nearest to a given MNI coordinate.

        Analogous to ``AlgonautsLoader.get_parcel_label()`` — finds the
        parcel centroid closest to the provided MNI (x, y, z) point.

        Parameters
        ----------
        coords : np.ndarray, shape (3,)
            MNI (x, y, z) coordinate in mm.
        tolerance_mm : float
            Maximum allowable distance to the nearest centroid. Returns
            None if no centroid is within this range. Default 10 mm.

        Returns
        -------
        str or None
        """
        atlas_info = self.get_atlas_info()
        centroids  = atlas_info["parcel_coords"]   # (n_parcels, 3)
        names      = atlas_info["parcel_names"]

        dists   = np.linalg.norm(centroids - np.asarray(coords), axis=1)
        nearest = int(np.argmin(dists))

        if dists[nearest] > tolerance_mm:
            return None

        return names[nearest]

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_func_dir(self, subject: str, session: str) -> str:
        """Return the func/ directory for a given subject and session."""
        return os.path.join(self.fmri_dir, subject, session, "func")

    def get_bold_path(
        self, subject: str, session: str, task: str, run: int
    ) -> str:
        """Build path to the surface BOLD dtseries file for one run."""
        fname = (
            f"{subject}_{session}_task-{task}_run-{run}"
            f"_space-fsLR_den-91k_bold.dtseries.nii"
        )
        return os.path.join(self.get_func_dir(subject, session), fname)

    def get_confounds_path(
        self, subject: str, session: str, task: str, run: int
    ) -> str:
        """Build path to the music confounds TSV file for one run."""
        fname = (
            f"{subject}_{session}_task-{task}_run-{run}"
            f"_desc-confounds_timeseries.tsv"
        )
        return os.path.join(self.get_func_dir(subject, session), fname)

    def get_brain_mask_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: int,
        space: str = "MNI152NLin2009cAsym",
    ) -> str:
        """Build path to the brain mask NIfTI file for one run.

        Parameters
        ----------
        space : str
            ``"MNI152NLin2009cAsym"`` (default) or ``"T1w"``.
        """
        fname = (
            f"{subject}_{session}_task-{task}_run-{run:02d}"
            f"_space-{space}_desc-brain_mask.nii.gz"
        )
        return os.path.join(self.get_func_dir(subject, session), fname)

    def get_segmentation_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: int,
        space: str = "T1w",
        seg_type: str = "aparcaseg",
    ) -> str:
        """Build path to a FreeSurfer segmentation (dseg) file for one run.

        Parameters
        ----------
        seg_type : str
            ``"aparcaseg"`` (cortical + subcortical) or ``"aseg"``
            (subcortical only).
        """
        fname = (
            f"{subject}_{session}_task-{task}_run-{run:02d}"
            f"_space-{space}_desc-{seg_type}_dseg.nii.gz"
        )
        return os.path.join(self.get_func_dir(subject, session), fname)

    def get_events_path(
        self, subject: str, session: str, task: str, run: int
    ) -> str:
        """Build path to the events TSV file for one run.

        Events files live in the sourcedata reference inside the
        preprocessed dataset, not alongside the BOLD files. Download
        them first with:
            datalad get sourcedata/hcptrt/sub-*/ses-*/func/*_events.tsv

        Returns path to:
            ``<fmri_dir>/sourcedata/hcptrt/<subject>/<session>/func/*_events.tsv``
        """
        fname = f"{subject}_{session}_task-{task}_run-{run:02d}_events.tsv"
        sourcedata_dir = os.path.join(
            self.fmri_dir, "sourcedata", "hcptrt",
            subject, session, "func"
        )
        return os.path.join(sourcedata_dir, fname)

    # ------------------------------------------------------------------
    # Session / run discovery
    # ------------------------------------------------------------------

    def list_sessions(self, subject: str) -> List[str]:
        """List available session IDs for a subject on disk.

        Returns
        -------
        list of str
            Sorted session IDs, e.g. ``["ses-001", "ses-002", ...]``.
        """
        subject_dir = os.path.join(self.fmri_dir, subject)
        sessions = sorted([
            d for d in os.listdir(subject_dir)
            if d.startswith("ses-")
            and os.path.isdir(os.path.join(subject_dir, d))
        ])
        return sessions

    def list_runs(
        self, subject: str, session: str, task: str
    ) -> List[int]:
        """List available run numbers for a subject / session / task.

        Returns
        -------
        list of int
            Sorted run numbers found on disk.
        """
        func_dir = self.get_func_dir(subject, session)
        pattern = os.path.join(
            func_dir,
            f"{subject}_{session}_task-{task}_run-*"
            f"_space-fsLR_den-91k_bold.dtseries.nii"
        )
        files = sorted(glob.glob(pattern))
        runs = []
        for f in files:
            base = os.path.basename(f)
            run_part = [p for p in base.split("_") if p.startswith("run-")][0]
            runs.append(int(run_part.replace("run-", "")))
        return runs

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------

    def load_bold(
        self, subject: str, session: str, task: str, run: int
    ) -> Tuple[np.ndarray, nib.Cifti2Image]:
        """Load raw surface BOLD timeseries for one run.

        Returns
        -------
        data : np.ndarray, shape (T, 91282)
            Raw (not denoised) BOLD surface timeseries.
        img : nib.Cifti2Image
            CIFTI image carrying header metadata.
        """
        path = self.get_bold_path(subject, session, task, run)
        if not os.path.exists(path):
            raise FileNotFoundError(f"BOLD file not found: {path}")
        return _load_cifti_surface(path)

    def load_confounds(
        self,
        subject: str,
        session: str,
        task: str,
        run: int,
        columns: Optional[List[str]] = None,
        fill_na: bool = True,
    ) -> pd.DataFrame:
        """Load confound regressors for one run.

        Parameters
        ----------
        columns : list of str, optional
            Subset of confound columns to return. Defaults to
            ``DEFAULT_CONFOUND_COLS``.
        fill_na : bool
            Forward-fill then zero-fill NaN values (derivative columns
            have NaN at the first timepoint). Default True.

        Returns
        -------
        pd.DataFrame, shape (T, n_confounds)
        """
        path = self.get_confounds_path(subject, session, task, run)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Confounds file not found: {path}")

        df = _read_tsv(path)

        if columns is None:
            columns = [c for c in DEFAULT_CONFOUND_COLS if c in df.columns]

        df = df[columns]

        if fill_na:
            df = df.ffill().fillna(0.0)

        return df

    def load_brain_mask(
        self,
        subject: str,
        session: str,
        task: str,
        run: int,
        space: str = "MNI152NLin2009cAsym",
    ) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Load the binary brain mask for one run.

        Returns
        -------
        mask : np.ndarray, shape (X, Y, Z), dtype bool
        img : nib.Nifti1Image
        """
        path = self.get_brain_mask_path(subject, session, task, run, space)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Brain mask not found: {path}")
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32).astype(bool), img

    def load_segmentation(
        self,
        subject: str,
        session: str,
        task: str,
        run: int,
        space: str = "T1w",
        seg_type: str = "aparcaseg",
    ) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Load the FreeSurfer segmentation (dseg) for one run.

        Returns
        -------
        seg : np.ndarray, shape (X, Y, Z), dtype int
        img : nib.Nifti1Image
        """
        path = self.get_segmentation_path(subject, session, task, run, space, seg_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Segmentation file not found: {path}")
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32).astype(int), img

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def denoise_bold(
        self,
        bold_data: np.ndarray,
        confounds_df: pd.DataFrame,
    ) -> np.ndarray:
        """Denoise BOLD by regressing out confound signals (OLS).

        Adds an intercept column automatically so the temporal mean is
        also removed (equivalent to mean-centering each grayordinate).

        Parameters
        ----------
        bold_data : np.ndarray, shape (T, G)
        confounds_df : pd.DataFrame, shape (T, n_confounds)

        Returns
        -------
        np.ndarray, shape (T, G) — residual BOLD after confound regression.
        """
        T = bold_data.shape[0]
        X = np.hstack([np.ones((T, 1)), confounds_df.values])  # (T, n+1)
        beta, _, _, _ = np.linalg.lstsq(X, bold_data, rcond=None)
        return (bold_data - X @ beta).astype(np.float32)

    # ------------------------------------------------------------------
    # Parcellation
    # ------------------------------------------------------------------

    def parcellate(
        self,
        bold_data: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Average surface BOLD into parcels using hcp_utils.

        Parameters
        ----------
        bold_data : np.ndarray, shape (T, 91282)

        Returns
        -------
        parcel_ts : np.ndarray, shape (T, n_parcels)
        parcel_names : list of str, length n_parcels
        """
        parcel_ts = hcp.parcellate(bold_data, self._parcel_obj)
        return parcel_ts.astype(np.float32), self.get_parcel_names()

    def get_cortex_only(self, bold_data: np.ndarray) -> np.ndarray:
        """Extract cortex-only grayordinates (~59k vertices), dropping subcortex.

        Parameters
        ----------
        bold_data : np.ndarray, shape (T, 91282)

        Returns
        -------
        np.ndarray, shape (T, ~59412)
        """
        return hcp.cortex_data(bold_data)

    # ------------------------------------------------------------------
    # Events loading
    # ------------------------------------------------------------------

    def load_epochs_info(
        self,
        subject: str,
        session: str,
        task: str,
        run: int,
        bloc_only: bool = True,
    ) -> pd.DataFrame:
        """Load the events TSV for one run.

        Analogous to ``AlgonautsLoader.load_transcript()`` — provides the
        timing information needed to epoch the BOLD timeseries by condition.

        The events file contains several trial_type values:
          - ``new_bloc_*``   : onset of a task block (e.g. new_bloc_left_hand)
          - ``response_*``   : individual button presses within a block
          - ``cross_fixation``: rest fixation periods
          - ``countdown``    : pre-task countdown

        Parameters
        ----------
        subject, session, task, run : str / int
        bloc_only : bool
            If True (default), return only ``new_bloc_*`` rows — these are
            the block onsets used for epoching. Set to False to get the
            full events table including individual button presses.

        Returns
        -------
        pd.DataFrame
            Columns: ``trial_type``, ``onset``, ``duration``, plus any
            task-specific columns (e.g. ``nbloc``, ``countdown_stim``).

        Raises
        ------
        FileNotFoundError
            If the events file has not been downloaded yet. Run:
            ``datalad get sourcedata/hcptrt/sub-*/ses-*/func/*_events.tsv``
            from inside your cneuromod.processed/fmriprep/hcptrt directory.
        """
        path = self.get_events_path(subject, session, task, run)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Events file not found: {path}\n"
                f"Download with: datalad get sourcedata/hcptrt/"
                f"{subject}/{session}/func/*_events.tsv"
            )

        df = _read_tsv(path)

        if bloc_only:
            df = df[df["trial_type"].str.startswith("new_bloc_")].reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Epoching
    # ------------------------------------------------------------------

    def epoch_bold_by_blocks(
        self,
        bold_data: np.ndarray,
        events_df: pd.DataFrame,
        trial_types: Optional[List[str]] = None,
        window_trs: int = 10,
        hrf_delay: Optional[int] = None,
        tr: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract BOLD windows anchored to task block onsets.

        Analogous to ``AlgonautsLoader.epoch_fmri_by_words()`` — takes a
        denoised BOLD timeseries and the events dataframe, and returns a
        set of fixed-length windows centred on each block onset (shifted
        by HRF delay).

        Parameters
        ----------
        bold_data : np.ndarray, shape (T, G) or (T, n_parcels)
            Denoised BOLD timeseries for one run.
        events_df : pd.DataFrame
            Output of ``load_epochs_info(bloc_only=True)``. Must contain
            ``onset`` (seconds) and ``trial_type`` columns.
        trial_types : list of str, optional
            Subset of ``trial_type`` values to include, e.g.
            ``["new_bloc_left_hand", "new_bloc_tongue"]``.
            If None, all ``new_bloc_*`` rows are used.
        window_trs : int
            Number of TRs to extract after the (HRF-shifted) block onset.
            Default 10 (~15 seconds at TR=1.49s), covering a typical
            ~12s HCP task block plus a couple of TRs of tail.
        hrf_delay : int, optional
            TR shift to account for haemodynamic lag. Defaults to
            ``self.hrf_delay`` (5 TRs).
        tr : float, optional
            Repetition time in seconds. Defaults to ``self.tr`` (1.49s).

        Returns
        -------
        epochs : np.ndarray, shape (n_blocks, window_trs, G_or_parcels)
            One epoch per block onset that fits within the run.
        onsets_tr : np.ndarray, shape (n_blocks,)
            TR index of each epoch start (after HRF shift).
        labels : np.ndarray of str, shape (n_blocks,)
            Condition label for each epoch (trial_type with
            ``new_bloc_`` prefix stripped, e.g. ``"left_hand"``).

        Returns ``(None, None, None)`` if no valid epochs were found.
        """
        if hrf_delay is None:
            hrf_delay = self.hrf_delay
        if tr is None:
            tr = self.tr

        T = bold_data.shape[0]

        # Filter to requested trial types
        df = events_df.copy()
        if trial_types is not None:
            df = df[df["trial_type"].isin(trial_types)]
        # Keep only new_bloc rows (guard against caller passing full events)
        df = df[df["trial_type"].str.startswith("new_bloc_")].reset_index(drop=True)

        if df.empty:
            return None, None, None

        epochs, onsets_tr, labels = [], [], []

        for _, row in df.iterrows():
            # Convert onset in seconds → TR index, then shift by HRF delay
            onset_tr = int(row["onset"] / tr) + hrf_delay
            end_tr   = onset_tr + window_trs

            # Skip if window falls outside the run
            if onset_tr < 0 or end_tr > T:
                continue

            epochs.append(bold_data[onset_tr:end_tr])        # (window_trs, G)
            onsets_tr.append(onset_tr)
            # Strip "new_bloc_" prefix to get clean condition name
            labels.append(row["trial_type"].replace("new_bloc_", ""))

        if not epochs:
            return None, None, None

        return (
            np.stack(epochs, axis=0).astype(np.float32),     # (n_blocks, window_trs, G)
            np.array(onsets_tr),
            np.array(labels),
        )

    # ------------------------------------------------------------------
    # High-level loaders (mirroring AlgonautsLoader interface)
    # ------------------------------------------------------------------

    def load_fmri_responses(
        self,
        subject: str,
        session: str,
        task: str,
        run: int,
        denoise: bool = True,
        parcellate: bool = False,
        confound_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Load (and optionally denoise / parcellate) BOLD for one run.

        Primary single-run entry point, analogous to
        ``AlgonautsLoader.load_fmri_responses()``.

        Parameters
        ----------
        denoise : bool
            Regress out confounds. Default True.
        parcellate : bool
            Average into parcels via hcp_utils. Default False.
        confound_columns : list of str, optional
            Columns to use for denoising.

        Returns
        -------
        np.ndarray
            Shape ``(T, 91282)`` or ``(T, n_parcels)`` if parcellated.
        """
        bold, _ = self.load_bold(subject, session, task, run)

        if denoise:
            confounds = self.load_confounds(
                subject, session, task, run, columns=confound_columns
            )
            bold = self.denoise_bold(bold, confounds.iloc[:bold.shape[0]])

        if parcellate:
            bold, _ = self.parcellate(bold)

        return bold

    def load_task_fmri(
        self,
        subject: str,
        task: str,
        session: Optional[str] = None,
        run: Optional[int] = None,
        denoise: bool = True,
        parcellate: bool = False,
        confound_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load continuous BOLD timeseries for all sessions/runs of one task.

        Mirrors ``AlgonautsLoader.load_episode_fmri()`` exactly — returns
        the same dict structure so existing scripts that consume
        ``load_episode_fmri`` output can consume this output unchanged,
        substituting run keys for episode keys.

        Parameters
        ----------
        subject : str
            e.g. ``"sub-01"``
        task : str
            One of ``HCPTRTLoader.TASKS``.
        session : str, optional
            Specific session to load (e.g. ``"ses-001"``). If None, loads
            all available sessions for the subject.
        run : int, optional
            Specific run number to load (e.g. ``1``). If None, loads
            all available runs for the specified session(s).
        denoise : bool
            Regress out confounds before returning. Default True.
        parcellate : bool
            Average grayordinates into parcels via hcp_utils. Default False.
        confound_columns : list of str, optional
            Confound columns for denoising.

        Returns
        -------
        dict with keys — identical layout to ``load_episode_fmri()``:
            ``"scenes_response"`` : dict mapping ``"ses-XXX_run-YY"`` →
                                    np.ndarray (T, G_or_parcels)
            ``"parcel_coords"``   : np.ndarray (n_parcels, 3) MNI centroids
            ``"parcel_ids"``      : np.ndarray (n_parcels,)
            ``"parcel_desc"``     : dict mapping int parcel_id →
                                      {"name", "hemisphere", "region", "coords"}
        """
        if session is not None:
            sessions = [session]
        else:
            sessions = self.list_sessions(subject)
        scenes_response = {}

        print(f"Loading hcptrt BOLD | {subject} | task-{task}")
        for sess in tqdm(sessions):
            if run is not None:
                runs = [run]
            else:
                runs = self.list_runs(subject, sess, task)
            for r in runs:
                key = f"{sess}_run-{r:02d}"
                try:
                    scenes_response[key] = self.load_fmri_responses(
                        subject, sess, task, r,
                        denoise=denoise,
                        parcellate=parcellate,
                        confound_columns=confound_columns,
                    )
                except FileNotFoundError as e:
                    print(f"  Skipping {key}: {e}")

        print(f"Loading parcel atlas info")
        atlas = self.get_atlas_info()

        return {
            "scenes_response": scenes_response,
            "parcel_coords":   atlas["parcel_coords"],
            "parcel_ids":      atlas["parcel_ids"],
            "parcel_desc":     atlas["parcel_desc"],
        }

    def load_all_tasks_fmri(
        self,
        subject: str,
        session: Optional[str] = None,
        run: Optional[int] = None,
        denoise: bool = True,
        parcellate: bool = False,
        confound_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load continuous BOLD for all 7 hcptrt tasks for one subject.

        Parameters
        ----------
        subject : str
        session : str, optional
            Specific session to load (e.g. ``"ses-001"``). If None, loads
            all available sessions for the subject.
        run : int, optional
            Specific run number to load (e.g. ``1``). If None, loads
            all available runs for the specified session(s).
        denoise, parcellate, confound_columns
            Passed through to ``load_task_fmri()``.

        Returns
        -------
        dict mapping task name → output of ``load_task_fmri()``
        """
        return {
            task: result
            for task in self.TASKS
            for result in [self.load_task_fmri(
                subject, task,
                session=session,
                run=run,
                denoise=denoise,
                parcellate=parcellate,
                confound_columns=confound_columns,
            )]
            if result["scenes_response"]
        }

    def load_task_epochs(
        self,
        subject: str,
        task: str,
        session: Optional[str] = None,
        run: Optional[int] = None,
        trial_types: Optional[List[str]] = None,
        window_trs: int = 10,
        denoise: bool = True,
        parcellate: bool = False,
        confound_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load block-epoched BOLD for all sessions/runs of one task.

        Mirrors ``AlgonautsLoader.load_episode_word_epochs()`` exactly —
        same dict structure so existing scripts consuming word-epoch output
        can consume this output with minimal changes, substituting block
        epochs for word epochs and run keys for episode keys.

        Combines ``load_fmri_responses()``, ``load_epochs_info()``, and
        ``epoch_bold_by_blocks()`` in a single call across all sessions
        and runs on disk for the given subject and task.

        Parameters
        ----------
        subject : str
            e.g. ``"sub-01"``
        task : str
            One of ``HCPTRTLoader.TASKS``.
        session : str, optional
            Specific session to load (e.g. ``"ses-001"``). If None, loads
            all available sessions for the subject.
        run : int, optional
            Specific run number to load (e.g. ``1``). If None, loads
            all available runs for the specified session(s).
        trial_types : list of str, optional
            ``new_bloc_*`` condition names to include, e.g.
            ``["new_bloc_left_hand", "new_bloc_tongue"]``.
            If None, all block conditions are included.
        window_trs : int
            Length of each epoch window in TRs. Default 10
            (~15s at TR=1.49s, covering a full ~12s HCP block).
        denoise, parcellate, confound_columns
            Passed through to ``load_fmri_responses()``.

        Returns
        -------
        dict with keys — identical layout to ``load_episode_word_epochs()``:
            ``"scenes_response"`` : dict mapping ``"ses-XXX_run-YY"`` →
                                    dict with keys:
                                      ``"trials"`` np.ndarray (n_blocks, window_trs, G)
                                      ``"start"``  np.ndarray (n_blocks,) onset TR indices
                                      ``"end"``    np.ndarray (n_blocks,) offset TR indices
                                      ``"labels"`` np.ndarray of str (n_blocks,) condition names
            ``"parcel_coords"``   : np.ndarray (n_parcels, 3) MNI centroids
            ``"parcel_ids"``      : np.ndarray (n_parcels,)
            ``"parcel_desc"``     : dict mapping int parcel_id →
                                      {"name", "hemisphere", "region", "coords"}

        Notes
        -----
        ``"labels"`` has no direct equivalent in ``load_episode_word_epochs()``
        (which uses word text instead). It is added here as an extra key inside
        each run dict so condition identity is always carried alongside the data.
        """
        if session is not None:
            sessions = [session]
        else:
            sessions = self.list_sessions(subject)
        scenes_response = {}

        print(f"Loading hcptrt block epochs | {subject} | task-{task}")
        for sess in tqdm(sessions):
            if run is not None:
                runs = [run]
            else:
                runs = self.list_runs(subject, sess, task)
            for r in runs:
                key = f"{sess}_run-{r:02d}"
                try:
                    bold = self.load_fmri_responses(
                        subject, sess, task, r,
                        denoise=denoise,
                        parcellate=parcellate,
                        confound_columns=confound_columns,
                    )
                    events = self.load_epochs_info(subject, sess, task, r)
                    trials, start, end = self.epoch_bold_by_blocks(
                        bold, events,
                        trial_types=trial_types,
                        window_trs=window_trs,
                    )
                    if trials is None:
                        print(f"  No valid epochs in {key}, skipping")
                        continue

                    # Recover per-epoch labels from events (same loop order
                    # as epoch_bold_by_blocks so indices align)
                    hrf_delay = self.hrf_delay
                    tr        = self.tr
                    df = events[events["trial_type"].str.startswith("new_bloc_")]
                    if trial_types is not None:
                        df = df[df["trial_type"].isin(trial_types)]
                    T = bold.shape[0]
                    labels = np.array([
                        row["trial_type"].replace("new_bloc_", "")
                        for _, row in df.iterrows()
                        if 0 <= int(row["onset"] / tr) + hrf_delay
                        and int(row["onset"] / tr) + hrf_delay + window_trs <= T
                    ])

                    scenes_response[key] = {
                        "trials": trials,   # (n_blocks, window_trs, G_or_parcels)
                        "start":  start,    # (n_blocks,) onset TR after HRF shift
                        "end":    end,      # (n_blocks,) = start + window_trs
                        "labels": labels,   # (n_blocks,) e.g. "left_hand", "tongue"
                    }

                except FileNotFoundError as e:
                    print(f"  Skipping {key}: {e}")

        print(f"Loading parcel atlas info")
        atlas = self.get_atlas_info()

        return {
            "scenes_response": scenes_response,
            "parcel_coords":   atlas["parcel_coords"],
            "parcel_ids":      atlas["parcel_ids"],
            "parcel_desc":     atlas["parcel_desc"],
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_hcptrt_loader(
    fmri_dir: str,
    parcellation: str = "mmp",
    subjects: Optional[List[str]] = None,
) -> HCPTRTLoader:
    """Create and return a default HCPTRTLoader instance.

    Parameters
    ----------
    fmri_dir : str
        Root of the preprocessed hcptrt dataset.
    parcellation : str
        hcp_utils parcellation. Default ``"mmp"`` (Glasser 360).
    subjects : list of str, optional
        Defaults to the three open-access subjects.

    Examples
    --------
    >>> loader = get_hcptrt_loader(
    ...     fmri_dir="data/cneuromod.processed/fmriprep/hcptrt"
    ... )
    """
    return HCPTRTLoader(
        fmri_dir=fmri_dir,
        subjects=subjects,
        parcellation=parcellation,
    )