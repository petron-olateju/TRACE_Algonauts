"""Parcel-to-lobe mapping tables for fMRI atlas parcellations.

This module centralises all atlas-specific region → lobe lookup dicts so that
loaders (hcptrt, algonauts) and preprocessing code share a single source of
truth.  To add support for a new parcellation:

    1. Define a new ``Dict[str, str]`` mapping region/network names to lobe
       labels (see existing dicts below for conventions).
    2. Register it in ``PARCELLATION_MAP`` at the bottom of this file under a
       short string key that matches the parcellation name used elsewhere in
       the codebase.

Lobe label conventions (used consistently across all maps)
----------------------------------------------------------
    "occipital"      Primary and association visual cortex
    "parietal"       Parietal cortex (dorsal visual, attention, somatosensory association)
    "temporal"       Lateral and medial temporal cortex; hippocampus; amygdala
    "prefrontal"     Lateral and medial prefrontal cortex; default-mode prefrontal nodes
    "somatomotor"    Primary motor and somatosensory cortex; premotor areas
    "insular"        Insular and opercular cortex; salience / cingulo-opercular network
    "cingulate"      Anterior and posterior cingulate cortex
    "orbitofrontal"  Orbitofrontal cortex; limbic network nodes
    "basal_ganglia"  Striatum (caudate, putamen, accumbens) and pallidum
    "diencephalon"   Thalamus and hypothalamus
    "cerebellum"     Cerebellar cortex and deep nuclei
    "brainstem"      Brainstem structures
"""

from typing import Dict

# ---------------------------------------------------------------------------
# MMP (Glasser HCP multimodal parcellation) area → lobe
# ---------------------------------------------------------------------------
# Maps each Glasser MMP area name (without L_/R_ prefix) to a coarse lobe.
# Based on the 22 cortical lobes in Glasser et al. 2016, collapsed to 8 broad
# lobes, plus subcortical structures.
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
    "IPS1":"parietal","MIP":"parietal","VIP":"parietal",
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
    # Subcortical — mapped to their primary neuroanatomical association
    # Basal ganglia: striatum + pallidum
    "accumbens_left":"basal_ganglia","accumbens_right":"basal_ganglia",
    "caudate_left":"basal_ganglia","caudate_right":"basal_ganglia",
    "putamen_left":"basal_ganglia","putamen_right":"basal_ganglia",
    "pallidum_left":"basal_ganglia","pallidum_right":"basal_ganglia",
    # Limbic / temporal — hippocampal formation and amygdala
    "hippocampus_left":"temporal","hippocampus_right":"temporal",
    "amygdala_left":"temporal","amygdala_right":"temporal",
    # Diencephalon — thalamus and hypothalamus
    "thalamus_left":"diencephalon","thalamus_right":"diencephalon",
    "diencephalon_left":"diencephalon","diencephalon_right":"diencephalon",
    # Cerebellum
    "cerebellum_left":"cerebellum","cerebellum_right":"cerebellum",
    # Brainstem
    "brainStem":"brainstem",
}

# ---------------------------------------------------------------------------
# Cole-Anticevic (ca_parcels / ca_network) network → lobe
# ---------------------------------------------------------------------------
# Maps the network name portion of a ca_parcels label (after stripping the
# trailing "-NN" parcel index) to a broad lobe / functional grouping.
# Subcortical structure names (lower-cased) fall back to MMP_LOBE at lookup
# time; this dict covers the cortical network names only.
CA_NETWORK_LOBE: Dict[str, str] = {
    "Visual1":              "occipital",
    "Visual2":              "occipital",
    "Somatomotor":          "somatomotor",
    "Auditory":             "temporal",
    "Cingulo-Opercular":    "insular",
    "Language":             "temporal",
    "Default":              "prefrontal",
    "Frontoparietal":       "parietal",
    "Dorsal-Attention":     "parietal",
    "Ventral-Attention":    "insular",
    "Ventral-Multimodal":   "temporal",
    "Posterior-Multimodal": "parietal",
    "Orbito-Affective":     "orbitofrontal",
}

# ---------------------------------------------------------------------------
# Schaefer atlas network → lobe
# ---------------------------------------------------------------------------
# Covers both 7-network and 17-network Schaefer parcellations.
# The network name is the middle portion of a Schaefer label after stripping
# hemisphere prefix and numeric suffix, e.g. "7Networks_LH_Vis_1" -> "Vis".
# 17-network names include A/B/C sub-network suffixes mapped to the same lobe
# as their parent network.
SCHAEFER_LOBE: Dict[str, str] = {
    # Visual
    "Vis":          "occipital",
    "VisCent":      "occipital",
    "VisPeri":      "occipital",
    # Somatomotor
    "SomMot":       "somatomotor",
    "SomMotA":      "somatomotor",
    "SomMotB":      "somatomotor",
    # Dorsal attention
    "DorsAttn":     "parietal",
    "DorsAttnA":    "parietal",
    "DorsAttnB":    "parietal",
    # Salience / ventral attention
    "SalVentAttn":  "insular",
    "SalVentAttnA": "insular",
    "SalVentAttnB": "insular",
    # Limbic
    "Limbic":       "orbitofrontal",
    "LimbicA":      "orbitofrontal",
    "LimbicB":      "temporal",
    # Frontoparietal / control
    "Cont":         "prefrontal",
    "ContA":        "prefrontal",
    "ContB":        "prefrontal",
    "ContC":        "prefrontal",
    # Default mode
    "Default":      "prefrontal",
    "DefaultA":     "prefrontal",
    "DefaultB":     "prefrontal",
    "DefaultC":     "prefrontal",
    # Temporal parietal (17-network only)
    "TempPar":      "parietal",
}

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------
# Maps parcellation name strings (as used in loader arguments) to their
# corresponding lobe dict.  Add new parcellations here alongside their dict.
PARCELLATION_MAP: Dict[str, Dict[str, str]] = {
    # HCP / HCPTRT parcellations
    "mmp":        MMP_LOBE,
    "yeo7":       {},          # Yeo networks are top-level labels; no lobe sub-division
    "yeo17":      {},          # same — add a SCHAEFER_LOBE-style dict here if needed
    "ca_parcels": CA_NETWORK_LOBE,
    "ca_network": CA_NETWORK_LOBE,
    # Algonauts / Schaefer parcellations
    "schaefer":   SCHAEFER_LOBE,
    "algonauts":  SCHAEFER_LOBE,
}


def get_lobe(parcellation: str, region_key: str, fallback: str = "") -> str:
    """Look up the lobe label for a region key in a given parcellation.

    Parameters
    ----------
    parcellation : str
        Parcellation name as registered in ``PARCELLATION_MAP``
        (e.g. ``"mmp"``, ``"ca_parcels"``, ``"schaefer"``).
    region_key : str
        The cleaned region / network / area name to look up.
    fallback : str
        Value to return when the key is not found. Defaults to ``region_key``
        itself if an empty string is passed, so callers always get a string.

    Returns
    -------
    str
        Lobe label, or ``fallback`` (defaulting to ``region_key``) if not found.
    """
    lobe_map = PARCELLATION_MAP.get(parcellation, {})
    result = lobe_map.get(region_key)
    if result is not None:
        return result
    # Cross-check MMP_LOBE as a universal fallback for subcortical keys
    result = MMP_LOBE.get(region_key)
    if result is not None:
        return result
    return fallback if fallback else region_key