import argparse
import os
import yaml
import numpy as np
import nibabel as nib
from tqdm import tqdm

from utils.dataloader import HCPTRTLoader
from utils.preprocessing import parcel_samples_hcptrt


def run_preprocessing_hcptrt(subject: str):
    """
    Generate X and Y for each task-session-run combination for the subject.
    Saves results independently in dataset/hcptrt/sub-**/ses-***/run-*/
    """
    # Load directory configuration to get fmri_dir
    with open("./configs/dirs.yaml", "r") as f:
        dir_configs = yaml.safe_load(f)
    
    with open("./configs/configs.yaml", "r") as f:
        configs = yaml.safe_load(f)
    
    fmri_dir = dir_configs["hcptrt"]["dirs"]["fmri"]
    params = configs["hcptrt"]["params"]
    tr = params["tr"]
    hrf_delay = params["hrf_delay"]
    parcellation = params["parcellation"]
    
    # Ensure subject ID is in sub-XX format
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"

    # Initialize loader with params from config
    loader = HCPTRTLoader(
        fmri_dir=fmri_dir, 
        subjects=[subject],
        tr=tr,
        hrf_delay=hrf_delay,
        parcellation=parcellation
    )
    
    print(f"Starting preprocessing for {subject} on HCPTRT dataset...")

    for task in loader.TASKS:
        sessions = loader.list_sessions(subject)
        for session in sessions:
            runs = loader.list_runs(subject, session, task)
            for run in runs:
                try:
                    bold_path = loader.get_bold_path(subject, session, task, run)
                    if not os.path.exists(bold_path):
                        continue
                    
                    # Get original number of timepoints (T)
                    img = nib.load(bold_path)
                    T = img.shape[0]
                    
                    # Generate X and Y using parcel_samples_hcptrt
                    # trials="continuous" with n_windows=T preserves original timepoints
                    X, Y = parcel_samples_hcptrt(
                        loader,
                        subject,
                        task,
                        session=session,
                        run=run,
                        trials="continuous",
                        time_collapse="windowed_mean",
                        n_windows=T
                    )
                    
                    # Define output directory
                    out_dir = f"dataset/hcptrt/{subject}/{session}/run-{run:02d}"
                    os.makedirs(out_dir, exist_ok=True)
                    
                    # Save X as numpy and Y as CSV
                    # Note: We add the task to the filename to avoid collisions if multiple tasks 
                    # were in the same folder, though here folders are session/run specific.
                    np.save(os.path.join(out_dir, f"X_{task}.npy"), X)
                    Y.to_csv(os.path.join(out_dir, f"Y_{task}.csv"), index=False)
                    
                    print(f"  Processed {task} | {session} | run-{run:02d} (T={T})")
                    
                except Exception as e:
                    print(f"  Error processing {task} | {session} | run-{run}: {e}")


def main():
    parser = argparse.ArgumentParser(description="TRACE: Algonauts & HCPTRT CLI")
    parser.add_argument(
        "mode", 
        choices=["preprocessing", "training"], 
        help="Select mode: preprocessing or training"
    )
    parser.add_argument(
        "dataset", 
        choices=["hcptrt", "algonauts"], 
        help="Select dataset"
    )
    parser.add_argument(
        "subject", 
        help="Subject identifier (e.g., 01 or sub-01)"
    )

    args = parser.parse_args()

    if args.mode == "preprocessing":
        if args.dataset == "hcptrt":
            run_preprocessing_hcptrt(args.subject)
        else:
            print(f"Preprocessing for {args.dataset} is not yet implemented.")
    elif args.mode == "training":
        print("Training mode is not yet implemented.")


if __name__ == "__main__":
    main()
