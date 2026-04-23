import os
from pathlib import Path

def shrink(filename, skip_factor=100):
    p = Path(filename)
    if not p.exists(): return
    
    print(f"Shrinking {p.name} from {p.stat().st_size / 1e6:.2f} MB...")
    with open(p, 'r') as f:
        lines = f.readlines()
        
    downsampled = lines[::skip_factor]
    
    with open(p, 'w') as f:
        f.writelines(downsampled)
        
    print(f"Done! New size: {p.stat().st_size / 1e6:.2f} MB.")

if __name__ == "__main__":
    results_dir = Path("fleming-model-based-brain/Model_Results")
    shrink(results_dir / "Force_amplitude_values.csv")
    shrink(results_dir / "Force_times.csv")
    shrink(results_dir / "sEMG_values.csv")
    shrink(results_dir / "sEMG_times.csv")
    
    print("Deleting unused .mat files...")
    import glob
    for mat_file in glob.glob(str(results_dir / "*.mat")):
        os.remove(mat_file)
        print("Removed", mat_file)
