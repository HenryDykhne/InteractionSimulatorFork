import argparse
from pathlib import Path
import shutil

import pandas as pd


# From https://github.com/sisl/InteractionSimulator/blob/main/intersim/utils.py
# LOCATIONS = [
#     'DR_USA_Roundabout_FT',
#     'DR_CHN_Roundabout_LN',
#     'DR_DEU_Roundabout_OF',
#     'DR_USA_Roundabout_EP',
#     'DR_USA_Roundabout_SR'
# ]

def convert_interaction_split(root_dir, sim_dir, split):
    target_dir = sim_dir / "datasets"
    target_dir.mkdir(exist_ok=True)
    trackfiles_dir = target_dir / "trackfiles"
    trackfiles_dir.mkdir(exist_ok=True)
    maps_dir = target_dir / "maps"
    maps_dir.mkdir(exist_ok=True)

    for scenario_path in (root_dir / split).glob("DR_*.csv"):
        scenario_name = scenario_path.stem.replace(f"_{split}", "")
        print(f"Processing: {scenario_name}")

        # Generate trackfiles files
        src_csv_file = scenario_path
        target_dir = trackfiles_dir / scenario_name
        target_dir.mkdir(exist_ok=True)
        df = pd.read_csv(src_csv_file)
        num_cases = int(df.iloc[-1][['case_id']])
        for case in range(num_cases):
            for agent_type in ['car', 'pedestrian/bicycle']:
                selected_rows = df[(df['case_id'] == case) & (df['agent_type'] == agent_type)]
                selected_rows = selected_rows[['track_id', 'frame_id', 'timestamp_ms', 'x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']]
                filename = f"{'vehicle_tracks' if agent_type[:3] == 'car' else 'pedestrian_tracks'}_{case:04}.csv"
                selected_rows.to_csv(target_dir / filename, index=False)

        # Copy map over to target location
        src_map_file = root_dir / "maps" / f"{scenario_name}.osm"
        target_map_file = maps_dir / f"{scenario_name}.osm"
        shutil.copy(src_map_file, target_map_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path("~/datasets/INTERACTION/INTERACTION-Dataset-DR-multi-v1_2"),
        help="Path to the INTERACTION dataset directory containing the train and val subfolders."
    )
    parser.add_argument(
        "--sim_dir",
        type=Path,
        default=Path("~/school/ece_750/InteractionSimulatorFork"),
        help="Path to the InteractionSimulator repo."
    )
    parser.add_argument(
        "--process_val",
        action='store_true',
        help="Toggle this arg to process val split instead of train."
    )
    args = parser.parse_args()
    root_dir = args.root_dir.expanduser()
    sim_dir = args.sim_dir.expanduser()

    split = "val" if args.process_val else "train"
    convert_interaction_split(root_dir, sim_dir, split)
