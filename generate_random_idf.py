#!/usr/bin/env python3
"""
Generate random IDF files using CoverageControl library.

Usage:
    python generate_random_idf.py --output-dir sim_env_save --num-files 5

This will create world_idf_4.idf, world_idf_5.idf, ..., world_idf_8.idf
"""

import argparse
import sys
from pathlib import Path

# Add to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_control as cc


def generate_random_idf(params, output_file, pos_file):
    """
    Generate a random IDF and save it to file.

    Args:
        params: cc.Parameters object with IDF generation settings
        output_file: Path to save the .idf file
        pos_file: Path to save the robot positions .pos file
    """
    # Create a CoverageSystem with random IDF
    env = cc.CoverageSystem(params)

    # Save using WriteEnvironment which creates both .idf and .pos files
    # The method takes base filenames without extensions
    # Note: WriteEnvironment creates files WITHOUT extensions, so we need to rename them
    pos_base = pos_file.stem  # Filename without extension
    idf_base = output_file.stem  # Filename without extension

    # WriteEnvironment needs to be called from the directory where files will be saved
    import os
    import shutil
    original_dir = os.getcwd()
    try:
        os.chdir(output_file.parent)
        env.WriteEnvironment(pos_base, idf_base)

        # Rename files to add extensions (WriteEnvironment creates files without extensions)
        if Path(idf_base).exists():
            shutil.move(idf_base, output_file.name)
        if Path(pos_base).exists():
            shutil.move(pos_base, pos_file.name)
    finally:
        os.chdir(original_dir)

    print(f"Generated: {output_file}")

    # Return the WorldIDF for info
    return env.GetWorldIDF()


def main():
    parser = argparse.ArgumentParser(description="Generate random IDF files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sim_env_save",
        help="Output directory (default: sim_env_save)"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help="Number of IDF files to generate (default: 1)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting index for file numbering (default: auto-detect next available)"
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=3,
        help="Number of Gaussian features (default: 3)"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=100,
        help="World map size (default: 100)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Auto-detect starting index if not specified
    if args.start_index is None:
        existing_files = sorted(output_dir.glob("world_idf_*.idf"))
        if existing_files:
            # Extract numbers from filenames
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.stem.split('_')[-1])
                    numbers.append(num)
                except ValueError:
                    pass
            start_index = max(numbers) + 1 if numbers else 1
        else:
            start_index = 1
    else:
        start_index = args.start_index

    # Set up parameters
    params = cc.Parameters()
    params.pNumRobots = 4  # Doesn't matter for IDF generation
    params.pWorldMapSize = args.world_size
    params.pResolution = 1

    # IDF parameters
    params.pNumGaussianFeatures = args.num_features
    params.pMinSigma = 3#10
    params.pMaxSigma = 10 #20
    params.pMinPeak = 6
    params.pMaxPeak = 10
    params.pNumPolygons = 0

    print(f"Generating {args.num_files} random IDF file(s)...")
    print(f"  World size: {args.world_size}x{args.world_size}")
    print(f"  Gaussian features: {args.num_features}")
    print(f"  Sigma range: [{params.pMinSigma}, {params.pMaxSigma}]")
    print(f"  Peak range: [{params.pMinPeak}, {params.pMaxPeak}]")
    print()

    # Generate files
    for i in range(args.num_files):
        file_index = start_index + i
        idf_file = output_dir / f"world_idf_{file_index}.idf"
        pos_file = output_dir / f"robot_pos_{file_index}.pos"

        # Generate and save
        world_idf = generate_random_idf(params, idf_file, pos_file)

        # Print some info about the generated IDF
        world_map = world_idf.GetWorldMap()
        print(f"  Also created: {pos_file.name}")
        print(f"  Max value: {world_map.max():.4f}, Mean: {world_map.mean():.4f}")

    print(f"\nDone! Generated {args.num_files} file(s) in {output_dir}")
    print(f"  world_idf_{start_index}.idf through world_idf_{start_index + args.num_files - 1}.idf")


if __name__ == "__main__":
    main()
