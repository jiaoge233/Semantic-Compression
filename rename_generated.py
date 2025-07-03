import os
import argparse
import re

def rename_generated_files(directory):
    """
    Renames files in a directory by removing a specific prefix.
    Example: 'generated_from_0_output_image.png' -> '0_output_image.png'
    """
    print(f"Scanning directory: {directory}")
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory}")
        return

    renamed_count = 0
    prefix_to_remove = "generated_from_"

    for filename in files:
        if filename.startswith(prefix_to_remove):
            old_path = os.path.join(directory, filename)
            # Create the new filename by removing the prefix
            new_filename = filename.replace(prefix_to_remove, '', 1)
            new_path = os.path.join(directory, new_filename)

            # Check for conflicts before renaming
            if os.path.exists(new_path):
                print(f"Skipping rename for '{filename}' because target '{new_filename}' already exists.")
                continue

            try:
                os.rename(old_path, new_path)
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming file {filename}: {e}")

    print(f"\nFinished processing. Total files renamed: {renamed_count}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch renames generated files by removing the 'generated_from_' prefix.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('directory', type=str, help='Path to the directory containing files to rename.')
    
    args = parser.parse_args()
    rename_generated_files(args.directory) 