# Legal Reasoning Project, NCCU (2025)
# merge_jsons.py: This code simply merges respective JSON files in a specific folder into a single JSON file.

import json
import os
from pathlib import Path
from typing import List, Any

def merge_json_files(input_folder: str, output_filename: str = "merged_output.json") -> None:
    """
    Reads all JSON files from the input_folder and merges them into a single JSON file.
    
    Args:
        input_folder (str): The path to the folder containing JSON files.
        output_filename (str): The name of the resulting merged file.
    """
    
    # Use pathlib for robust cross-platform path handling
    folder_path = Path(input_folder)
    # Write output to the same directory as this script
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / output_filename
    
    # List to hold all merged records
    all_data: List[Any] = []
    
    files_processed = 0
    
    print(f"Scanning folder: {folder_path}...")
    print(f"Writing merged output to: {output_path}")

    # Iterate through all files in the directory
    for file_path in folder_path.glob("*.json"):
        # Skip the output file if it would match the intended output path (avoid recursion)
        try:
            if file_path.resolve() == output_path.resolve():
                continue
        except Exception:
            if file_path.name == output_filename:
                continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Logic to handle different JSON structures
                if isinstance(data, list):
                    # If the file contains a list (like your examples), extend the main list
                    all_data.extend(data)
                elif isinstance(data, dict):
                    # If the file contains a single object, append it
                    all_data.append(data)
                
                files_processed += 1
                print(f"✔ Loaded: {file_path.name}")
                
        except json.JSONDecodeError:
            print(f"⚠ Warning: Could not decode {file_path.name}. Skipping.")
        except Exception as e:
            print(f"⚠ Error reading {file_path.name}: {e}")

    # Write the merged result
    if all_data:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # ensure_ascii=False is CRITICAL for readable Chinese characters
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            
            print("-" * 30)
            print(f"Success! Merged {files_processed} files into:")
            print(f"-> {output_path.absolute()}")
            print(f"Total records: {len(all_data)}")
            
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print("No valid JSON data found to merge.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # You may change this path to your specific folder path
    target_folder = "./extraction_output"  
    
    merge_json_files(target_folder, "osh_doc_merged.json")