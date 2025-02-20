import os
import json
import math


def check_values(obj, key_path="", filename=""):
    """Recursively checks if innermost values are valid numbers, prints issues."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key_path = f"{key_path}.{key}" if key_path else key
            check_values(value, new_key_path, filename)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            check_values(item, f"{key_path}[{idx}]", filename)
    else:
        if (
            not isinstance(obj, (int, float))
            or math.isnan(obj)
            or math.isinf(obj)
        ):
            print(f"Invalid number in {filename} at '{key_path}': {obj}")


def check_json_files(directory):
    """Iterates through all JSON files in a directory and checks their values."""
    for filename in os.listdir(directory):
        if "mod_list" in filename:
            continue 
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    check_values(data, filename=filename)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {filename}: {e}")


# Set your directory containing JSON files
json_directory = "./nc_workspace_tmp/"  # Change this to your actual directory
check_json_files(json_directory)
