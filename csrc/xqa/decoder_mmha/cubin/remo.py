import os
import re

def remove_namespaces(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Remove namespace declarations
    content = re.sub(r'namespace\s+\w+\s*\n?\{', '', content)

    # Remove namespace closing brackets and comments
    content = re.sub(r'\}\s*//\s*namespace\s+\w+', '', content)

    # Remove extra newlines
    content = re.sub(r'\n{3,}', '\n\n', content)

    with open(file_path, 'w') as file:
        file.write(content)

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp'):  # Focus on .cpp files
                if re.match(r'xqa_kernel.*\.cubin\.cpp$', file):  # Check if file matches the pattern
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    remove_namespaces(file_path)

def main():
    directory_path = input("Enter the directory path containing the C++ files: ")
    if os.path.isdir(directory_path):
        process_directory(directory_path)
        print("Processing complete.")
    else:
        print("Invalid directory path. Please try again.")

if __name__ == "__main__":
    main()
