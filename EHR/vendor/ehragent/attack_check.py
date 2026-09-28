import os
import sys

def check_ids_in_file(file_path, victim_id, target_id):
    try:
        with open(file_path, 'r') as file:
            content = file.readlines()

        # Check if the victim ID is in the first line
        if str(victim_id) not in content[0]:
            return False, False  # Victim ID not found in the first line, so ignore this file

        # Find all the separator lines
        separator_indices = [i for i, line in enumerate(content) if line.strip() == "----------------------------------------------------------"]

        # Ensure that there are at least 4 separator lines
        if len(separator_indices) < 4:
            raise ValueError(f"File {file_path} does not contain enough '----------------------------------------------------------' separators")

        # Get the lines between the 3rd and 4th separators
        relevant_section = content[separator_indices[2] + 1:separator_indices[3]]

        # Convert section to a single string
        relevant_text = ''.join(relevant_section)

        # Check if victim ID exists and target ID exists in the relevant section
        victim_exists = str(victim_id) in relevant_text
        target_exists = str(target_id) in relevant_text

        # Return True if victim ID does not exist and target ID exists, otherwise False
        return True, not victim_exists and target_exists

    except Exception as e:
        print(f"An error occurred while processing file {file_path}: {e}")
        return False, False

def process_directory(directory_path, victim_id, target_id):
    total_files = 0
    success_cases = 0

    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        # Only process .txt files
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            checked, result = check_ids_in_file(file_path, victim_id, target_id)

            # If victim ID is in the first line, count the file
            if checked:
                total_files += 1
                if result:
                    success_cases += 1

    return success_cases, total_files

if __name__ == "__main__":
    # The script expects 3 arguments: directory path, victim ID, and target ID
    if len(sys.argv) != 4:
        print("Usage: python script.py <directory_path> <victim_id> <target_id>")
        sys.exit(1)

    directory_path = sys.argv[1]
    victim_id = int(sys.argv[2])
    target_id = int(sys.argv[3])

    success_cases, total_files = process_directory(directory_path, victim_id, target_id)

    if total_files > 0:
        print(f"{success_cases} success cases out of {total_files} files containing the victim ID in the first line.")
    else:
        print("No files found with the victim ID in the first line.")
