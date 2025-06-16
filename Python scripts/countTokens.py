import os
import json
import re
from collections import defaultdict

# Path to the root directory
ROOT_DIR = os.path.join(os.path.dirname(__file__), 'NJR_HUMAN_EVAL_FINAL')

# Function to count words in a string
def count_words(text):
    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b', text)
    return len(words)

# Function to count words in a JSON file
def count_words_in_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return count_words(content)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return 0

def main():
    # Dictionary to group results by task name
    task_results = defaultdict(list)

    # Open a file to save the results
    with open('word_counts.txt', 'w', encoding='utf-8') as output_file:
        for root, dirs, files in os.walk(ROOT_DIR):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    word_count = count_words_in_json(file_path)
                    folder_name = os.path.basename(root)
                    parent_folder = os.path.basename(os.path.dirname(root))
                    result = f"{parent_folder}/{folder_name}: {word_count} words"
                    task_results[folder_name].append(result)

        # Write grouped results to the file
        for task, results in task_results.items():
            output_file.write(f"Task: {task}\n")
            for result in results:
                output_file.write(f"  {result}\n")
            output_file.write("\n")

if __name__ == "__main__":
    main()
