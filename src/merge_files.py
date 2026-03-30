import os
import glob

def extract_number(filepath):
    basename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(basename)[0]
    try:
        return int(name_without_ext)
    except ValueError:
        return -1

def merge_text_files(input_folder, output_filename):
    absolute_input = os.path.abspath(input_folder)
    print(f"Searching for files in: {absolute_input}")
    
    search_path = os.path.join(input_folder, "*.txt")
    file_list = glob.glob(search_path)
    
    print(f"Found {len(file_list)} files.")
    
    if len(file_list) == 0:
        return

    file_list.sort(key=extract_number)
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    skipped_files = 0
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for i, file_path in enumerate(file_list):
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read().strip()
                if content:
                    outfile.write(content + "\n")
                else:
                    skipped_files += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Merged {i + 1} files...")
                
    print(f"Successfully merged all files into {output_filename}")
    print(f"Skipped {skipped_files} empty files.")

if __name__ == "__main__":
    INPUT_FOLDER = "data/raw/hindi_corpus/train"
    OUTPUT_FILE = "data/raw/hindi_corpus/merged_train.txt"
    merge_text_files(INPUT_FOLDER, OUTPUT_FILE)
