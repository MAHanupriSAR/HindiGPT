import os
import glob

def merge_text_files(input_folder, output_filename):
    search_path = os.path.join(input_folder, "*.txt")
    file_list = glob.glob(search_path)
    
    print(f"Found {len(file_list)} files.")
    
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for i, file_path in enumerate(file_list):
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read().strip()
                if content:
                    outfile.write(content + "\n")
            
            if (i + 1) % 1000 == 0:
                print(f"Merged {i + 1} files")
                
    print(f"Merged into {output_filename}")

if __name__ == "__main__":
    INPUT_FOLDER = "data/raw/hindi_corpus/train"
    OUTPUT_FILE = "data/raw/hindi_corpus/merged_train.txt"
    merge_text_files(INPUT_FOLDER, OUTPUT_FILE)