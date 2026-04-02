import re
import sys
import os

def parse_tab_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split into chunks based on "Measure X:"
    measure_pattern = r'Measure\s+(\d+)\s*:'
    chunks = re.split(measure_pattern, content)
    
    # re.split with a group returns: [pre-text, num1, block1, num2, block2...]
    # We need to skip the first element (pre-text)
    data = {}
    for i in range(1, len(chunks), 2):
        m_num = chunks[i]
        block = chunks[i+1]
        
        # Standard tab order: 0=e, 1=B, 2=G, 3=D, 4=A, 5=E
        # We initialize with None to detect if a string was found
        measure_strings = [None] * 6
        
        lines = block.strip().split('\n')
        for line in lines:
            if '|' not in line: continue
            
            parts = line.split('|', 1)
            label = parts[0].strip()
            # Remove all whitespace including non-breaking spaces (\xa0)
            notes = parts[1].replace(' ', '').replace('\xa0', '').strip()
            
            # Map source labels to standard positions
            if label == 'E' and measure_strings[0] is None: idx = 0 # High E
            elif label == 'B': idx = 1
            elif label == 'G': idx = 2
            elif label == 'D': idx = 3
            elif label == 'A': idx = 4
            elif label == 'E': idx = 5 # Low E
            else: continue
            
            measure_strings[idx] = notes if notes else "-"

        data[m_num] = [s if s is not None else "-" for s in measure_strings]

    return data

def save_tab(data, output_path):
    string_names = ['e', 'B', 'G', 'D', 'A', 'E']
    # Sort measures numerically
    measure_keys = sorted(data.keys(), key=int)
    step = 8 
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(measure_keys), step):
            chunk = measure_keys[i : i + step]
            
            # Header: Just the numbers
            header = "    "
            for m_num in chunk:
                header += f"{m_num:<10}"
            f.write(header.rstrip() + "\n")
            
            for s_idx in range(6):
                line = f"{string_names[s_idx]}|"
                for m_num in chunk:
                    val = data[m_num][s_idx]
                    # If the measure is empty or just a dash, fill with dashes
                    if val == "-":
                        line += "----------"
                    else:
                        # Add a dash before and pad after to reach 10 chars
                        line += f"-{val:-<9}"
                line += "|"
                f.write(line + "\n")
            f.write("\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 human_readable.py <filename>")
    else:
        fn = sys.argv[1]
        tab_data = parse_tab_file(fn)
        if tab_data:
            save_tab(tab_data, "human_" + fn)
            print(f"Processed {len(tab_data)} measures into human_{fn}")