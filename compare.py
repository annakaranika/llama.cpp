def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_floats(line): 
    return [float(x) for x in line.strip().split() if is_float(x)]

def compare_lines_multi(file1_path, start1, file2_path, start2, num_lines, tol=1e-6):
    with open(file1_path, 'r') as f1:
        lines1 = f1.readlines()
    with open(file2_path, 'r') as f2:
        lines2 = f2.readlines()

    end1 = start1 + num_lines
    end2 = start2 + num_lines

    if end1 > len(lines1) or end2 > len(lines2):
        print("❌ Line range out of bounds.")
        return

    all_match = True
    for i in range(num_lines):
        lnum1 = start1 + i
        lnum2 = start2 + i

        floats1 = parse_floats(lines1[lnum1])
        floats2 = parse_floats(lines2[lnum2])

        if len(floats1) != len(floats2):
            print(f"❌ Line {lnum1+1} vs {lnum2+1}: Float count mismatch ({len(floats1)} vs {len(floats2)})")
            all_match = False
            continue

        for j, (a, b) in enumerate(zip(floats1, floats2)):
            if abs(a - b) > tol:
                print(f"❌ Line {lnum1+1}, Value {j+1}: {a} != {b}")
                all_match = False

    if all_match:
        print("✅ All compared lines match within tolerance.")

# Example usage:
# Compare 10 lines starting at line 100 in file1 with lines starting at 200 in file2
compare_lines_multi("dump7.txt", int(input("Start line in file1: "))-1,
                    "adump1.txt", int(input("Start line in file2: "))-1,
                    int(input("Number of lines to compare: ")))
