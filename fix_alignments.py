def fix_groundtruth_to_single_line(input_path, output_path):
    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError("Each transform should span exactly 3 lines (3x3 matrix)")

    fixed_lines = []
    for i in range(0, len(lines), 3):
        # Flatten the three lines into one line
        row1 = lines[i].split()
        row2 = lines[i+1].split()
        row3 = lines[i+2].split()
        full_line = row1 + row2 + row3
        if len(full_line) != 9:
            raise ValueError(f"Invalid number of elements in transform at lines {i}-{i+2}")
        fixed_lines.append(" ".join(full_line))

    with open(output_path, "w") as f:
        f.write("\n".join(fixed_lines) + "\n")

    print(f"✅ Transforms flattened to 1-line format in: {output_path}")
    print(f"✔️  Total transforms: {len(fixed_lines)}")

# Example usage
fix_groundtruth_to_single_line(
    input_path="/home/nugh75/Git/archeology-fragment-reconstruction/dataset/250331-clean/set24/images/fragments/ortho_24.png/s8/other_folder/groundTruth.txt",
    output_path="/home/nugh75/Git/archeology-fragment-reconstruction/dataset/250331-clean/set24/images/fragments/ortho_24.png/s8/other_folder/groundTruth.txt"
)