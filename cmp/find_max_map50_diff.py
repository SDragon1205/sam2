def find_top_map50_diff_gt(file1, file2, top):
    def extract_blocks(filepath):
        with open(filepath) as f:
            lines = f.readlines()
        return [lines[i:i+4] for i in range(0, len(lines), 4)]

    blocks1 = extract_blocks(file1)
    blocks2 = extract_blocks(file2)

    diffs = []

    for b1, b2 in zip(blocks1, blocks2):
        img1, img2 = b1[0].strip(), b2[0].strip()
        if img1 != img2:
            continue  # skip if paths mismatch

        def extract_map50(line):
            parts = line.split("mAP50:")
            if len(parts) > 1:
                return float(parts[1].split(",")[0])
            return None

        map1 = extract_map50(b1[2])
        map2 = extract_map50(b2[2])

        if map1 is not None and map2 is not None and map2 > map1:
            diff = map2 - map1
            diffs.append((img1, diff, map1, map2))

    # sort by descending diff
    ans = sorted(diffs, key=lambda x: x[1], reverse=True)[:top]
    return ans

top10 = find_top_map50_diff_gt("1.txt", "2.txt", 20)
for i, (img, diff, m1, m2) in enumerate(top10, 1):
    print(f"{i}. {img}\n   Model1: {m1:.4f}, Model2: {m2:.4f}, ΔmAP50: {diff:.4f}\n")
# def find_max_map50_diff_gt(file1, file2):
#     def extract_blocks(filepath):
#         with open(filepath) as f:
#             lines = f.readlines()
#         return [lines[i:i+4] for i in range(0, len(lines), 4)]

#     blocks1 = extract_blocks(file1)
#     blocks2 = extract_blocks(file2)

#     max_diff = -1
#     max_path = None

#     for b1, b2 in zip(blocks1, blocks2):
#         img1, img2 = b1[0].strip(), b2[0].strip()
#         if img1 != img2:
#             continue  # skip if paths mismatch

#         def extract_map50(line):
#             parts = line.split("mAP50:")
#             if len(parts) > 1:
#                 return float(parts[1].split(",")[0])
#             return None

#         map1 = extract_map50(b1[2])
#         map2 = extract_map50(b2[2])

#         if map1 is not None and map2 is not None and map2 > map1:
#             diff = map2 - map1
#             if diff > max_diff:
#                 max_diff = diff
#                 max_path = img1

#     return max_path, max_diff

# img_path, diff = find_max_map50_diff_gt("1.txt", "2.txt")
# print(f"最大差異的影像: {img_path}\n差異值: {diff:.4f}")