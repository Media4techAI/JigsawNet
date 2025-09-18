'''
Convert pairwise alignment transformation to one stitched image
'''

import cv2
import numpy as np

'''
transform dst to src
'''
def FusionImage(src, dst, transform, bg_color=[0,0,0]):
    black_bg = [0,0,0]
    if bg_color!=black_bg:
        src[np.where((src == bg_color).all(axis=2))] = [0,0,0]
        dst[np.where((dst == bg_color).all(axis=2))] = [0, 0, 0]

    color_indices = np.where((dst != black_bg).any(axis=2))
    color_pt_num = len(color_indices[0])
    one = np.ones(color_pt_num)

    color_indices = list(color_indices)
    color_indices.append(one)
    color_indices = np.array(color_indices)

    transformed_lin_pts = np.matmul(transform, color_indices)
    # bounding box after transform
    try:
        dst_min_row = np.floor(np.min(transformed_lin_pts[0])).astype(int)
        dst_min_col = np.floor(np.min(transformed_lin_pts[1])).astype(int)
        dst_max_row = np.ceil(np.max(transformed_lin_pts[0])).astype(int)
        dst_max_col = np.ceil(np.max(transformed_lin_pts[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    # global bounding box
    src_color_indices = np.where((src != black_bg).any(axis=2))
    try:
        src_min_row = np.floor(np.min(src_color_indices[0])).astype(int)
        src_min_col = np.floor(np.min(src_color_indices[1])).astype(int)
        src_max_row = np.ceil(np.max(src_color_indices[0])).astype(int)
        src_max_col = np.ceil(np.max(src_color_indices[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    min_row = min(dst_min_row, src_min_row)
    max_row = max(dst_max_row, src_max_row)
    min_col = min(dst_min_col, src_min_col)
    max_col = max(dst_max_col, src_max_col)

    offset_row = -min_row
    offset_col = -min_col

    offset_transform = np.float32([[1,0,offset_col],[0,1,offset_row]])
    dst_transform = np.matmul(np.matrix([[1,0,offset_row],[0,1,offset_col],[0,0,1]]), transform)
    # convert row, col to opencv x,y
    dst_transform = np.float32([[dst_transform[0,0], dst_transform[1,0], dst_transform[1,2]], [dst_transform[0,1], dst_transform[1,1], dst_transform[0,2]]])

    src_transformed = cv2.warpAffine(src, offset_transform, (max_col-min_col, max_row-min_row))
    dst_transformed = cv2.warpAffine(dst, dst_transform, (max_col-min_col, max_row-min_row))

    # overlap detection
    a = np.all(src_transformed == black_bg, axis=2)
    b = np.all(dst_transformed != black_bg, axis=2)
    c = np.logical_and(a, b)
    c = c.reshape((c.shape[0], c.shape[1]))
    non_overlap_indices = np.where(c)
    if len(np.where(b)[0]) == 0:
        assert False and "no valid pixels in transformed dst image, please check the transform process"
    else:
        overlap_ratio = 1 - len(non_overlap_indices[0]) / len(np.where(b)[0])

    # fusion
    bg_indices = np.where(a)
    src_transformed[bg_indices] = dst_transformed[bg_indices]

    offset_transform_matrix = np.float32([[1, 0, offset_row], [0, 1, offset_col], [0,0,1]])
    return [src_transformed, overlap_ratio, offset_transform_matrix]

import os
import Utils as Utils
import cv2

def generate_alignment_images(
        fragments_dir,
        alignments_path=None,
        output_dir=None,
        bg_color=None,
    ):
    """
    Generate stitched preview images for each alignment in a folder.

    Inputs:
    - fragments_dir: directory containing fragment_XXXX.png and optional bg_color.txt
    - alignments_path: path to alignments.txt (defaults to fragments_dir/alignments.txt)
    - output_dir: directory to save stitched images (defaults to fragments_dir/alignment_images)
    - bg_color: list [B,G,R] background color; if None, tries bg_color.txt, else uses [0,0,0]

    Output:
    - A list of saved image file paths.
    """
    # Resolve defaults
    if alignments_path is None:
        alignments_path = os.path.join(fragments_dir, "alignments.txt")
    if output_dir is None:
        output_dir = os.path.join(fragments_dir, "alignment_images")

    with open(alignments_path) as f:
        for ln in f:
            if not ln.startswith('Node'):
                parts = ln.split()
                A = np.array(list(map(float, parts[3:12]))).reshape(3,3)
                break

    # Discover background color if not provided
    if bg_color is None:
        bg_color_file = os.path.join(fragments_dir, "bg_color.txt")
        if os.path.exists(bg_color_file):
            try:
                with open(bg_color_file) as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            # bg_color.txt is stored as RGB; OpenCV uses BGR
                            rgb = [int(x) for x in parts]
                            bg_color = rgb[::-1]
                            break
            except Exception:
                bg_color = None
    if bg_color is None:
        bg_color = [0, 0, 0]

    os.makedirs(output_dir, exist_ok=True)
    
    # Parse alignments
    alignments = Utils.Alignment2d(alignments_path)

    saved_paths = []
    for idx, alignment in enumerate(alignments.data):
        v1 = alignment.frame1
        v2 = alignment.frame2
        trans = alignment.transform
        
        # Load fragment images
        img1_path = os.path.join(fragments_dir, "fragment_{:04}.png".format(v1 + 1))
        img2_path = os.path.join(fragments_dir, "fragment_{:04}.png".format(v2 + 1))
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            # Skip if missing fragments
            continue


        fused = FusionImage(img1, img2, trans, bg_color)
        if len(fused) == 0:
            continue
        fused_img = fused[0]

        out_name = "alignment_{:04}_v{:04}_v{:04}_r{}".format(idx, v1 + 1, v2 + 1, alignment.rank)
        out_path = os.path.join(output_dir, out_name + ".png")
        cv2.imwrite(out_path, fused_img)
        saved_paths.append(out_path)

    return saved_paths


# fragments_dir = "C:\\Users\\user\\Git\\JigsawNet\\Examples\\MIT_ex"
# alignments_path = "C:\\Users\\user\\Git\\JigsawNet\\Examples\\MIT_ex\\alignments.txt"
# output_dir = "C:\\Users\\user\\Git\\JigsawNet\\Examples\\MIT_ex\\alignment_images"
# generate_alignment_images(fragments_dir, alignments_path=alignments_path, output_dir=output_dir, bg_color=[232, 8, 248])  # pink background
