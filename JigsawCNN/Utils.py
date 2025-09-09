'''
This file save some common data structures
'''

import operator
import numpy as np

class GtPose:
    def __init__(self, gt_filename):
        self.data = []
        with open(gt_filename) as f:
            all_line = [line.rstrip() for line in f]
            for index, line in enumerate(all_line):
                if index%2!=0:
                    m1, m2, m3, m4, m5, m6, m7, m8, m9 = [t(s) for t, s in zip((float, float, float, float, float, float, float, float, float), line.split())]
                    pose = np.array([[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]])
                    self.data.append(pose)

        # transform ground truth to identity
        baseline = np.linalg.inv(self.data[0])
        for i in range(len(self.data)):
            self.data[i] = np.matmul(baseline, self.data[i])

class GtPoseMatrix:
    def __init__(self, gt_filename):
        self.data=list()
        with open(gt_filename) as f:
            all_line = [line.rstrip() for line in f]
            for i in range(len(all_line)//4):
                matrix = all_line[i*4:(i+1)*4]
                line1 = matrix[0]
                line2 = matrix[1]
                line3 = matrix[2]
                line4 = matrix[3]
                fragmentId = int(line1)
                assert i==fragmentId
                m1, m2, m3 = [t(s) for t, s in zip((float, float, float), line2.split())]
                m4, m5, m6 = [t(s) for t, s in zip((float, float, float), line3.split())]
                m7, m8, m9 = [t(s) for t, s in zip((float, float, float), line4.split())]
                pose = np.array([[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]])
                self.data.append(pose)



class PoseContainer:
    def __init__(self, pose_list):
        self.data = pose_list

    def SaveToFile(self, filename):
        with open(filename, 'w') as f:
            for fragmentId, pose in self.data:
                data_item = "%d\t%d\n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n"%(fragmentId, fragmentId, pose[0, 0], pose[0, 1], pose[0, 2], pose[1, 0], pose[1, 1], pose[1, 2], pose[2, 0], pose[2, 1], pose[2, 2])
                f.write(data_item)

    def CompareWithGT(self, gtpose, t_threshold, r_threshold):
        evaluation = dict()
        all_successful = True
        for fragmentId, pose in self.data:
            gt_pose = gtpose.data[fragmentId]
            err_mat = np.matmul(pose, np.linalg.inv(gt_pose))

            theta = np.arccos(err_mat[0,0])*180/3.1415926
            t = np.sqrt(err_mat[0,2]**2+err_mat[1,2]**2)
            if theta<r_threshold and t<t_threshold:
                evaluation[fragmentId] = True
            else:
                evaluation[fragmentId] = False
                all_successful = False
        return evaluation, all_successful

class Transform2d:
    def __init__(self, v1=-1, v2=-1, score=-1, transform=np.identity(3), stitchLine=None):
        self.frame1 = v1
        self.frame2 = v2
        self.score = score
        self.transform = transform
        self.stitchLine = stitchLine

        # rank between frame1 and frame2
        self.rank = -1

class Alignment2d:
    def __init__(self, relative_transform_filename):
        self.data = []
        # for example, {'0 1': [0,1,2]} means from 0--1 to find data[0,1,2]
        self.mapIdpair2Transform = {}
        # for example, {'0 1 1': 0} means from 0--1 and 1st to find data[0]
        self.mapIdpairRank2Transform = {}
        # for example, {0: [0,1,2,100]} means from 0 to find data[0,1,2,100] in which either 0-x or x-0
        self.id2Transform = {}

        with open(relative_transform_filename) as f:
            all_line = [line.rstrip() for line in f]
            node_num = 0
            for line in all_line:
                if line[0:4] == "Node":
                    node_num+=1
                else:
                    data_str_list = line.split()
                    v1,v2,score, m1,m2,m3,m4,m5,m6,m7,m8,m9 = [t(s) for t,s in zip((int,int, float, float,float,float,float,float,float,float,float,float), data_str_list[0:12])]
                    transform = np.array([[m1,m2,m3], [m4,m5,m6], [m7,m8,m9]])

                    stitchLine = []
                    stitch_line_c = data_str_list[13:]
                    for i in range(len(stitch_line_c)//2):
                        col = float(stitch_line_c[i*2])
                        row = float(stitch_line_c[i*2+1])
                        stitchLine.append([row, col])
                    self.data.append(Transform2d(v1, v2, score, transform, stitchLine))


        self.data = sorted(self.data, key=operator.attrgetter('score'), reverse=True)
        self.data = sorted(self.data, key=operator.attrgetter('frame2'))
        self.data = sorted(self.data, key=operator.attrgetter('frame1'))

        for i, item in enumerate(self.data):
            idpair = '%d %d'%(item.frame1, item.frame2)
            if idpair in self.mapIdpair2Transform:
                self.mapIdpair2Transform[idpair].append(i)
            else:
                self.mapIdpair2Transform[idpair] = [i]
            if item.frame1 in self.id2Transform:
                self.id2Transform[item.frame1].append(i)
            else:
                self.id2Transform[item.frame1] = [i]
            if item.frame2 in self.id2Transform:
                self.id2Transform[item.frame2].append(i)
            else:
                self.id2Transform[item.frame2] = [i]

        for key,value in self.mapIdpair2Transform.items():
            for rank, index in enumerate(value):
                new_key = "%s %d"%(key, rank+1)
                self.mapIdpairRank2Transform[new_key] = index
                self.data[index].rank = rank+1

def ExpandROI(aligned_img, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col, max_expand_threshold=32):
    min_row1, min_col1, max_row1, max_col1 = bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col
    min_row2, min_col2, max_row2, max_col2 = bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col
    height, width = aligned_img.shape[:2]

    ''' 1. try to move upper and lower boundary first '''
    for i in range(1, max_expand_threshold):
        row = bbox_min_row - i
        if row < 0:
            break
        bg_pixel_count = 0
        for col in range(bbox_min_col, bbox_max_col):
            if 0 <= col < width and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, bbox_max_col - bbox_min_col)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            min_row1 = row

    for i in range(1, max_expand_threshold):
        row = bbox_max_row + i
        if row >= height:
            break
        bg_pixel_count = 0
        for col in range(bbox_min_col, bbox_max_col):
            if 0 <= col < width and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, bbox_max_col - bbox_min_col)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            max_row1 = row

    for i in range(1, max_expand_threshold):
        col = bbox_min_col - i
        if col < 0:
            break
        bg_pixel_count = 0
        for row in range(min_row1, max_row1):
            if 0 <= row < height and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, max_row1 - min_row1)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            min_col1 = col

    for i in range(1, max_expand_threshold):
        col = bbox_max_col + i
        if col >= width:
            break
        bg_pixel_count = 0
        for row in range(min_row1, max_row1):
            if 0 <= row < height and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, max_row1 - min_row1)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            max_col1 = col

    area1 = (max_col1 - min_col1) * (max_row1 - min_row1)

    ''' 2. try to move left and right boundary first '''
    for i in range(1, max_expand_threshold):
        col = bbox_min_col - i
        if col < 0:
            break
        bg_pixel_count = 0
        for row in range(bbox_min_row, bbox_max_row):
            if 0 <= row < height and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, bbox_max_row - bbox_min_row)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            min_col2 = col

    for i in range(1, max_expand_threshold):
        col = bbox_max_col + i
        if col >= width:
            break
        bg_pixel_count = 0
        for row in range(bbox_min_row, bbox_max_row):
            if 0 <= row < height and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, bbox_max_row - bbox_min_row)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            max_col2 = col

    for i in range(1, max_expand_threshold):
        row = bbox_min_row - i
        if row < 0:
            break
        bg_pixel_count = 0
        for col in range(min_col2, max_col2):
            if 0 <= col < width and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, max_col2 - min_col2)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            min_row2 = row

    for i in range(1, max_expand_threshold):
        row = bbox_max_row + i
        if row >= height:
            break
        bg_pixel_count = 0
        for col in range(min_col2, max_col2):
            if 0 <= col < width and np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / max(1, max_col2 - min_col2)
        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            max_row2 = row

    area2 = (max_col2 - min_col2) * (max_row2 - min_row2)

    if area1 > area2:
        return [min_row1, min_col1, max_row1, max_col1]
    else:
        return [min_row2, min_col2, max_row2, max_col2]
    
    
def ConvertRawStitchLine2BBoxRatio(raw_stitch_line, stitched_img, transform, offset_transform, max_expand_threshold):
    print("Converting raw stitch line to bounding box ratio...")
    
    # Transform each point
    new_stitch_line = []
    for pt in raw_stitch_line:
        row, col = pt
        new_pt = np.matmul(transform, np.array([row, col, 1]))
        new_stitch_line.append([
            new_pt[0] + offset_transform[0, 2],
            new_pt[1] + offset_transform[1, 2]
        ])

    rows, cols, _ = stitched_img.shape

    # Clamp transformed stitch line points to image bounds BEFORE computing min/max
    clamped_stitch_line = []
    for pt in new_stitch_line:
        r = np.clip(pt[0], 0, rows - 1)
        c = np.clip(pt[1], 0, cols - 1)
        clamped_stitch_line.append([r, c])

    a = np.transpose(clamped_stitch_line)

    bbox_min_row = int(np.floor(np.min(a[0])))
    bbox_min_col = int(np.floor(np.min(a[1])))
    bbox_max_row = int(np.ceil(np.max(a[0])))
    bbox_max_col = int(np.ceil(np.max(a[1])))
    
    # Clamp to image bounds
    if (bbox_min_row < 0 or bbox_min_col < 0 or
        bbox_max_row > rows or bbox_max_col > cols or
        bbox_min_row >= bbox_max_row or bbox_min_col >= bbox_max_col):

        print(
            f"[⚠️  Clamping BBox] Got bbox=[{bbox_min_row}, {bbox_min_col}, {bbox_max_row}, {bbox_max_col}] "
            f"to image shape {stitched_img.shape}"
        )

        bbox_min_row = max(0, bbox_min_row)
        bbox_min_col = max(0, bbox_min_col)
        bbox_max_row = min(rows, bbox_max_row)
        bbox_max_col = min(cols, bbox_max_col)
        
        # 🛡️ Ensure bbox is non-degenerate
        if bbox_max_row <= bbox_min_row:
            if bbox_min_row < rows - 1:
                bbox_max_row = bbox_min_row + 1
            else:
                bbox_min_row = max(0, bbox_max_row - 1)

        if bbox_max_col <= bbox_min_col:
            if bbox_min_col < cols - 1:
                bbox_max_col = bbox_min_col + 1
            else:
                bbox_min_col = max(0, bbox_max_col - 1)


        # Check for collapsed box
        if bbox_min_row >= bbox_max_row or bbox_min_col >= bbox_max_col:
            raise ValueError(
                f"[❌ Skipping] Degenerate bbox after clamping: "
                f"[{bbox_min_row}, {bbox_min_col}, {bbox_max_row}, {bbox_max_col}]"
            )

    print(f"Bounding box before expansion: [{bbox_min_row}, {bbox_min_col}, {bbox_max_row}, {bbox_max_col}]")

    # Expand the ROI safely
    try:
        new_min_row, new_min_col, new_max_row, new_max_col = ExpandROI(
            stitched_img, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col,
            max_expand_threshold=max_expand_threshold
        )
    except IndexError as e:
        raise RuntimeError(
            f"[❌ ExpandROI Failed] IndexError with input bbox: "
            f"[{bbox_min_row}, {bbox_min_col}, {bbox_max_row}, {bbox_max_col}] — {e}"
        )

    print(f"Bounding box after expansion: [{new_min_row}, {new_min_col}, {new_max_row}, {new_max_col}]")

    # Convert to normalized ratios
    new_min_row_ratio = new_min_row / rows
    new_min_col_ratio = new_min_col / cols
    new_max_row_ratio = new_max_row / rows
    new_max_col_ratio = new_max_col / cols

    return [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio]

def calculatePoseErr(gt_pose, pose):
    err = np.matmul(gt_pose, np.linalg.inv(pose))
    if np.abs(err[0, 0] - 1) < 1e-3:
        err[0, 0] = 1
    if np.abs(err[0, 0] + 1) < 1e-3:
        err[0, 0] = -1
    r_err = np.arccos(err[0, 0]) * 180 / np.pi
    t_err = np.sqrt(err[0, 2] ** 2 + err[1, 2] ** 2)

    return [r_err, t_err]