import cv2
import csv
import numpy as np
import os
from .face_blend_common import FaceBlendCommon


class FaceFilter:
    def __init__(self, filter_dir_path: str="../assets/filters"):
        self.filter_dir_path = filter_dir_path
        self.filters_configs = {
            'anonymous':
                [{'path': os.path.join(self.filter_dir_path , "anonymous.png"),
                'anno_path': os.path.join(self.filter_dir_path , "anonymous_annotations.csv"),
                'morph': True, 'animated': False, 'has_alpha': True}],
            'dog':
                [{'path': os.path.join(self.filter_dir_path, "dog-ears.png"),
                'anno_path': os.path.join(self.filter_dir_path, "dog-ears_annotations.csv"),
                'morph': False, 'animated': False, 'has_alpha': True},
                {'path': os.path.join(self.filter_dir_path, "dog-nose.png"),
                'anno_path': os.path.join(self.filter_dir_path, "dog-nose_annotations.csv"),
                'morph': False, 'animated': False, 'has_alpha': True}],
            'cat':
                [{'path': os.path.join(self.filter_dir_path, "cat-ears.png"),
                'anno_path': os.path.join(self.filter_dir_path, "cat-ears_annotations.csv"),
                'morph': False, 'animated': False, 'has_alpha': True},
                {'path': os.path.join(self.filter_dir_path, "cat-nose.png"),
                'anno_path': os.path.join(self.filter_dir_path, "cat-nose_annotations.csv"),
                'morph': False, 'animated': False, 'has_alpha': True}],
            'jason-joker':
                [{'path': os.path.join(self.filter_dir_path, "jason-joker.png"),
                'anno_path': os.path.join(self.filter_dir_path, "jason-joker_annotations.csv"),
                'morph': True, 'animated': False, 'has_alpha': True}],
        }
        self.loaded_filters = {}
        for filter in self.filters_configs.keys():
            self.loaded_filters[filter] = self.load_filter(filter)

    @staticmethod
    def load_filter_img(img_path: str, has_alpha: bool):
        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha = None
        if has_alpha:
            b, g, r, alpha = cv2.split(img)
            img = cv2.merge((b, g, r))
        return img, alpha

    @staticmethod
    def load_landmarks(annotation_file: str):
        with open(annotation_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            points = {}
            for i, row in enumerate(csv_reader):
                # skip head or empty line if it's there
                try:
                    x, y = int(row[1]), int(row[2])
                    points[row[0]] = (x, y)
                except ValueError:
                    continue
            return points
    
    @staticmethod
    def find_convex_hull(points):
        hull = []
        hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
        addPoints = [
            [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
            [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
            [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
            [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
            [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
        ]
        hullIndex = np.concatenate((hullIndex, addPoints))
        for i in range(0, len(hullIndex)):
            hull.append(points[str(hullIndex[i][0])])

        return hull, hullIndex

    def load_filter(self, filter_name="anonymous"):
        filters = self.filters_configs[filter_name]
        multi_filter_runtime = []

        for filter in filters:
            temp_dict = {}

            img1, img1_alpha = FaceFilter.load_filter_img(filter['path'], filter['has_alpha'])

            temp_dict['img'] = img1
            temp_dict['img_a'] = img1_alpha

            points = FaceFilter.load_landmarks(filter['anno_path'])

            temp_dict['points'] = points

            if filter['morph']:
                # Find convex hull for delaunay triangulation using the landmark points
                hull, hullIndex = FaceFilter.find_convex_hull(points)

                # Find Delaunay triangulation for convex hull points
                sizeImg1 = img1.shape
                rect = (0, 0, sizeImg1[1], sizeImg1[0])
                dt = FaceBlendCommon.calculateDelaunayTriangles(rect, hull)

                temp_dict['hull'] = hull
                temp_dict['hullIndex'] = hullIndex
                temp_dict['dt'] = dt

                if len(dt) == 0:
                    continue

            if filter['animated']:
                filter_cap = cv2.VideoCapture(filter['path'])
                temp_dict['cap'] = filter_cap

            multi_filter_runtime.append(temp_dict)

        return filters, multi_filter_runtime
    
    def apply_filter(self, image: np.ndarray, bboxs, all_landmarks, filter_type: str="anonymous"):
        # add 2 points: 68 -> 70 points
        for i in range(len(all_landmarks)):
            all_landmarks[i] = np.vstack([all_landmarks[i], np.array([all_landmarks[i][0][0], int(bboxs[i][1])])])
            all_landmarks[i] = np.vstack([all_landmarks[i], np.array([all_landmarks[i][16][0], int(bboxs[i][1])])])
            all_landmarks[i] = all_landmarks[i].tolist()

        filters, multi_filter_runtime = self.loaded_filters[filter_type]

        for landmarks in all_landmarks:
            for idx, filter in enumerate(filters):
                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime["img"]
                points1 = filter_runtime["points"]
                img1_alpha = filter_runtime["img_a"]

                if filter['morph']:
                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']
                    hull1 = filter_runtime['hull']

                    # create copy of frame
                    warped_img = np.copy(image)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(landmarks[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])

                        FaceBlendCommon.warpTriangle(img1, warped_img, t1, t2)
                        FaceBlendCommon.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(image, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                else:
                    dst_points = [landmarks[int(list(points1.keys())[0])], landmarks[int(list(points1.keys())[1])]]
                    tform = FaceBlendCommon.similarityTransform(list(points1.values()), dst_points)
                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (image.shape[1], image.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (image.shape[1], image.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(image, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                image = output = np.uint8(output)
        return image

if __name__ == "__main__":
    from face_predictor import FacePredictor
    from face_keypoints_predictor import FaceKeypointsPredictor

    # Init models
    face_predictor = FacePredictor()
    keypoints_predictor = FaceKeypointsPredictor("../weights/mobilenetv3.onnx")

    image = cv2.imread("../assets/images/a.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxs = face_predictor.predict(image)
    landmarks = keypoints_predictor.detect_landmarks(image, bboxs)

    print(bboxs)

    print(len(landmarks))

    facefilter = FaceFilter()
    image = facefilter.apply_filter(image, bboxs, landmarks, filter_type="jason-joker")

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 600)
    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()