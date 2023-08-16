import cv2
import numpy as np
from openvino.inference_engine import IECore
class TextDetector:
    
    def __init__(self) -> None:
        ie = IECore()
        self.network = ie.read_network(
            model="models/horizontal-text-detection-0001.xml",
            weights="models/horizontal-text-detection-0001.bin",
        )
        self.execution_net = ie.load_network(self.network, "CPU")
        
        self.input_layer = next(iter(self.execution_net.input_info))
        self.output_layer = next(iter(self.execution_net.outputs))
        
        self.colors = {"red": (0, 0, 255), "green": (0, 255, 0)}

    def draw_overlay_frame(self, original_image, resized_image, predictions, res_folder = 'results/'):
        # Fetch image shapes to calculate ratio
        (real_y, real_x), (resized_y, resized_x) = original_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

        if predictions.shape[0] != 0:
            print(predictions.shape)
            # Remove confidence factor and left with n, 4 (n_size, (xmin, ymin, xmax, ymax))
            points = predictions[:, 0:4]

            # Resize the points back to original image size
            points = points * np.array([ratio_x, ratio_y, ratio_x, ratio_y])

            # convert the n, 4 to n, 4, 2 (n_size, (coord), (x, y)) using 8 points
            points = np.array([convert_to_8_points(*p) for p in points])

            # Convert to int
            points = points.astype(int)

            warped, warped_rect = warp(points, original_image)
            max_probability = 0

            # if the max prediciton probability exceeds the threshold then we return TEXT
            if len(predictions) != 0:
                for i in range(len(predictions)):
                    max_probability = max(max_probability, predictions[i][4])
            
            # Only show warped image if max probability is high enough
            if max_probability >= 0.45:
                # Put max probability on warped image
                warped = cv2.putText(warped, f"{max_probability:.2f}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["red"], 1, cv2.LINE_AA)
                
                cv2.imshow("warped", warped)
        
        for box in predictions:
            # Pick confidence factor from last place in array
            conf = box[-1]
            
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2 
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]

            original_image = cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), self.colors["green"], 5)
            
            original_image = cv2.putText(original_image, f"{conf:.2f}", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["red"], 1, cv2.LINE_AA)
        
        return original_image
        
    def draw_overlay(self, im_name, original_image, resized_image, predictions, res_folder = 'results/'):
        # Fetch image shapes to calculate ratio
        (real_y, real_x), (resized_y, resized_x) = original_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

        # Remove confidence factor and left with n, 4 (n_size, (xmin, ymin, xmax, ymax))
        points = predictions[:, 0:4]

        # Resize the points back to original image size
        points = points * np.array([ratio_x, ratio_y, ratio_x, ratio_y])

        # convert the n, 4 to n, 4, 2 (n_size, (coord), (x, y)) using 8 points
        points = np.array([convert_to_8_points(*p) for p in points])

        # Convert to int
        points = points.astype(int)

        warped, warped_rect = warp(points, original_image)
        cv2.imshow("warped", warped)
        cv2.imshow("original", original_image)
        cv2.imshow("resize", resized_image)
        cv2.waitKey(0)
        return
        # exit()
        
        for box in predictions:
            # Pick confidence factor from last place in array
            conf = box[-1]
            
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2 
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]

            original_image = cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), self.colors["green"], 5)
            
            original_image = cv2.putText(original_image, f"{conf:.2f}", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["red"], 1, cv2.LINE_AA)
        
        cv2.imwrite(res_folder + im_name, original_image)

    def test_frame(self, img, threshold=0.5):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # extracting model inputs: batch_size = 1, num_channels = 3 (RGB), height = 704, width = 704
        batch_size, num_channels, height, width = self.network.input_info[self.input_layer].tensor_desc.dims
        
        # resizing the input image to desired size
        resized_image = cv2.resize(img, (width, height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        
        # inferene on the input imahe
        result = self.execution_net.infer(inputs={self.input_layer: input_image})
        
        # output predictions
        predictions = result["boxes"]
        # removing no predictions  
        predictions_req = predictions[~np.all(predictions == 0, axis=1)]
        
        return self.draw_overlay_frame(img, resized_image, predictions_req)
        
                
        if max_probability >= threshold:
            return "TEXT PRESENT"
            
        return "TEXT NOT PRESENT"
    
    def test(self, image_path, threshold=0.5):
        im_name = image_path.split('/')[-1]
        # reading image
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # extracting model inputs: batch_size = 1, num_channels = 3 (RGB), height = 704, width = 704
        batch_size, num_channels, height, width = self.network.input_info[self.input_layer].tensor_desc.dims
        
        # resizing the input image to desired size
        resized_image = cv2.resize(img, (width, height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        
        # inferene on the input imahe
        result = self.execution_net.infer(inputs={self.input_layer: input_image})
        
        # output predictions
        predictions = result["boxes"]
        # removing no predictions  
        predictions_req = predictions[~np.all(predictions == 0, axis=1)]

        self.draw_overlay(im_name, img, resized_image, predictions_req)
        
        max_probability = 0
        
        # if the max prediciton probability exceeds the threshold then we return TEXT
        if len(predictions_req) != 0:
            for i in range(len(predictions_req)):
                max_probability = max(max_probability, predictions_req[i][4])
                
        if max_probability >= threshold:
            return "TEXT PRESENT"
            
        return "TEXT NOT PRESENT"

def warp_text_boxes(text_boxes, M):
    """
    Warp the prewarp text box rect to its warped position
    --------------------
    text_boxes: All detected text_boxes
    M: Transform matrix
    --------------------
    return: new_bounding_rect

    """

    new_boxes = []
    for box in text_boxes:

        temp_points = []
        for p in box:
            px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (
                (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
            )
            py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (
                (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
            )
            p_after = (int(px), int(py))
            temp_points.append(p_after)

        new_boxes.append(temp_points)

    new_boxes = np.array(new_boxes)
    new_boxes = np.int0(new_boxes)

    return new_boxes

def sort_points_clockwise(points):
    # sorted by all row first column which is X axis
    x_sorted = points[np.argsort(points[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # sorted each left most's Y axis
    tl, bl = left_most[np.argsort(left_most[:, 1]), :]
    tr, br = right_most[np.argsort(right_most[:, 1]), :]

    return np.array((tl, tr, br, bl))

def convert_to_8_points(xmin, ymin, xmax, ymax):
    # Create an array to store the 8 points
    points = np.zeros((8, 2))

    # Top left point
    points[0] = xmin, ymin

    # Top right point
    points[1] = xmax, ymin

    # Bottom right point
    points[2] = xmax, ymax

    # Bottom left point
    points[3] = xmin, ymax

    # Top center point
    points[4] = (xmin + xmax) / 2, ymin

    # Right center point
    points[5] = xmax, (ymin + ymax) / 2

    # Bottom center point
    points[6] = (xmin + xmax) / 2, ymax

    # Left center point
    points[7] = xmin, (ymin + ymax) / 2

    return points

def warp(text_boxes, image, new_width=1000, new_height=600, b_padding=0, debug=False):
    """
    find convex of the text detection then transform its image and text location
    -------------------------
    text_boxes: text detection text boxes
    image: origin cropped image
    -------------------------
    return: warped_image, new_bounding_rect

    """
    # Reshape to 2 dimension
    boxes = np.flip(text_boxes.reshape((-1, 2)), 0)

    # Find convex from det craft boxes
    hull = cv2.convexHull(boxes)
    # Reshape to have [n, 2]
    hull = hull.reshape((-1, 2))
    hull = np.int0(hull)

    # find rotated rect and sort tl, tr, br, bl
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    sorted = sort_points_clockwise(box)

    # get dst perspective and warp
    dst = np.array(
        [
            [0, 0],
            [new_width - 1, 0],
            [new_width - 1, new_height - 1],
            [0, new_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(np.float32(sorted), dst)

    warped = cv2.warpPerspective(image, M, (new_width, new_height))

    padding = np.ones((b_padding, 1000, 3), dtype="uint8")
    warped = np.vstack((warped, padding))

    warped_text_boxes = warp_text_boxes(text_boxes, M)
    warped_bounding_text_boxes = [cv2.boundingRect(box) for box in warped_text_boxes]
    # dst_points = np.array(dst_points)
    # dst = np.load('id_card_dst_template.npy')
    # M, mask = cv2.findHomography(boxes, dst, cv2.RANSAC, 5)
    # warped = cv2.warpPerspective(image, M, (new_width, new_height))

    if debug:
        # for box in warped_text_boxes:
        #     cv2.drawContours(warped, [box], 0, (255, 0, 255), 3)

        for box in warped_bounding_text_boxes:
            cv2.rectangle(
                warped,
                (box[0], box[1]),
                (box[0] + box[2], box[1] + box[3]),
                (255, 0, 0),
                1,
            )

        cv2.namedWindow("Hull points on card", cv2.WINDOW_NORMAL)
        cv2.namedWindow("origin image", cv2.WINDOW_NORMAL)
        cv2.imshow("Hull points on card", warped)
        cv2.imshow("origin image", image)
        cv2.waitKey(0)

    return warped, warped_bounding_text_boxes
    
    
if __name__ == "__main__":
    obj = TextDetector()
    threshold = 0.5
    
    import time
    t1 = time.time()
    # Inferenceing image containing text (based on threshold)
    # text_image_path = "ted_lasso.jpeg"
    # print(obj.test(text_image_path, threshold))
    
    # Inferenceing image without text (based on threshold)
    # non_text_image_path = "knight-berserk-desktop-background.jpg"
    # print(obj.test(non_text_image_path, threshold))

    # test = "IMG_1411.jpg"
    # print(obj.test(test, threshold))

    # test = "adjusted.jpg"
    # print(obj.test(test, threshold))

    test = "received.jpg"
    print(obj.test(test, threshold))
    print(time.time() - t1)
        