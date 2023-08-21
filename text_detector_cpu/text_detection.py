import cv2
import numpy as np
from openvino.inference_engine import IECore
import networkx as nx

from inference import warp

class TextDetectorPixelLink:
    
    def load_ie_model(self, model_xml, weight_bin, device='CPU') -> None:
        """Loads a model in the Inference Engine format"""

        ie = IECore()

        self.network = ie.read_network(
            model=model_xml,
            weights=weight_bin,
        )

        self.execution_net = ie.load_network(self.network, "CPU")

        self.input_layer = next(iter(self.execution_net.input_info))
        self.output_layer = next(iter(self.execution_net.outputs))
        self.network.batch_size = 1

    def __init__(self, model_xml, weight_bin, conf=.6, device='CPU', ext_path=''):
        self.load_ie_model(model_xml, weight_bin, device)
        self.confidence = conf
        self.expand_ratio = (1.1, 1.05)
    
    def get_detections(self, frame):
        """Returns all detections on frame"""
        b, h, w, c = self.network.input_info[self.input_layer].input_data.shape

        resized_image = cv2.resize(img, (w, h))

        # Transpose to NCHW (1, 3, H, W) = (0,1,2) + expand_dims
        input_image = np.expand_dims(resized_image.transpose(0, 1, 2), 0)

        out = self.execution_net.infer(inputs={self.input_layer: input_image})

        # The output according to the documentation (1, 1, N, 7)

        # The output of link logits is (N, H, W, C(16))
        # It indicate the probability of the pixel being a link, 
        # a localization of the link, and the direction of the link (8 directions) * 2 softmax
        # kLocOutput
        link_logits = out["model/link_logits_/add"]

        # The output of segm logits is (1, H, W, 2)
        # It indicate the classification of the pixel being a text or not
        # kClsOutput
        segm_logits = out["model/segm_logits/add"]

        # Post process to get contours of the text
        # Contour can be obtained directly from segm_logits without link_logits
        cnts = self.__post_processing(link_logits, segm_logits, resize=resized_image)

        # Extract the text region from the frame
        cnts = self.__extract_text_region(frame, cnts)

        return cnts
    
    def __extract_text_region(self, frame, cnts):

        # Construct input shape of 320 by 192
        input_image = np.zeros([192, 320, 3], dtype=np.uint8)


        (real_y, real_x), (resized_y, resized_x) = frame.shape[:2], input_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

        # TODO: Extract the text region from the frame
        # Use the cnts to extract the text region

        # Reshape cnts from (N, 4, 1, 2) to (N, 4, 2)
        cnts = cnts.reshape([cnts.shape[0], 4, 2])

        # Scale the cnts back to the original size
        cnts = cnts * np.array([ratio_x, ratio_y])
        cnts = cnts.astype(np.int64)

        return cnts
    
    def test(self, frame, cnts):
        warped, bounding_rects = warp(cnts, frame, b_padding=100)

        bounding_rects = np.array(bounding_rects)
        for rect in bounding_rects:
            x, y, w, h = rect
            cv2.rectangle(warped, (x, y), (x + w, y + h), (255, 0, 0), 1)

        rotated_bounding_boxes = np.zeros([cnts.shape[0], 4, 2], dtype=np.int64)
        print(rotated_bounding_boxes.shape)

        for cnt in cnts:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            # Get the rotated bounding box
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
             
            # convert all coordinates floating point values to int
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        
        cv2.imshow("image", frame)
        cv2.imshow("warped", warped)
        cv2.waitKey(0)
    
    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True) 
    
    def softmax2(self, data):
        rdata = np.array(data)
        last_dim = 2
        for i in range(0, len(rdata), last_dim):
            m = np.max(rdata[i:i+last_dim])
            rdata[i:i+last_dim] = np.exp(rdata[i:i+last_dim] - m)
            s = np.sum(rdata[i:i+last_dim])
            rdata[i:i+last_dim] = rdata[i:i+last_dim] / s
            
        return rdata

    def line(self, p1, p2):

        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def line_intersection(self, l1, l2):

        # l1 is a numpy array with shape 2, 2 with each row containing x, y

        d  = l1[0] * l2[1] - l1[1] * l2[0]
        dx = l1[2] * l2[1] - l1[1] * l2[2]
        dy = l1[0] * l2[2] - l1[2] * l2[0]
        if d != 0:
            x = (dx / d).astype(np.int64)
            y = (dy / d).astype(np.int64)
            return [x,y]
        else:
            return None
    
    def get_rotated_bbox(self, contours):
        return_contours = []

        for i in range(len(contours)):

            cnt = np.array(contours[i]).astype(np.int64)
            rect = cv2.minAreaRect(cnt)
            return_contours.append(cv2.boxPoints(rect).astype(np.int64).reshape([4, 1, 2]))

        return return_contours

        #Creates bounding box using minAreaRect, gets the points, the rectangle need not be parallel to x-axis

    def remove_small_boxes(self, contours, min_area, max_area=None):
        """
        input - contour, min_area, max_are
        return - thresholded contour
        """

        contours = self.get_rotated_bbox(contours)

        return_contours = []

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > min_area:
                if max_area!=None:
                    if area < max_area:
                        return_contours.append(contours[i])
                else:
                    return_contours.append(contours[i])

        return return_contours

    def inside_point(self, point, rect):

        # point is a list (x, y)
        # rect is a contour with shape [4, 2]

        rect = rect.reshape([4, 1, 2]).astype(np.int64)

        dist = cv2.pointPolygonTest(rect,(point[0], point[1]),True)

        if dist>0:
            # print(dist)
            return True
        else:
            return False

    def intersection_union(self, cont1, cont2):
        # Assuming both contours are rectangle with shape [4, 1, 2]

        cont1 = cont1.reshape([cont1.shape[0], 2]).astype(np.float64)
        cont2 = cont2.reshape([cont2.shape[0], 2]).astype(np.float64)

        intersection_points = []

        line_i = [self.line(cont1[i], cont1[(i+1)%4]) for i in range(4)]
        line_j = [self.line(cont2[j], cont2[(j+1)%4]) for j in range(4)]

        min_i_x = [np.min(cont1[[i, (i+1)%4], 0]) for i in range(4)]
        max_i_x = [np.max(cont1[[i, (i+1)%4], 0]) for i in range(4)]
        min_j_x = [np.min(cont2[[j, (j+1)%4], 0]) for j in range(4)]
        max_j_x = [np.max(cont2[[j, (j+1)%4], 0]) for j in range(4)]

        min_i_y = [np.min(cont1[[i, (i+1)%4], 1]) for i in range(4)]
        max_i_y = [np.max(cont1[[i, (i+1)%4], 1]) for i in range(4)]
        min_j_y = [np.min(cont2[[j, (j+1)%4], 1]) for j in range(4)]
        max_j_y = [np.max(cont2[[j, (j+1)%4], 1]) for j in range(4)]

        for i in range(4):
            if self.inside_point(cont1[i], cont2):
                intersection_points += [cont1[i]]
            if self.inside_point(cont2[i], cont1):
                intersection_points += [cont2[i]]

        for i in range(4):
            for j in range(4):
                point = self.line_intersection(line_i[i], line_j[j])
                cond1 = point is not None
                if not cond1:
                    continue
                cond2 = point[0] >= min_i_x[i] and point[0] >= min_j_x[j] and point[0] <= max_i_x[i] and point[0] <= max_j_x[j]
                if not cond2:
                    continue
                cond3 = point[1] >= min_i_y[i] and point[1] >= min_j_y[j] and point[1] <= max_i_y[i] and point[1] <= max_j_y[j]
                if not cond3:
                    continue
                intersection_points += [point]
        
        if len(intersection_points) < 3:
            return 1, 0

        contour = np.array(intersection_points).reshape([len(intersection_points), 1, 2]).astype(np.int64)
        contour = cv2.convexHull(contour)
        intersection_val = cv2.contourArea(contour)
        if intersection_val!=0:
            union_val = (cv2.contourArea(cont1.reshape([4, 1, 2]).astype(np.int64))+cv2.contourArea(cont2.reshape([4, 1, 2]).astype(np.int64)) - intersection_val)
            return union_val, intersection_val
        else:
            return 1, 0

    def overlap_remove(self, contour,threshold):

        contour = np.array(contour)
        to_remove = np.zeros([contour.shape[0]])

        for i in range(contour.shape[0]):

            for j in range(i+1,contour.shape[0]):

                if to_remove[j] == 1:
                    continue

                union, inter = self.intersection_union(contour[i], contour[j])
                cnt_a_1, cnt_a_2 = cv2.contourArea(contour[i]), cv2.contourArea(contour[j])	
                
                if (inter/cnt_a_1) > threshold:
                    if (inter/cnt_a_2) > threshold:
                        if cnt_a_2 > cnt_a_1:
                            to_remove[i] = 1
                        else:
                            to_remove[j] = 1
                    else:
                        to_remove[i] = 1
                elif (inter/cnt_a_2) > threshold:
                    to_remove[j] = 1

        return contour[np.where(to_remove == 0)[0]]

    def __post_processing(self, link_data, segm_data, cls_conf_threshold=0.8, link_conf_threshold=0.8, resize=None, use_link=False, real_target=None):
        min_area = 50
        min_height = 10

        # Compute softmax on the last dimension of the link_data
        # link_data = self.softmax(link_data)

        # print(link_data[0, 0, 0, :])
        
        # out = self.slice_and_get_second_channel(softmax_link_y)
        # print(out.shape)

        segm_data = self.softmax(segm_data)

        # Get text/non-text mask from segm_data from second channel
        # This will mask out the non-text regions to 0 and text regions to 1
        pixel_pred = (segm_data[:, :, :, 1] > cls_conf_threshold).astype(np.float32)

        print("Shape:", pixel_pred.shape)

        # Reshape from (1, 192, 320) to 2D
        pixel_pred = np.reshape(pixel_pred, (192, 320))

        if use_link:
            link_pred = np.reshape(link_data, (192, 320, 16))
            link_pred = np.exp(link_pred)
            link_predicted_8_channel = np.zeros([link_pred.shape[0], link_pred.shape[1], 8])	

            for i in range(8):
                
                link_pred[:, :, 2*i:2*i+2] = link_pred[:, :, 2*i:2*i+2]/np.sum(link_pred[:, :, 2*i:2*i+2], axis=2)[:, :, None]
                link_predicted_8_channel[:, :, i] = (link_pred[:, :, 2*i+1] > link_conf_threshold).astype(np.float32)

                #For each in link_predicted_8_channel, there are 2 channels of link_pred, normalised mean given to all three
                #If this is above threshold, then make the channel value 1, else 0


        # Initialization of some useful values
        image_size = pixel_pred.shape
        target = np.zeros([image_size[0], image_size[1], 3])
        target[:, :, 0] = pixel_pred*255
        target = target.astype(np.uint8)

        # Get the contours
        if use_link:
            # Initialization of Graph
            moves = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            edges = np.zeros(link_predicted_8_channel.shape)
            row, column = np.where(pixel_pred==1)
            g = nx.Graph()
            g.add_nodes_from(zip(row,column))
            link_predicted_8_channel = link_predicted_8_channel*pixel_pred[:, :, None]
            link_boundary = (np.any(link_predicted_8_channel==0, axis=2).astype(np.float32)*pixel_pred*255).astype(np.uint8)

            # Processing to allow us to use nx.connected_components
            pixel_pred = np.pad(pixel_pred, 1, 'constant', constant_values = 0)
            for i in range(8):
                x,y = moves[i]
                edges[:,:,i] = link_predicted_8_channel[:,:,i]*pixel_pred[1+x:1+x+image_size[0], 1+y:1+y+image_size[1]]

            for i in range(8):
                row, column = np.where(edges[:,:,i]==1)
                g_edges1 = list(zip(row,column))
                g_edges2 = list(zip(row+moves[i][0],column+moves[i][1]))
                g.add_edges_from(zip(g_edges1,g_edges2))

            # Set of connected components

            connected = list(nx.connected_components(g))

            # Converting to fit the countour half of the program

            for i in range(len(connected)):
                connected[i] = np.flip(np.array(list(connected[i])).reshape([len(connected[i]), 1, 2]), axis=-1)
        else:
            # Get the contours from the pixel_pred mask without link_data
            connected, _ = cv2.findContours(pixel_pred.copy().astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        

        connected = self.remove_small_boxes(connected,min_area=min_area)
        connected = self.overlap_remove(connected, 0.2)

        return connected

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)
        return detections
    
    def draw_frame(self, frame, detections):
        """Draws detections on the frame"""

        for detection in detections:
            left, top, right, bottom = detection[0]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, str(round(detection[1], 2)), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

if __name__ == "__main__":

    # Init the detector
    weight_bin = "intel/text-detection-0003/FP32/text-detection-0003.bin"
    model_xml = "intel/text-detection-0003/FP32/text-detection-0003.xml"

    detector = TextDetectorPixelLink(model_xml, weight_bin)

    # test
    img = cv2.imread("IMG_1411.jpg")
    # crop to center. the image size is H:4032, W:2268
    # For larger image, the text detection is not working
    # It's better to first crop the image to the center or use object detection to detect the text region
    # Then resize resize to even smaller size to speed up the text detection

    img = img[1500:2700, 400:1900, :]
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # print(img.shape)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    # img = cv2.imread("ted_lasso.jpeg")
    # cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Original image", img)

    import time
    t1 = time.time()
    detections = detector.get_detections(img)

    # Test
    print("Time:", time.time() - t1)
    detector.test(img, detections)
