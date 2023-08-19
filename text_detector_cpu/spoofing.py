import cv2
import numpy as np
from openvino.inference_engine import IECore

class SpoofDetector:
    
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
        _, _, h, w = self.network.input_info[self.input_layer].input_data.shape
        # out = self.net.forward(cv2.resize(frame, (w, h)))
        resized_image = cv2.resize(img, (w, h))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

        out = self.execution_net.infer(inputs={self.input_layer: input_image})

        # Output is a softmax

        return out["Softmax_440"]

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        outputs = [self.execution_net.forward(frame) for frame in batch]
        return outputs

class FaceDetector:
    
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
        _, _, h, w = self.network.input_info[self.input_layer].input_data.shape
        print(h, w)
        # out = self.net.forward(cv2.resize(frame, (w, h)))
        resized_image = cv2.resize(img, (w, h))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

        out = self.execution_net.infer(inputs={self.input_layer: input_image})

        # The output according to the documentation (1, 1, N, 7)
        detection_out = out["detection_out"]

        detections = self.__decode_detections(detection_out, frame.shape)

        return detections

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
    weight_bin = "intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin"
    model_xml = "intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"

    detector = FaceDetector(model_xml, weight_bin)
    weight_bin = "models/MN3_antispoof.bin"
    model_xml = "models/MN3_antispoof.xml"
    spoof_detector = SpoofDetector(model_xml, weight_bin)

    # test
    # img = cv2.imread("/Users/sith007/Downloads/bro_hok_2.jpeg")
    img = cv2.imread("/Users/sith007/Downloads/right-me.jpg")
    img = cv2.imread("IMG_1411.jpg")

    # Apply median blur to the image to remove noise
    img = cv2.medianBlur(img, 9)

    detections = detector.get_detections(img)

    face1 = img[detections[0][0][1]:detections[0][0][3], detections[0][0][0]:detections[0][0][2]]
    # img = detector.draw_frame(img, detections)
    prob = spoof_detector.get_detections(face1)

    print(prob.shape)
    print(prob[0])
    real_prob = prob[0][0]
    fake_prob = prob[0][1]

    # Draw the probability on the image
    cv2.putText(face1, "Real: " + str(round(real_prob, 2)) + "%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(face1, "Fake: " + str(round(fake_prob, 2)) + "%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    
    cv2.imshow("img1", img)
    cv2.imshow("Face1", face1)
    cv2.waitKey(0)