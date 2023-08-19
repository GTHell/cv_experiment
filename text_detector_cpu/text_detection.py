import cv2
import numpy as np
from openvino.inference_engine import IECore

class TextDetector:
    
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
        print(h, w)
        # out = self.net.forward(cv2.resize(frame, (w, h)))
        resized_image = cv2.resize(img, (w, h))

        # Transpose to NCHW (1, 3, H, W) = (0,1,2) + expand_dims
        input_image = np.expand_dims(resized_image.transpose(0, 1, 2), 0)

        out = self.execution_net.infer(inputs={self.input_layer: input_image})

        # The output according to the documentation (1, 1, N, 7)
        link_logits = out["model/link_logits_/add"]
        segm_logits = out["model/segm_logits/add"]

        link = link_logits[0, 0, :, :]
        print(link_logits.shape)
        print(segm_logits.shape)
        exit()

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
    weight_bin = "intel/text-detection-0003/FP16/text-detection-0003.bin"
    model_xml = "intel/text-detection-0003/FP16/text-detection-0003.xml"

    detector = TextDetector(model_xml, weight_bin)

    # test
    img = cv2.imread("IMG_1411.jpg")

    detections = detector.get_detections(img)