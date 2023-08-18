from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, RIL
from tesserocr import get_languages
import cv2
import tesserocr
images = []

print(get_languages('/usr/local/Cellar/tesseract/5.3.2_1/share/tessdata'))
print(tesserocr.tesseract_version())  # print tesseract-ocr version
print(tesserocr.get_languages())

with PyTessBaseAPI(psm=PSM.AUTO_OSD, path='/usr/local/Cellar/tesseract/5.3.2_1/share/tessdata') as api:
    image = cv2.imread("IMG_1411.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # crop to center of image with shape of 4000x2000
    image = image[1600:2600, 400:1900]

    pil_img = Image.fromarray(image)
    api.SetImage(pil_img)

    it = api.AnalyseLayout()
    orientation, direction, order, deskew_angle = it.Orientation()
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
    print('Found {} textline image components.'.format(len(boxes)))

    for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object
        # box is a dict with x, y, w and h keys
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        ocrResult = api.GetUTF8Text()
        conf = api.MeanTextConf()
        print('Box[{}] x={}, y={}, w={}, h={}, '.format(i, box['x'], box['y'], box['w'], box['h']))

        x, y , w, h = box['x'], box['y'], box['w'], box['h']
        cv2.rectangle(image, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (0, 255, 0), 2)

        # images.append = image[y:y+h, x:x+w]
    cv2.imshow('textline[]'.format(i), image)
    cv2.waitKey(0)
        # print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
        #       "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))