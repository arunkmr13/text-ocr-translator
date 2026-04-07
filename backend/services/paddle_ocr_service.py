from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # auto handles many langs

def extract_text_blocks(image):
    result = ocr.ocr(image, cls=True)

    blocks = []
    for line in result:
        for word in line:
            box = word[0]
            text = word[1][0]

            x1 = int(box[0][0])
            y1 = int(box[0][1])
            x2 = int(box[2][0])
            y2 = int(box[2][1])

            blocks.append({
                "text": text,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1
            })

    return blocks