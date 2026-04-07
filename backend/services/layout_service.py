import pytesseract
from pytesseract import Output

def detect_layout(image):
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    blocks = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            blocks.append({
                "text": data['text'][i],
                "x": data['left'][i],
                "y": data['top'][i],
                "w": data['width'][i],
                "h": data['height'][i]
            })
    return blocks
