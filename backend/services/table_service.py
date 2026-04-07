import cv2

def extract_table_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,255,1,1,11,2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    cells = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:
            cells.append((x,y,w,h))

    return cells

cells = extract_table_cells(image)

table_data = []
for (x,y,w,h) in cells:
    crop = image[y:y+h, x:x+w]
    text = extract_text(crop)
    table_data.append(text)