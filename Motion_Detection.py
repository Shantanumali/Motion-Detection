import cv2

cap = cv2.VideoCapture(0)
ret, f1 = cap.read()
ret, f2 = cap.read()

while cap.isOpened():
    difference = cv2.absdiff(f1, f2)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contou in contours:
        #x,y is the top left contour point
        (x, y, w, h) = cv2.boundingRect(contou)

        if cv2.contourArea(contou) < 10000:
            continue
        cv2.rectangle(f1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(f1, "Alert!", (20, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Motion Detection", f1)
    f1 = f2
    ret, f2 = cap.read()

    if cv2.waitKey(40)  == 27:
        break

cv2.destroyAllWindows()
cap.release()