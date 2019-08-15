import cv2

cap = cv2.VideoCapture(0)
i = 0
while (1):
    ret, frame = cap.read()
    cv2.imwrite('F:/faceImage/' + str(i) + '.jpg', frame)
    # 如果按下Esc键或拍够600张，则退出
    if cv2.waitKey(1) == 27 or i==599:
        break
    i += 1
    cv2.imshow('capture', frame)
cap.release()
cv2.destroyAllWindows()