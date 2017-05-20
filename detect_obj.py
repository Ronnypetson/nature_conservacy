import cv2
#reading the image
image = cv2.imread("train/target/img_00003.jpg")
edged = cv2.Canny(image, 10, 250)
#cv2.imshow("Edges", edged)
cv2.imwrite("train/target/edged.jpg",edged)
#cv2.waitKey(0)

#applying closing function
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#cv2.imshow("Closed", closed)
cv2.imwrite("train/target/closed.jpg",closed)
#cv2.waitKey(0)

#finding_contours
_,cnts,_ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
idx = 0
for c in cnts:
    #contour = numpy.array([[[0,0]], [[10,0]], [[10,10]], [[5,4]]])
    #area = cv2.contourArea(contour)
    peri = cv2.arcLength(c, True)
    #peri = 100
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv2.imwrite("train/target/output.jpg",image)
    #
    x,y,w,h = cv2.boundingRect(c)
    if w > 26 and h > 26:
        idx+=1
        new_img=image[y:y+h,x:x+w]
        cv2.imwrite('train/target/'+ str(idx) + '.jpg', new_img)
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
