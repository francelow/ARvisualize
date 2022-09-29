import os
import cv2 as cv2
import numpy as np

#-----------------------Generating ArUco markers in OpenCV-----------------------
def generateArucoMarkers():
    # Load the predefined dictionary
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Generate the marker
    markerImage1 = np.zeros((200, 200), dtype=np.uint8)
    markerImage2 = np.zeros((200, 200), dtype=np.uint8)
    markerImage3 = np.zeros((200, 200), dtype=np.uint8)
    markerImage4 = np.zeros((200, 200), dtype=np.uint8)

    markerImage1 = cv2.aruco.drawMarker(dictionary, 33, 200, markerImage1, 1)
    markerImage2 = cv2.aruco.drawMarker(dictionary, 34, 200, markerImage2, 1)
    markerImage3 = cv2.aruco.drawMarker(dictionary, 35, 200, markerImage3, 1)
    markerImage4 = cv2.aruco.drawMarker(dictionary, 36, 200, markerImage4, 1)

    cv2.imwrite("Markers/marker33.png", markerImage1)
    cv2.imwrite("Markers/marker34.png", markerImage2)
    cv2.imwrite("Markers/marker35.png", markerImage3)
    cv2.imwrite("Markers/marker36.png", markerImage4)

#---------------------------------------------------------------------------------

#-----------------------------Detecting Aruco markers-----------------------------

def loadOverlayImages(path):
    myList = os.listdir(path)
    print("myList: ", myList)
    noOfMarkers = len(myList)
    print("Total number of markers detected: ", noOfMarkers)
    augDict = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f"{path}/{imgPath}")
        augDict[key] = imgAug
    return augDict

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f"DICT_{markerSize}X{markerSize}_{totalMarkers}")
    arucoDict = cv2.aruco.Dictionary_get(key)
    arucoParam = cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    # print(markerIds)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, markerCorners)
    return [markerCorners, markerIds]

def augmentArucoSquares():
    cap = cv2.VideoCapture(0)
    # Load overlay images from Markers folder
    augDict = loadOverlayImages("Gallery")

    while True:
        success, img = cap.read()
        foundArucos = findArucoMarkers(img)
        # Loop through all markers and augment each one
        if len(foundArucos[0]) != 0:
            for bbox, id in zip(foundArucos[0], foundArucos[1]):
                # Check if the overlay image is available for specific aruco id
                if int(id) in augDict:
                    img = augmentAruco(bbox, id, img, augDict[int(id)])

        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break


def augmentAruco(bbox, id, img, overlayImg, drawId=True):
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    ht, wt, c = overlayImg.shape
    pt1 = np.array([tl, tr, br, bl])
    pt2 = np.float32([[0,0], [wt,0], [wt,ht], [0, ht]])
    # Find homography
    matrix, _ = cv2.findHomography(pt2, pt1)
    # Warp perspective
    imgOut = cv2.warpPerspective(overlayImg, matrix, (img.shape[1], img.shape[0]))
    # print('imgOut: ', imgOut)
    cv2.fillConvexPoly(img, pt1.astype(int), (0, 0, 0))
    imgOut = img + imgOut

    # Draw id
    # print("id: ", str(id))
    # print("tl: ", tl[0])
    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 0, 255), thickness=2)

    return imgOut

def main():
    cap = cv2.VideoCapture(0)
    # Load overlay images from Markers folder
    augDict = loadOverlayImages("Markers")

    while True:
        success, img = cap.read()
        foundArucos = findArucoMarkers(img)
        # Loop through all markers and augment each one
        if len(foundArucos[0]) != 0:
            for bbox, id in zip(foundArucos[0], foundArucos[1]):
                # Check if the overlay image is available for specific aruco id
                if int(id) in augDict:
                    img = augmentAruco(bbox, id, img, augDict[int(id)])


        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break

main()

#---------------------------------------------------------------------------------


#-------------------------------AR application------------------------------------
# Calculate Homography
# h, status = cv.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
# warped_image = cv.warpPerspective(im_src, h, (frame.shape[1], frame.shape[0]))

# Prepare a mask representing region to copy from the warped image into the original frame.
# mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8);
# cv.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv.LINE_AA);

# Erode the mask to not copy the boundary effects from the warping
# element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3));
# mask = cv.erode(mask, element, iterations=3);

# Copy the mask into 3 channels.
# warped_image = warped_image.astype(float)
# mask3 = np.zeros_like(warped_image)
# for i in range(0, 3):
#     mask3[:, :, i] = mask / 255

# Copy the masked warped image into the original frame in the mask region.
# warped_image_masked = cv.multiply(warped_image, mask3)
# frame_masked = cv.multiply(frame.astype(float), 1 - mask3)
# im_out = cv.add(warped_image_masked, frame_masked)