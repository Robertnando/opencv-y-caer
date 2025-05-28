import cv2 as cv

#Se lee la imagen del gato
img = cv.imread('Fotos/gato-1.jpeg')

#Se mustra la imagen con primero el nombre de esta y despues la imagen
#cv.imshow('Cat', img) 

capture = cv.VideoCapture('Videos/ChickenCausa.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

#cv.waitKey(0)