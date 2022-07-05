import cv2,os
def process(fname):
    print("filename==",fname)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    path=fname
    img=cv2.imread(path)
    desize=(500,500)
    img=cv2.resize(img,desize)
    cv2.imshow("input",img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
    cv2.imshow('output',img)
    cv2.waitKey(0)
#process("E:\\Project\\newcode\\sample\\0331.jpg")
    
