import streamlit as st  
import easyocr
import numpy as np
import math
import io
import xlsxwriter
import imutils
import pandas as pd
from PIL import Image
import cv2
reader = easyocr.Reader(['en'])
st.title("Hello")
@st.cache_data
def convert_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    adjusted = cv2.convertScaleAbs(image, alpha=1.3, beta=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adjusted_gray=cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    Result = reader.readtext(adjusted_gray)

    wordlist=[]
    for word in Result:
        y1= int(word[0][0][1])
        y2 = int(math.ceil(word[0][2][1]))
        x1=int(word[0][0][0])
        x2=int(math.ceil(word[0][2][0]))
        crop_gray=adjusted_gray[y1:y2,x1:x2].copy()
    
        blurred = cv2.GaussianBlur(crop_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist = (dist * 255).astype("uint8")
        dist = cv2.threshold(dist, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        kernel_erosion = np.ones((3, 3), np.uint8)
        dist = cv2.erode(dist, kernel_erosion, iterations=1)
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final = cv2.dilate(dist, kernel_dilation,iterations=2)
        kernel_closing = np.ones((3, 3), np.uint8)
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_closing, iterations=3)
        extracted_word = reader.readtext(final,detail=0)

        wordlist.append([y1,x1,y2,x2,extracted_word])

    sorted_wordlist=sorted(wordlist,key=lambda x: (x[0],x[1]))
    data = []
    row=[]
    for i in range (len(sorted_wordlist)):    
        if i == 0:
            row.append(sorted_wordlist[i])
        else:
            h = sorted_wordlist[i-1][2]-sorted_wordlist[i-1][0]
            if sorted_wordlist[i][0]<= (sorted_wordlist[i-1][2]-0.2*h) and sorted_wordlist[i][0]<= row[0][2]:
                row.append(sorted_wordlist[i])
            else:
                sorted_row=sorted(row,key = lambda x:(x[1]))
                newrow=[]
                for word in sorted_row:
                    newrow.append(word[4][0])
                data.append(newrow)
                row.clear()
                row.append(sorted_wordlist[i])
        if i == len(sorted_wordlist)-1:
            sorted_row=sorted(row,key = lambda x:(x[1]))
            newrow=[]
            for word in sorted_row:
                newrow.append(word[4][0])
            data.append(newrow)
            row.clear()
    df = pd.DataFrame(data)
    df.drop(df.index[0])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1',index=False,header=False)
        writer.save()
    return buffer
file = st.file_uploader("Please upload an image",type=["jpg"])
if file is None:
    st.text('Please upload an Image')
else:
    st.text("File uploaded successfully!")
    df_xlsx = convert_image(file)
  
st.download_button(label='Download XlSX',data = df_xlsx,file_name='extracted.xlsx',mime="application/vnd.ms-excel")