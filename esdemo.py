import os
from PIL import Image
from fpdf import FPDF
import glob
pdf = FPDF()
#sdir = "imageFolder/"
def process(uname,path):
    
    fname = path
    if os.path.exists(fname):
        cover = Image.open(fname)
        w,h = cover.size
        pdf = FPDF(unit = "pt", format = [w,h])
        image = fname
        pdf.add_page()
        pdf.image(image,0,0,w,h)
        pdf.output(uname+"output.pdf", "F")
        print("done")

