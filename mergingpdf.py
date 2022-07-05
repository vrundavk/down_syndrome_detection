from PyPDF2 import PdfFileMerger
def process(uname):
    pdfs = [uname+'final.pdf', uname+'output.pdf']

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(uname+"result.pdf")
    merger.close()

