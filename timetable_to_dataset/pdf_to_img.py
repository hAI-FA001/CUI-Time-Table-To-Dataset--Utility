import pymupdf


PDF_FILE = 'cui ftt.pdf'
zoom = 3  # zoom in so it's easier for OCR
mat = pymupdf.Matrix(zoom, zoom)

with pymupdf.open(PDF_FILE) as pdf_doc:
    for idx, pg in enumerate(pdf_doc.pages()):
        px = pg.get_pixmap(matrix=mat)
        px.save(f'./out/out-{idx+1}.png')
