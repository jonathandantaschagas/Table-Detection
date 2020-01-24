
from __future__ import print_function
import os
import cv2
import csv
import imutils
import pandas as pd
from unidecode import unidecode
from pytesseract import pytesseract as pt
import argparse


import sys
import re


def progress_bar(total, progress):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rLoading: [{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()



parser = argparse.ArgumentParser(description="Detecção de tabelas e extração de texto para formato csv")
parser.add_argument('-i', '--image', required=False, help='Selecione um imagem com extensão jpg, jpeg ou png')
args = vars(parser.parse_args())


def pre_process_image(img, save_in_file, morph_size = (8,8)): #, morph_size=(8, 8)
    print('\n3º - Initializing pre process image')
    #print(img)
    # Converter a imagem
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Definir limite
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size) #
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)

    print('Image successfully processed!')
    return pre


def find_text_boxes(pre, min_text_height_limit=10, max_text_height_limit=40):
    # Procurando contornos de textos presentes na imagem
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    print('\n4º - Searching for text strucutured:')
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Pegando a caixas ao redor do texto baseado no tamanho
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    print('Text structured founded!')
    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    print('\n5º - Searching for cells in texts')
    rows = {}
    cols = {}

    # Agrupando as caixas de seleção pelas posições.
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # Removendo clusters com menos de duas colunas
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Ordernando as celulas pela coordenada de X
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Ordenando as linhas pelas coordenada se Y
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    print('Cells founded!')

    return table_cells


def build_lines(table_cells):
    print('\n6º - Building lines')
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for i, box in enumerate(table_cells):
        # print('{}º horizontal line builded!'.format(i + 1))
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for i, box in enumerate(table_cells[0]):
        # print('{}º vertical line builded!'.format(i + 1))
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))


    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    print('{} rows'.format(len(hor_lines) - 1))
    print('{} columns'.format(len(ver_lines) -1))

    return hor_lines, ver_lines

# Aplicando OCR Pytesseract
def get_text(img):
    text = pt.image_to_string(img)
    text = unidecode(text)
    return text


def extract_data(img_path, img_name, image_ext):

    print('\n2º - Initializing extract data')

    if not os.path.exists('asset'):
        try:
            os.mkdir('asset')
        except OSError:
            print('Failed to create "asset"')

    if not os.path.exists(os.path.join('asset', 'upload')):
        try:
            os.mkdir('asset\\upload')
        except OSError:
                print('Failed to create folder "upload"')

    if not os.path.exists(os.path.join('asset', 'upload', img_name)):
        try:
            os.mkdir('asset\\upload\\{}'.format(img_name))
        except OSError:
            print('Failed to create folder "{}"'.format(img_name))

    pre_file = os.path.join('asset', 'upload', img_name, img_name + '-' + 'pre.png')
    out_file = os.path.join('asset', 'upload', img_name, img_name + '-' + 'out.png')
    tmp_file = os.path.join('asset', 'upload', img_name, img_name + '-' + 'tmp.png')

    img = cv2.imread(img_path)

    processed = pre_process_image(img, pre_file)
    text_boxes = find_text_boxes(processed)
    cells = find_table_in_boxes(text_boxes)

    hor_lines, ver_lines = build_lines(cells)

    view = img.copy()

    print('\n7º - Drawing rectagle around the text')
    for box in text_boxes:
        (x,y,w,h) = box
        cv2.rectangle(view, (x,y), (x + w - 2, y + h -2), (0,255,0), 1)

    print('\n8º - Drawing horizonatal lines to build cells on detected table')
    rows = []
    for line in hor_lines:
        [x1,y1,x2,y2] = line
        rows.append(line)
        cv2.line(view, (x1, y1), (x2, y2), (0, 0, 255), 1)

    print('\n8º - Drawing vertical lines to build cells on detected table')
    columns = []
    for line in ver_lines:
        [x1,y1,x2,y2] = line
        columns.append(line)
        cv2.line(view, (x1, y1), (x2, y2), (0, 0, 255), 1)

    print('\n9º - Get coordinates from rows and columns drawn. Top, left, width, height. (x,y,w,h)')
    row = []
    for idx, rs in enumerate(rows):
    	if idx == 0:
    		continue
    	x,y = rows[idx-1][1], rs[1]
    	rs = [x,y]
    	row.append(rs)

    column = []
    for idx, cols in enumerate(columns):
        if idx == 0:
            continue

        x, y = columns[idx-1][0],cols[0]
        cols = [x,y]
        column.append(cols)

    print('\n10º Grouping coords of rows and columns in cells')
    cell = []
    for r in row:
    	row_ = []
    	for c in column:
    		x1, x2, y1, y2 = r[0], r[1], c[0], c[1]
    		cel = [x1,x2,y1,y2]
    		row_.append(cel)
    	cell.append(row_)
    # print(cell)


    file = []
    print('\n11º - Crop cells and send to pytesseract.')

    if not os.path.exists(os.path.join('asset', 'upload', img_name, 'cells')):
        try:
            os.mkdir('asset\\upload\\{}\\cells'.format(img_name))
        except OSError:
            print('Failed to create folder cells')


    for i, c in enumerate(cell):
        progress_bar(len(cell), i +1)
        r = []
        for j, text_cell in enumerate(c):
            (x1, x2, y1, y2) = text_cell
            crop_img = img[x1:x2, y1:y2]
            text = get_text(crop_img)
            r.append(text.replace('\n', ' '))
            # print('Row: {}, Column: {} - x: {}, y: {}, w: {}, h: {}'.format(i +1, j +1, x, y, w, h))
            cv2.imwrite(os.path.join('asset','upload', img_name, 'cells', "row {} col{}.{}".format(str(i + 1),str(j+1),"png")) , crop_img) # Salvar imagem
            #cv2.imshow("cropped", crop_img) # Open image on windows
        file.append(r)
    file = pd.DataFrame(file)
    file.to_csv(os.path.join('asset/upload', img_name, 'output.csv'), sep=';', encoding='utf-8', index=False, header=False)
    cv2.imwrite(out_file, view)
    print('\n\nResult: {} rows and {} cols exported to file'.format(len(hor_lines) - 1, len(ver_lines)-1))





if __name__ == "__main__":
    print('1º - The process has begin')
    img_path = args['image']
    img_name = img_path.split('.')[0]
    img_ext = img_path.split('.')[1]

    extract_data(img_path, img_name, img_ext)
