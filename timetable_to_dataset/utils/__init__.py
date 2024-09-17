import datetime
import cv2
import pytesseract
import re
import numpy as np

DAYS = ['mo', 'tu', 'we', 'th', 'fr', 'sa', 'su']

def dur_wrapper(fn, *args, **kwargs):
    start = datetime.datetime.now().replace(microsecond=0)
    ret = fn(*args, **kwargs)
    print(f'Took {datetime.datetime.now().replace(microsecond=0) - start}...\n')
    return ret

def get_sorted_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    sort_cnts = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))
    
    return sort_cnts

def read_txt(ROI):
    txt: str = pytesseract.image_to_string(
                    ROI
                    )
    txt = txt.strip().replace('\n', ' ')

    if not txt:
        gray = ROI
        _, thresh = cv2.threshold(gray, 0, 254, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
        txt: str = pytesseract.image_to_string(
            thresh
            )
        txt = txt.strip().replace('\n', ' ').replace(',', ' ')
    
    if not txt:
        gray = ROI
        gray = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, thresh = cv2.threshold(gray, 0, 254, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        txt: str = pytesseract.image_to_string(
            thresh
            )
        txt = txt.strip().replace('\n', ' ').replace(',', ' ')

    return txt

def retry_read_txt_until(ROI, initial_txt,
                         condition_fn, remove_invalid_chars_fn=lambda x: x,
                         retries=2, retry_scale=1.0, scale_inc=0.4,
                         try_downsampling=False):
    txt = initial_txt
    total_retries = retries
    
    # try upsampling
    while retries > 0 and not condition_fn(txt):
        retries -= 1
        retry_scale += scale_inc

        retry_res = cv2.resize(ROI, (0,0), fx=retry_scale, fy=retry_scale, interpolation=cv2.INTER_LINEAR)
        try:
            txt_b = read_txt(retry_res)

            txt_b = bytes(txt_b, 'utf-8').decode('utf-8', 'ignore')
            txt_b = remove_invalid_chars_fn(txt_b)
            
            if condition_fn(txt_b):  # prefer string that satisfies our condition
                if condition_fn(txt):
                    txt = txt if len(txt) > len(txt_b) else txt_b  # if both satisfy, keep the one with more info
                else:
                    txt = txt_b
            # print(txt)
        except:
            pass

    if try_downsampling:
        retries = total_retries
        retry_scale = 1.0
        # Note: if above worked, then condition_fn would return true and control wouldn't enter in this loop
        while retries > 0 and not condition_fn(txt):
            retries -= 1
            retry_scale -= scale_inc
            retry_res = cv2.resize(ROI, (0,0), fx=retry_scale, fy=retry_scale, interpolation=cv2.INTER_AREA)
            try:
                txt_b = read_txt(retry_res)
                txt = txt if len(txt) > len(txt_b) else txt_b
                # print(txt)
            except:
                pass

    return txt, condition_fn(txt)

def remove_student_batch_info(txt):
    return re.sub(r'((fa)|(FA)|(sp)|(SP))[0-9][0-9]', '', txt)

def remove_course_code(txt):
    return re.sub(r'[A-Za-z]{3}[0-9]{3}', '', txt)

def extract_rooms(txt: str):
    return re.findall(r'([A-Za-z]-?[0-9]{1,3})|(0-[1-9]{1})', txt)

def is_time_slot(txt):
    return bool(re.findall(r'[0-9]{1,2}:(00)|(30)', txt))

def has_rooms(txt):
    return bool(extract_rooms(remove_student_batch_info(remove_course_code(txt))))

def check_expectations(row, txt, day):
    errs = []

    if row and len(row) >= 20 and txt and txt not in DAYS:
        # with open('./errors.txt', '+a') as f:
        #     f.write(f'Mismatch b/w number of slots and text: {row} | {txt}\n')
        errs += [f'Mismatch b/w number of slots and text: {row} | {txt}']

    if txt in DAYS and DAYS.index(txt) != (DAYS.index(day) +1) % 7:
        # with open('./errors.txt', '+a') as f:
        #     f.write(f'Mismatch in expected day, got {txt} instead of {day}: {row}\n')
        errs += [f'Mismatch in expected day, got {txt} instead of {day}: {row}\n']
    
    # if txt in DAYS and not all([extract_rooms(r) for r in row]):
    #     with open('./errors.txt', '+a') as f:
    #         f.write(f'Could not read rooms for: {day} | {row}\n')
    
    return errs

def write_slots(data):
    with open('./out-csv/manual.csv', 'w') as f:
        f.write('Day,' + ','.join([str(i) for i in range(1, 21)]))
        f.write('\n')
        for k_day in data:
            for time_slots in data[k_day]:
                f.write(k_day + ',' + ','.join(time_slots))
                f.write('\n')

def print_slots(data):
    for k in data:
        for s in data[k]:
            print(k, len(s), s)

def get_img_num(txt):
    # e.g: out-1.png -> [out, 1.png] -> 1.png -> [1, png] -> 1
    return int(txt.split('-')[-1].split('.')[0])

def remove_colors(img):
    # replace colors with white, so all text have white background
    # for brighter colors
    not_close_to_black = np.mean(img, axis=-1, keepdims=False) > 70
    img[not_close_to_black] = np.array([255,255,255])

    # for dark colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturated = img_hsv[:, :, 1] > 100
    img[saturated] = np.array([255,255,255])

    return img