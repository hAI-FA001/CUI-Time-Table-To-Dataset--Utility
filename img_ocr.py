import numpy as np
import cv2
import pytesseract
import os
import datetime
import re


DAYS = ['mo', 'tu', 'we', 'th', 'fr', 'sa', 'su']
rows = {}
day = None
row = None


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
        # gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 254, cv2.THRESH_BINARY)
        # cv2.imshow('thresh1', thresh)
        thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        # cv2.imshow('thresh2', thresh)
        # cv2.waitKey(0)
        txt: str = pytesseract.image_to_string(
            thresh
            )
        txt = txt.strip().replace('\n', ' ').replace(',', ' ')
    # # print("READ: ", txt)

    if not txt:
        gray = ROI
        gray = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, thresh = cv2.threshold(gray, 0, 254, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        txt: str = pytesseract.image_to_string(
            thresh
            )
        txt = txt.strip().replace('\n', ' ').replace(',', ' ')

    # if not txt:
    #     cv2.imshow('Can\'t find', ROI)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     raise ValueError(f'Could not find text...')
    return txt

def retry_read_txt_until(ROI, initial_txt, condition_fn, remove_invalid_chars_fn=lambda x: x, retries=9, retry_scale=1.0):
    txt = initial_txt
    
    # try upsampling
    while retries > 0 and not condition_fn(txt):
        retries -= 1
        retry_scale += 0.1

        retry_res = cv2.resize(ROI, (0,0), fx=retry_scale, fy=retry_scale)
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

    # try downsampling if it didn't work
    # retries = 9
    # retry_scale = 1.0
    # # Note: if above worked, then condition_fn would return true and control wouldn't enter in this loop
    # while retries > 0 and not condition_fn(txt):
    #     retries -= 1
    #     retry_scale -= 0.1
    #     retry_res = cv2.resize(ROI, (0,0), fx=retry_scale, fy=retry_scale)
    #     try:
    #         txt_b = read_txt(retry_res)
    #         txt = txt if len(txt) > len(txt_b) else txt_b
    #         # print(txt)
    #     except:
    #         pass

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
    if row and len(row) >= 20 and txt not in DAYS:
        with open('./errors.txt', '+a') as f:
            f.write(f'Mismatch b/w slots and text: {row} | {txt}\n')

    if txt in DAYS and DAYS.index(txt) != (DAYS.index(day) +1) % 7:
        with open('./errors.txt', '+a') as f:
            f.write(f'Mismatch in expected day, got {txt} instead of {day}: {row}\n')
    
    # if txt in DAYS and not all([extract_rooms(r) for r in row]):
    #     with open('./errors.txt', '+a') as f:
    #         f.write(f'Could not read rooms for: {day} | {row}\n')

def task(img_name, scaling_factor=1.0):
    global rows, row, day

    img = cv2.imread(f'./out/{img_name}')

    # replace colors with white, so all text have white background
    not_close_to_zero = np.mean(img, axis=-1, keepdims=False) > 70
    img[not_close_to_zero] = np.array([255,255,255])

    if scaling_factor != 1.0:
        res = cv2.resize(img, (0, 0), fx=scaling_factor, fy=scaling_factor)
    else:
        res = img
    sort_cnts = get_sorted_contours(res)

    for cnt in sort_cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        
        if abs(scaling_factor*200 - h) <= 10:
            ROI = res[y:y+h, x:x+w]
            
            if 0 <= ROI.mean() <= 254:
                # cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0), 2)
                
                try:
                    txt = read_txt(ROI)
                except:
                    txt = ''
                
                txt, is_satisfied = retry_read_txt_until(ROI, txt, lambda t: bool(t))
                case_sensitive_txt = txt
                txt = txt.lower()
                
                if day:
                    check_expectations(row, txt, day.lower())
                    
                    if txt not in DAYS and not is_time_slot(txt) and not has_rooms(txt):
                        room_txt, is_satisfied = retry_read_txt_until(ROI[h-60:h, :], '', has_rooms, lambda x: re.sub(r'[^A-Za-z0-9-]', '', x))
                        # print(f'{txt}\t{room_txt}\t{is_satisfied}')
                        
                        if not is_satisfied:
                            with open('./errors.txt', '+a') as f:
                                f.write(f'[AFTER RETRIES] Could not read room for: {day} | {txt} | {room_txt}\n')
                            # print(f'[AFTER RETRIES] Couldn\'t read room for: {day} | {txt}')
                        else:
                            txt += ' ' + room_txt.lower()
                            case_sensitive_txt += ' ' + room_txt
                        #     print('APPEND', txt, case_sensitive_txt)
                        # print()

                if txt in DAYS:
                    if row and any([bool(r.strip()) for r in row]):
                        rows[day] = rows.get(day, []) + [row]
                    day = txt
                    row = []
                else:
                    # calc number of slots occupied based on width
                    for _ in range(int(w // (100*scaling_factor))):
                        row.append(case_sensitive_txt)
            else:
                # cv2.rectangle(res, (x,y), (x+w,y+h), (0,0,255), 2)
                row.append(' ')
    
    return txt

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

if __name__ == '__main__':
    # truncate it on each run
    with open('./errors.txt', 'w') as f:
        pass

    write_interval = 100
    img_names = os.listdir('./out/')
    img_names = sorted(img_names, key=get_img_num)

    for idx, img_name in enumerate(img_names):
        print(f'[{idx+1}/{len(img_names)}] [{(idx+1) / len(img_names) * 100:7.2f}%] Processing {img_name}...')
        txt = dur_wrapper(task, img_name)
        
        if (idx + 1) % write_interval == 0:
            print(f'\nSaving data ({idx+1}/{len(img_names)})...')
            dur_wrapper(write_slots, rows)
            print()
        # break

    rows[day] = rows.get(txt.lower(), []) + [row]

    dur_wrapper(write_slots, rows)
