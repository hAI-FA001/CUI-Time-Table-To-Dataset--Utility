import cv2
import os
import re
from concurrent.futures import ProcessPoolExecutor

from timetable_to_dataset.utils import *


def get_text(roi, row, day):
    try:
        txt = read_txt(roi)
    except Exception as e:
        print(f'[EXCEPTION CAUGHT] {e}')
        txt = ''
        raise e
    
    txt, is_satisfied = retry_read_txt_until(roi, txt, lambda t: bool(t))
    
    if not is_satisfied:
        eroded_ROI = roi
        for _ in range(5): eroded_ROI = cv2.erode(eroded_ROI, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        txt, is_satisfied = retry_read_txt_until(eroded_ROI, txt, lambda t: bool(t), try_downsampling=True)
    
    # 1 day has 20 slots, so this should be the next day
    if row and not is_satisfied and len(row) == 20:
        txt = DAYS[(DAYS.index(day) + 1) % 7]
    
    return txt

def get_room(roi, txt, day):
    errs = []
    room_txt, is_satisfied = retry_read_txt_until(roi, '',
                                             lambda t: has_rooms(t) or 'lab' in t.lower(),
                                             lambda x: re.sub(r'[^A-Za-z0-9-]', '', x))
    if not is_satisfied:
        # with open('./errors.txt', '+a') as f:
        #     f.write(f'[AFTER RETRIES] Could not read room for: {day} | {txt} | {room_txt}\n')
        errs += [f'[AFTER RETRIES] Could not read room for: {day} | {txt} | {room_txt}\n']

    return room_txt, is_satisfied, errs

def task(img_name):
    rows = {}
    errs = []
    day = None
    row = None

    img = cv2.imread(f'./out/{img_name}')
    sort_cnts = get_sorted_contours(img)
    img = remove_colors(img)

    for cnt in sort_cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        
        if abs(200 - h) <= 10:
            ROI = img[y:y+h, x:x+w]
            
            if ROI.mean() < 255:
                txt_roi = ROI[:h-60, :]
                room_roi = ROI[h-60:, :]
                # cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0), 2)

                case_sensitive_txt = get_text(txt_roi, row, day)
                txt = case_sensitive_txt.lower()

                if day:
                    errs += check_expectations(row, txt, day.lower())
                    
                    if txt not in DAYS and not is_time_slot(txt) and not has_rooms(txt):
                        room_txt, is_satisfied, room_errs = get_room(room_roi, txt, day)
                        errs += room_errs
                        
                        if is_satisfied:
                            txt += ' ' + room_txt.lower()
                            case_sensitive_txt += ' ' + room_txt
                        
                case_sensitive_txt = case_sensitive_txt.replace(',', ' ')

                if txt in DAYS:
                    if row and any([bool(r.strip()) for r in row]):
                        rows[day] = rows.get(day, []) + [row]
                    day = txt
                    row = []
                else:
                    # calc number of slots occupied based on width
                    for _ in range(int(w // 100)):
                        row.append(case_sensitive_txt)
            else:
                row.append(' ')
    
    if row and any([bool(r.strip()) for r in row]):
        rows[day] = rows.get(day, []) + [row]

    return rows, errs


if __name__ == '__main__':
    write_interval = 500
    img_names = os.listdir('./out/')
    
    rows = {}
    errs = []
    with ProcessPoolExecutor() as ex:
        futures = []
        for idx, img_name in enumerate(img_names):
            # print(f'[{idx+1}/{len(img_names)}] [{(idx+1) / len(img_names) * 100:7.2f}%] Processing {img_name}...')
            futures.append(ex.submit(task, img_name))
            # if (idx+1) % 50 == 0:

        total = len(futures)
        for i, f in enumerate(futures):
            print(f'[{i+1}/{total}] Processing... {(i+1) / total * 100:10.2f}%')
            r, e = dur_wrapper(f.result)

            for k in r:
                rows[k] = rows.get(k, []) + r[k]
            
            errs += e
    
    with open('./errors.txt', 'w') as f:
        f.writelines(errs)

    dur_wrapper(write_slots, rows)
