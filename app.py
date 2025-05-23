import os
import gradio as gr
from utils import get_img, find_faces_img, is_real_person, analyze_vid
from db import setup_db, record_attendance
setup_db()
media_dir = 'test_media'

def handle_file(path):
    if not os.path.exists(path):
        msg=f"Can't find file: {path}"
        return(msg)

    fname = os.path.basename(path).lower()
    
    # working with image
    if fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png") or fname.endswith(".webp"):
        img = get_img(path)
        face_boxes = find_faces_img(img)
        seen = len(face_boxes)
        real = 0

        for box in face_boxes:
            t, r, b, l = box
            try:
                snippet = img[t:b, l:r]
                if is_real_person(snippet):
                    real += 1
            except:
                continue

        record_attendance(seen, real)
        msg = f"[IMG] {fname} — {seen} seen / {real} real"
        print(msg)
        return msg

    # or maybe it's a video?
    elif fname.endswith(".mp4") or fname.endswith(".webm") or fname.endswith(".avi") or fname.endswith(".mov"):
        seen, real = analyze_vid(path)
        record_attendance(seen, real)
        msg = f"[VID] {fname} — {seen} seen / {real} real"
        print(msg)
        return msg

    else:
        msg = f"File format not supported -> {fname}"
        print(msg)
        return msg
iface=gr.Interface(
    fn=handle_file,
    inputs=gr.File(label="Uplaod"),
    outputs="text",
    title="Face Recognition",
    description="Internhsip"
)
iface.launch()
def go_through_folder():
    if not os.path.isdir(media_dir):
        print("media folder missing.")
        return

    things = os.listdir(media_dir)
    if not things:
        print("Empty folder.")
        return

    print(f"Checking {len(things)} items in '{media_dir}'...\n")

    for item in things:
        full = os.path.join(media_dir, item)
        if os.path.isfile(full):
            handle_file(full)

if __name__ == '__main__':
    setup_db()
    go_through_folder()
