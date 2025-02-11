import cv2
import os

def save_frames(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    parent_folder = os.path.dirname(video_path)
    output_folder = video_name
    output_path = os.path.join(parent_folder, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frame_interval = int(fps // 10)  # 10 frames per second

    frame_idx = 0
    saved_frame_count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(output_path, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_idx += 1

    video.release()
    print(f"Frames saved to '{output_path}'.")

video_path = "Clips/3097.mp4"
save_frames(video_path)
