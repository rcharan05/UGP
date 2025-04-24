import os
import cv2
import yt_dlp
import ffmpeg
import moviepy
import pytesseract
import shutil
import numpy as np
import time
import uuid
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from moviepy import VideoFileClip
from difflib import SequenceMatcher
from urllib.parse import urlparse, parse_qs
import multiprocessing

# Set Tesseract executable path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\rchar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# Configuration
BASE_OUTPUT_DIR = r"D:\WOTD\New"  # Base folder where global Meaning and Sentence folders will reside.
FPS = 1
TEXT_SIMILARITY_THRESHOLD = 0.5

# Create global output folders for Meaning and Sentence
MEANING_FOLDER = os.path.join(BASE_OUTPUT_DIR, "Meaning")
SENTENCE_FOLDER = os.path.join(BASE_OUTPUT_DIR, "Sentence")
os.makedirs(MEANING_FOLDER, exist_ok=True)
os.makedirs(SENTENCE_FOLDER, exist_ok=True)

def get_playlist_info(url):
    print(f"Fetching playlist information from {url} ...")
    parsed_url = urlparse(url)
    qs = parse_qs(parsed_url.query)
    start_index = None
    if 'index' in qs:
        try:
            start_index = int(qs['index'][0])
            print(f"Starting from video number: {start_index}")
        except Exception as e:
            print(f"Could not parse index parameter: {e}")
    
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:
                    playlist_name = info.get('title', 'Unknown_Playlist')
                    videos = []
                    for entry in info['entries']:
                        if entry is not None:
                            videos.append({
                                'url': f"https://youtu.be/{entry['id']}",
                                'title': entry.get('title', 'Unknown_Video')
                            })
                else:
                    playlist_id = None
                    if 'list=' in url:
                        playlist_id = url.split('list=')[1].split('&')[0]
                    if playlist_id:
                        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
                        try:
                            playlist_info = ydl.extract_info(playlist_url, download=False)
                            if playlist_info and 'entries' in playlist_info:
                                playlist_name = playlist_info.get('title', 'Unknown_Playlist')
                                videos = []
                                for entry in playlist_info['entries']:
                                    if entry is not None:
                                        videos.append({
                                            'url': f"https://youtu.be/{entry['id']}",
                                            'title': entry.get('title', 'Unknown_Video')
                                        })
                            else:
                                raise ValueError("Could not extract playlist entries")
                        except Exception as pe:
                            print(f"Failed to extract playlist, falling back to single video: {str(pe)}")
                            playlist_name = "Single_Video"
                            videos = [{
                                'url': f"https://youtu.be/{info['id']}",
                                'title': info.get('title', 'Unknown_Video')
                            }]
                    else:
                        playlist_name = "Single_Video"
                        videos = [{
                            'url': f"https://youtu.be/{info['id']}",
                            'title': info.get('title', 'Unknown_Video')
                        }]
                playlist_name = "".join(c for c in playlist_name if c.isalnum() or c in (' ', '-', '_')).strip()
                if not videos:
                    raise ValueError("No valid videos found in the playlist")
                if start_index is not None:
                    videos = videos[start_index - 1:]
                    print(f"Processing {len(videos)} videos from video {start_index} onward.")
                print(f"Found {len(videos)} videos in playlist: {playlist_name}")
                for i, video in enumerate(videos, 1):
                    print(f"{i}. {video['title']}")
                return playlist_name, videos
            except Exception as e:
                print(f"Error extracting playlist info: {str(e)}")
                if 'list=' in url:
                    playlist_id = url.split('list=')[1].split('&')[0]
                    alt_url = f"https://www.youtube.com/playlist?list={playlist_id}"
                    print(f"Trying alternative playlist URL: {alt_url}")
                    return get_playlist_info(alt_url)
                else:
                    raise
    except Exception as e:
        print(f"Failed to process URL: {str(e)}")
        raise ValueError("Could not extract video information from the provided URL")
    
def download_video_and_get_info(url, output_file, cookies_from_browser="chrome"):
    print(f"Downloading video from {url} ...")
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": output_file,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 10,
        "sleep_interval": 2,
    }
    if cookies_from_browser:
        ydl_opts["cookies_from_browser"] = cookies_from_browser
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    print("Download complete.")
    return info

def extract_screenshot(video_file, time_sec, output_image):
    print(f"Extracting frame at {time_sec} seconds for screenshot selection...")
    (ffmpeg
     .input(video_file, ss=time_sec)
     .output(output_image, vframes=1)
     .global_args("-threads", "0")
     .run(overwrite_output=True))
    print(f"Screenshot saved as {output_image}.")

def select_crop_region(image_path, region_name):
    image = cv2.imread(image_path)
    clone = image.copy()
    roi = None
    refPt = []
    def click_and_crop(event, x, y, flags, param):
        nonlocal roi
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[:] = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            roi = (refPt[0][0], refPt[0][1], refPt[1][0]-refPt[0][0], refPt[1][1]-refPt[0][1])
            cv2.rectangle(image, refPt[0], refPt[1], (0,255,0), 2)
            cv2.imshow(f"Select {region_name} Region", image)
    #cv2.namedWindow(f"Select {region_name} Region")
    # allow manual resize & full‐image display
    cv2.namedWindow(f"Select {region_name} Region", cv2.WINDOW_NORMAL)

    # immediately resize the window to match the image’s resolution
    # note: image.shape is (height, width, channels)
    h, w = image.shape[:2]
    cv2.resizeWindow(f"Select {region_name} Region", w, h)

    cv2.setMouseCallback(f"Select {region_name} Region", click_and_crop)
    while True:
        cv2.imshow(f"Select {region_name} Region", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image = clone.copy()
            roi = None
        elif key in [13, 32]:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()
    return roi

def extract_frames(video_file, frames_folder, fps=1):
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    os.makedirs(frames_folder)
    print(f"Extracting frames from {video_file} into {frames_folder} ...")
    (ffmpeg
     .input(video_file)
     .filter('fps', fps=fps)
     .output(os.path.join(frames_folder, "frame_%04d.png"))
     .global_args("-threads", "0")
     .run(overwrite_output=True))
    print("Frame extraction complete.")

def get_text_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def get_frame_text(image, text_coords):
    x, y, w, h = text_coords
    text_region = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray).strip()

def detect_text_intervals(frames_folder, text_coords_tuple, desc_lines):
    """
    Detect intervals in which each expected description line is visible.
    Uses two text ROI selections:
      - For the first description line, use text_coords_tuple[0].
      - For the remaining lines, use text_coords_tuple[1].
    Waits until a description line is detected to mark the segment start,
    then marks its end when the description stops being detected.
    Returns a list of tuples: (start_time, end_time, description)
    """
    print("\nDetecting text intervals...")
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    intervals = []
    current_desc_idx = 0
    in_segment = False
    stability_counter = 0
    STABILITY_THRESHOLD = 2
    start_time = None
    rewind_frames = int(2 * FPS)
    
    i = 0
    n_frames = len(frame_files)
    while i < n_frames and current_desc_idx < len(desc_lines):
        frame_path = os.path.join(frames_folder, frame_files[i])
        image = cv2.imread(frame_path)
        if image is None:
            i += 1
            continue

        current_roi = text_coords_tuple[0] if current_desc_idx == 0 else text_coords_tuple[1]
        ocr_text = get_frame_text(image, current_roi)
        expected_line = desc_lines[current_desc_idx]
        similarity = get_text_similarity(ocr_text, expected_line)
        
        if similarity > TEXT_SIMILARITY_THRESHOLD:
            if not in_segment:
                stability_counter += 1
                if stability_counter >= STABILITY_THRESHOLD:
                    start_time = i - (STABILITY_THRESHOLD - 1)
                    in_segment = True
                    stability_counter = 0
                    print(f"Started segment for '{expected_line}' at {start_time}s")
            else:
                stability_counter = 0
        else:
            if in_segment:
                stability_counter += 1
                if stability_counter >= STABILITY_THRESHOLD:
                    end_time = i - (STABILITY_THRESHOLD - 1)
                    intervals.append((start_time, end_time, expected_line))
                    print(f"Ended segment for '{expected_line}' at {end_time}s")
                    in_segment = False
                    stability_counter = 0
                    current_desc_idx += 1
                    i = max(0, i - rewind_frames)
                    continue
            else:
                stability_counter = 0
        i += 1

    if in_segment and current_desc_idx < len(desc_lines):
        end_time = n_frames
        intervals.append((start_time, end_time, desc_lines[current_desc_idx]))
        print(f"Ended segment for '{desc_lines[current_desc_idx]}' at {end_time}s (end of video)")
    
    return intervals

def clip_video_segment(video_file, start_sec, end_sec, output_file, person_crop_coords):
    print(f"Processing segment from {start_sec} to {end_sec} seconds...")
    duration = end_sec - start_sec
    print(f"Clipping segment from {start_sec} to {end_sec} seconds into {output_file} ...")
    x, y, w, h = person_crop_coords
    (ffmpeg
     .input(video_file, ss=start_sec, t=duration)
     .crop(x, y, w, h)
     .output(output_file,
             video_bitrate='5000k',
             acodec='aac',
             vcodec='libx264',
             g=1,
             keyint_min=1,
             strict='-2')
     .global_args("-threads", "0")
     .overwrite_output()
     .run(quiet=True))

def process_video(video_info, playlist_name, text_coords_tuple, person_coords):
    """
    Processes one video:
      - Downloads the video.
      - Extracts frames and detects text intervals.
      - For each segment:
            * If its description contains "meaning" or "example" (case-insensitive) 
              or contains "1)", "2)", or "3)", store the segment in the global Sentence folder.
            * Otherwise, store in the global Meaning folder.
            * A CSV row is generated for each segment: [unique_id, video_title, description].
      - If no segments are detected, temporary files are removed.
    Returns (success, meaning_rows, sentence_rows)
    """
    try:
        video_title = "".join(c for c in video_info['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
        print(f"\nProcessing video: {video_title}")
        temp_dir = os.path.join(BASE_OUTPUT_DIR, "temp", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
    
        video_file = os.path.join(temp_dir, "video.mp4")
        info = download_video_and_get_info(video_info['url'], video_file)
        description = info.get("description", "")
        desc_lines = [line.strip() for line in description.splitlines() if line.strip()]
        if not desc_lines:
            print("No description lines found. Skipping video.")
            shutil.rmtree(temp_dir)
            return True, [], []
    
        frames_dir = os.path.join(temp_dir, "frames")
        extract_frames(video_file, frames_dir, fps=FPS)
        segments = detect_text_intervals(frames_dir, text_coords_tuple, desc_lines)
        if not segments:
            print("No text intervals detected using initial ROI pair. Reprocessing with second text ROI only...")
            segments = detect_text_intervals(frames_dir, (text_coords_tuple[1], text_coords_tuple[1]), desc_lines)
            if not segments:
                print("Still no text intervals detected. Marking video as failed.")
                shutil.rmtree(temp_dir)
                return False, [], []
    
        segments = [(round(s, 3), round(e, 3), d) for s, e, d in segments]
        video_uid = str(uuid.uuid4())
        meaning_rows = []   # Rows for segments saved in Meaning folder
        sentence_rows = []  # Rows for segments saved in Sentence folder
        print("\nCreating segment files...")

        for seg in segments:
            s, e, seg_desc = seg
            # Check if the segment text contains "meaning" or "example" (case-insensitive) 
            # or if it contains "1)", "2)", or "3)".
            if ("meaning" in seg_desc.lower() or 
                "example" in seg_desc.lower() or 
                "1)" in seg_desc or "2)" in seg_desc or "3)" in seg_desc):
                out_folder = SENTENCE_FOLDER
                segment_type = "sentence"
            else:
                out_folder = MEANING_FOLDER
                segment_type = "meaning"
        
            segment_file = os.path.join(out_folder, f"{video_uid}_{segment_type}_{s}_{e}.mp4")
            clip_video_segment(video_file, s, e, segment_file, person_coords)
            row = [video_uid, video_title, seg_desc]
            if segment_type == "sentence":
                sentence_rows.append(row)
                print(f"Created Sentence segment for video UID {video_uid} from {s}s to {e}s")
            else:
                meaning_rows.append(row)
                print(f"Created Meaning segment for video UID {video_uid} from {s}s to {e}s")
    
        shutil.rmtree(temp_dir)
        print(f"Completed processing video: {video_title}")
        return True, meaning_rows, sentence_rows
    except Exception:
        import traceback
        with open("worker_error.log", "a") as f:
            f.write(traceback.format_exc())
        return False, [], []
def append_or_write_csv(csv_path, header, rows):
    # If the file exists, open in append mode; otherwise, write header and rows.
    mode = 'a' if os.path.exists(csv_path) else 'w'
    with open(csv_path, mode=mode, newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        if mode == 'w':
            csv_writer.writerow(header)
        csv_writer.writerows(rows)

def main():
    all_meaning_rows = []
    all_sentence_rows = []
    common_rows = []  # Combined rows for common CSV
    playlist_urls = []
    print("Enter YouTube playlist URLs (enter an empty line to finish):")
    while True:
        url = input().strip()
        if not url:
            break
        playlist_urls.append(url)
    if not playlist_urls:
        print("No playlists provided. Exiting.")
        return

    # --- First Pass: ROI selection from the first playlist's first video ---
    first_playlist_url = playlist_urls[0]
    first_playlist_name, first_videos = get_playlist_info(first_playlist_url)
    first_playlist_dir = os.path.join(BASE_OUTPUT_DIR, "temp_roi")
    os.makedirs(first_playlist_dir, exist_ok=True)
    temp_video_file = os.path.join(first_playlist_dir, "temp_video.mp4")
    download_video_and_get_info(first_videos[0]['url'], temp_video_file)
    with VideoFileClip(temp_video_file) as clip:
        middle_time = clip.duration / 2
    screenshot_path = os.path.join(first_playlist_dir, "screenshot.png")
    extract_screenshot(temp_video_file, middle_time, screenshot_path)
    print("\nSelecting initial ROI pair (used for first pass)")
    print("Please select the TEXT region for the FIRST description line...")
    text_coords_first = select_crop_region(screenshot_path, "Text (First Line)")
    if not text_coords_first:
        print("No text region selected. Exiting.")
        return
    print("\nPlease select the TEXT region for REMAINING description lines...")
    text_coords_remaining = select_crop_region(screenshot_path, "Text (Remaining Lines)")
    if not text_coords_remaining:
        print("No second text region selected. Exiting.")
        return
    print("\nPlease select the PERSON region...")
    person_coords = select_crop_region(screenshot_path, "Person")
    if not person_coords:
        print("No person region selected. Exiting.")
        return
    global_roi = {'text': (text_coords_first, text_coords_remaining), 'person': person_coords}
    os.remove(temp_video_file)
    os.remove(screenshot_path)
    shutil.rmtree(first_playlist_dir)
    
    all_failed = []
    for playlist_url in playlist_urls:
        playlist_name, videos = get_playlist_info(playlist_url)
        print(f"\nProcessing playlist: {playlist_name}")
        failed_videos = []
        with ProcessPoolExecutor() as executor:
            future_to_video = {executor.submit(process_video, video, playlist_name, global_roi['text'], global_roi['person']): video for video in videos}
            for future in as_completed(future_to_video):
                video = future_to_video[future]
                try:
                    success, meaning_rows, sentence_rows = future.result()
                    if not success:
                        print(f"Video '{video['title']}' failed in first pass.")
                        failed_videos.append(video)
                    else:
                        all_meaning_rows.extend(meaning_rows)
                        all_sentence_rows.extend(sentence_rows)
                        common_rows.extend(meaning_rows)
                        common_rows.extend(sentence_rows)
                except Exception as exc:
                    print(f"Video '{video['title']}' generated an exception: {exc}")
                    failed_videos.append(video)
        print(f"\nCompleted processing playlist: {playlist_name}")
        if failed_videos:
            for video in failed_videos:
                all_failed.append((playlist_name, video))
    
    # --- Second Pass for failed videos (if any) ---
    if all_failed:
        print(f"\n{len(all_failed)} videos across playlists did not yield text intervals in the first pass.")
        print("Proceeding to second pass for these videos.")
        failed_playlist_name, failed_video = all_failed[0]
        playlist_dir = os.path.join(BASE_OUTPUT_DIR, "temp_retry")
        os.makedirs(playlist_dir, exist_ok=True)
        temp_video_file = os.path.join(playlist_dir, "temp_video_retry.mp4")
        download_video_and_get_info(failed_video['url'], temp_video_file)
        with VideoFileClip(temp_video_file) as clip:
            middle_time = clip.duration / 2
        temp_ss = os.path.join(playlist_dir, "temp_screenshot_retry.png")
        extract_screenshot(temp_video_file, middle_time, temp_ss)
        print("\nSelecting SECOND ROI pair for the failed videos")
        print("Please select the TEXT region for the FIRST description line (second ROI)...")
        new_text_roi_first = select_crop_region(temp_ss, "Second Text (First Line)")
        if not new_text_roi_first:
            print("No second text region selected. Skipping second pass.")
            os.remove(temp_video_file)
            os.remove(temp_ss)
            all_failed = []
        else:
            print("\nPlease select the TEXT region for REMAINING description lines (second ROI)...")
            new_text_roi_remaining = select_crop_region(temp_ss, "Second Text (Remaining Lines)")
            if not new_text_roi_remaining:
                print("No second text region for remaining lines selected. Skipping second pass.")
                os.remove(temp_video_file)
                os.remove(temp_ss)
                all_failed = []
            else:
                print("\nPlease select the PERSON region (second ROI)...")
                new_person_roi = select_crop_region(temp_ss, "Second Person")
                os.remove(temp_video_file)
                os.remove(temp_ss)
                if not new_person_roi:
                    print("No second person region selected. Skipping second pass.")
                    all_failed = []
                else:
                    new_text_roi_tuple = (new_text_roi_first, new_text_roi_remaining)
                    with ProcessPoolExecutor() as executor:
                        future_to_failed = {executor.submit(process_video, video, playlist_name, new_text_roi_tuple, new_person_roi): (playlist_name, video) for (playlist_name, video) in all_failed}
                        for future in as_completed(future_to_failed):
                            pl_name, video = future_to_failed[future]
                            try:
                                success, meaning_rows, sentence_rows = future.result()
                                if not success:
                                    print(f"Video '{video['title']}' in playlist {pl_name} failed in second pass.")
                                else:
                                    all_meaning_rows.extend(meaning_rows)
                                    all_sentence_rows.extend(sentence_rows)
                                    common_rows.extend(meaning_rows)
                                    common_rows.extend(sentence_rows)
                            except Exception as exc:
                                print(f"Video '{video['title']}' in playlist {pl_name} generated an exception during second pass: {exc}")
        if os.path.exists(playlist_dir):
            shutil.rmtree(playlist_dir)
    
    header = ["Unique_ID", "Video_Title", "Description"]
    # Write (or append to) CSV file for Meaning folder
    meaning_csv_path = os.path.join(MEANING_FOLDER, "segments.csv")
    append_or_write_csv(meaning_csv_path, header, all_meaning_rows)
    print(f"\nMeaning CSV file created/updated at: {meaning_csv_path}")
    
    # Write (or append to) CSV file for Sentence folder
    sentence_csv_path = os.path.join(SENTENCE_FOLDER, "segments.csv")
    append_or_write_csv(sentence_csv_path, header, all_sentence_rows)
    print(f"\nSentence CSV file created/updated at: {sentence_csv_path}")
    
    # Write (or append to) common CSV file
    common_csv_path = os.path.join(BASE_OUTPUT_DIR, "video_segments_common.csv")
    append_or_write_csv(common_csv_path, header, common_rows)
    print(f"\nCommon CSV file created/updated at: {common_csv_path}")
    
    print("\nCompleted processing all playlists.")


multiprocessing.freeze_support()
multiprocessing.set_start_method("spawn", force=True)

if __name__ == "__main__":
    main()
