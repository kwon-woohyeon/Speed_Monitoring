import os
import urllib.request
import json
import pandas as pd
import cv2
from ultralytics import YOLO
import math
import cvzone
from SimpleTracker import SimpleTracker

# API Key 및 CCTV API URL 설정
API_KEY = "4fbdeaf2b28e41159e6d96a264694d45"
CCTV_API_URL = (
    f'https://openapi.its.go.kr:9443/cctvInfo?apiKey={API_KEY}&type=ex&cctvType=1'
    f'&minX=126.756&maxX=126.7936&minY=37.48135&maxY=37.5843&getType=json'
)

# YOLO 모델 및 마스크 경로 설정
YOLO_MODEL_PATH = "yolov8s.pt"
MASK_IMAGE_PATH = "C:\\whyne\\tensorflow\\11.29\\cctv_mask.png"

# 클래스 이름 정의
CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    # 생략된 클래스
]
CONFIDENCE_THRESHOLD = 0.3

# 시작 및 끝 선 좌표
START_LINE = [175, 203, 374, 208]
END_LINE = [110, 240, 374, 244]

# 저장 폴더 설정
OUTPUT_DIR = "detected_objects"

def clear_output_directory(output_dir):
    """지정된 디렉터리의 모든 파일 삭제"""
    os.makedirs(output_dir, exist_ok=True)  # os.makedirs로 폴더 생성 및 확인 간소화
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"삭제됨: {file_path}")

def fetch_cctv_data(api_url):
    """CCTV API 호출"""
    try:
        with urllib.request.urlopen(api_url) as response:
            json_str = response.read().decode('utf-8')
        json_object = json.loads(json_str)
        return pd.json_normalize(json_object["response"]["data"], sep=',')
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return pd.DataFrame()  # 빈 DataFrame 반환으로 오류 방지

def filter_detections(results, class_names, conf_threshold):
    """YOLO 결과에서 특정 클래스 및 신뢰도를 만족하는 객체 필터링"""
    return [
        [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]) - int(box.xyxy[0][0]),
         int(box.xyxy[0][3]) - int(box.xyxy[0][1])]
        for r in results for box in r.boxes if
        (int(box.cls[0]) < len(class_names) and
         class_names[int(box.cls[0])] in {"car", "truck", "bus", "motorbike"} and
         box.conf[0] > conf_threshold)
    ]

def save_roi(frame, obj_id, elapsed_time):
    """ROI(객체 영역) 저장"""
    if frame.size > 0:  # ROI가 비어있지 않을 경우에만 저장
        file_name = f"{OUTPUT_DIR}/object_{obj_id}_time_{elapsed_time:.2f}.png"
        cv2.imwrite(file_name, frame)
        print(f"ROI 저장: {file_name}")

def process_tracking(obj_id, cx, cy, current_time, tracking_times, start_ids, end_ids):
    """객체 추적 및 시간 기록"""
    if START_LINE[1] <= cy <= START_LINE[3] and obj_id not in start_ids:
        tracking_times[obj_id] = {"start_time": current_time}
        start_ids.add(obj_id)
    elif END_LINE[1] <= cy <= END_LINE[3] and obj_id in start_ids and obj_id not in end_ids:
        elapsed_time = current_time - tracking_times[obj_id]["start_time"]
        tracking_times[obj_id]["elapsed_time"] = elapsed_time
        end_ids.add(obj_id)

def overlay_saved_images(frame):
    """저장된 이미지를 불러와 동일한 위치에 합성하고 과속 텍스트 추가"""
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    if not files:
        return frame  # 저장된 이미지가 없으면 원본 프레임 반환

    for i, file_path in enumerate(files[:5]):  # 최대 5개 이미지 처리
        saved_image = cv2.imread(file_path)
        if saved_image is None:
            continue

        # 이미지를 1/3 크기로 축소
        resized_image = cv2.resize(saved_image, (0, 0), fx=0.3, fy=0.3)

        # 동일한 위치에 이미지 합성 (우측 상단)
        x_offset = frame.shape[1] - resized_image.shape[1] - 10  # 우측 여백 10
        y_offset = 10  # 상단 여백 10

        # 이미지를 덮어쓰는 방식으로 합성
        frame[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image

    
    cv2.putText(
        frame,
        "speeding",
        (535, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # 텍스트 크기
        (0, 0, 255),  # 빨간색 (BGR 포맷)
        2,  # 두께
        cv2.LINE_AA
    )

    return frame


def main():
    """메인 실행"""
    clear_output_directory(OUTPUT_DIR)

    cctv_data = fetch_cctv_data(CCTV_API_URL)
    if cctv_data.empty or 'cctvurl' not in cctv_data.columns:
        print("CCTV URL 데이터를 찾을 수 없습니다.")
        return

    kimpo_url = cctv_data.get('cctvurl', [None])[4]
    if not kimpo_url:
        print("CCTV URL을 찾을 수 없습니다.")
        return

    capture = cv2.VideoCapture(kimpo_url)
    if not capture.isOpened():
        print("CCTV 스트림을 열 수 없습니다.")
        return

    model = YOLO(YOLO_MODEL_PATH)
    mask = cv2.imread(MASK_IMAGE_PATH)
    if mask is None:
        print(f"마스크 이미지를 불러올 수 없습니다: {MASK_IMAGE_PATH}")
        return

    tracker = SimpleTracker()
    tracking_times = {}
    start_ids = set()
    end_ids = set()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("영상 스트림을 불러오지 못했습니다.")
            break

        masked_frame = cv2.bitwise_and(frame, mask)
        results = model.predict(masked_frame)
        detections = filter_detections(results, CLASS_NAMES, CONFIDENCE_THRESHOLD)

        results_tracker = tracker.update(detections)
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        cv2.line(frame, tuple(START_LINE[:2]), tuple(START_LINE[2:]), (0, 0, 255), 2)
        cv2.line(frame, tuple(END_LINE[:2]), tuple(END_LINE[2:]), (0, 0, 255), 2)

        for x1, y1, w, h, obj_id in results_tracker:
            cx, cy = x1 + w // 2, y1 + h // 2
            cvzone.putTextRect(frame, f'ID: {obj_id}', (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=5)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            process_tracking(obj_id, cx, cy, current_time, tracking_times, start_ids, end_ids)

            if obj_id in tracking_times and "elapsed_time" in tracking_times[obj_id]:
                elapsed_time = tracking_times[obj_id]["elapsed_time"]
                if elapsed_time <= 0.55:
                    save_roi(frame, obj_id, elapsed_time)
                cvzone.putTextRect(frame, f'Time: {elapsed_time:.2f}s', (x1, y1 - 10), scale=1, thickness=1, offset=3, colorR=(0, 0, 255))

        # 저장된 이미지를 프레임에 합성
        frame = overlay_saved_images(frame)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
