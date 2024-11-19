import cv2

# 輸入影片檔案
input_video = 'videos/xain_2.mp4'  # 請將 'input.mp4' 替換為你的影片路徑
output_video = 'output_clip.mp4'  # 輸出的片段檔案名稱

# 設定擷取的時間範圍（單位：秒）
start_time = 5  # 開始時間
end_time = 10    # 結束時間

# 讀取影片
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("無法開啟影片檔案")
    exit()

# 取得影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 幀率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 計算要擷取的幀範圍
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# 確保時間範圍合法
if start_frame >= total_frames or end_frame > total_frames:
    print("擷取範圍超出影片長度")
    cap.release()
    exit()

# 設定影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 跳到開始幀
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 擷取範圍內的幀
current_frame = start_frame
while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像，擷取中斷")
        break

    out.write(frame)  # 寫入輸出影片
    current_frame += 1

print(f"片段已儲存為 {output_video}")

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
