import cv2
from ultralytics import YOLO
import time


class Bone_recognize:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.position = {}
        self.fps = 0

    def raise_hand(self, result):
        left_shoulder = result[0].keypoints.xy[0][5]
        left_hand = result[0].keypoints.xy[0][9]
        right_shoulder = result[0].keypoints.xy[0][6]
        right_hand = result[0].keypoints.xy[0][10]
        print(f"left_shoulder: {left_shoulder[1]}\nright_shoulder: {right_shoulder[1]}\nleft_hand: {left_hand[1]}\nright_hand: {right_hand[1]}")

        left_hand_raised = left_hand[1] < left_shoulder[1]  
        right_hand_raised = right_hand[1] < right_shoulder[1]  
        
        
        if left_shoulder[1] == 0 or left_hand[1] == 0:
            return False, right_hand_raised
        if right_shoulder[1] == 0 or right_hand[1] == 0:
            return left_hand_raised, False
        
        return left_hand_raised, right_hand_raised

    def center(self, result, total_time):
        box = result[0].boxes
        if box:
            if box.id != None:
                for i in range(len(box.id)):
                    ID = int(box.id[i].item())
                    ID_name = self.model.names[int(box.cls[i].item())]
                    x1, y1, x2, y2 = box.xyxy[i][0].item(), box.xyxy[i][1].item(), box.xyxy[i][2].item(), box.xyxy[i][3].item()
                    center = [(x1 + x2) / 2, (y1 + y2) / 2]

                    if ID in self.position:
                        if len(self.position[ID]) == 1:
                            self.position[ID].append(center)
                        elif len(self.position[ID]) == 2:
                            self.position[ID].pop(0)
                            self.position[ID].append(center)
                            
                            distance = ((self.position[ID][1][0] - self.position[ID][0][0]) ** 2+(self.position[ID][1][1] - self.position[ID][0][1]) ** 2) ** 0.5
                            print("ID",ID,end=" ")
                            print("name:",ID_name)
                            print("目前座標",center)
                            print("與上次座標之距離",distance)
                            print("速度", distance / total_time)

                    elif ID not in self.position:
                        self.position[ID] = [center]
                        
        # with open('example.txt', 'w') as f:
        #     for i in self.position.items():
        #         f.write(f"{i}/n")

    def run(self):
        avg_fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # 幀率
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        recording = False
        ready_to_stop = False
        recording_complete = True
        recording_hint = False
        all_clip_time = []
        
        while self.cap.isOpened():
            sucess, frame = self.cap.read()
            if sucess:


                start = time.perf_counter()

                result = self.model.track(frame, conf=0.3, verbose=False)
                annotated_frame = result[0].plot()
                
                end = time.perf_counter()
                total_time = end - start
                fps = 1 / total_time

                left_hand_raised, right_hand_raised = self.raise_hand(result)
                # 0:nose,1:left eye,2:right-eye,3:left-ear,4:right-ear,5:left-shoulder,6:right-shoulder,7:left-elbow,8:right-elbow,
                # 9:left-wrist,10:right-wrist,11:left-hip,12:right-hip,13:left-knee,14:right-knee, 15:left-ankle,16:right-ankle
                
                if recording_hint == True:
                    cv2.circle(annotated_frame, (1230, 680), 13, (0, 0, 255), -1)
                    line_length = 100  # 每個角的線長
                    line_thickness = 2  # 線條粗細
                    color = (255, 255, 255)  # 白色角標

                    # 左上角
                    cv2.line(annotated_frame, (10, 10), (line_length, 10), color, line_thickness) 
                    cv2.line(annotated_frame, (10, 10), (10, line_length), color, line_thickness)

                    # 右上角
                    cv2.line(annotated_frame, (1270, 10), (1270 - line_length, 10), color, line_thickness)  
                    cv2.line(annotated_frame, (1270, 10), (1270, line_length), color, line_thickness) 

                    # 左下角
                    cv2.line(annotated_frame, (10, 710), (line_length, 710), color, line_thickness)  
                    cv2.line(annotated_frame, (10, 710), (10, 710 - line_length), color, line_thickness) 

                    # 右下角
                    cv2.line(annotated_frame, (1270, 710), (1270 - line_length, 710), color, line_thickness)  
                    cv2.line(annotated_frame, (1270, 710), (1270, 710 - line_length), color, line_thickness) 
                    
                if left_hand_raised == True or right_hand_raised == True:
                    if recording == False and recording_complete == True:
                        recording = True
                        recording_hint = True
                        recording_complete = False
                        
                        start_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) # 開始擷取片段時間
                        
                    if (recording == True and ready_to_stop == True) or (recording == True and int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) >= total_frames):
                        recording = False
                        recording_hint = False
                        ready_to_stop = False
                        
                        end_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) # 結束擷取片段時間
                        all_clip_time.append([start_frame, end_frame])
 
                    cv2.putText(annotated_frame, "Hand Raised", (1050, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("Hand Raised: True",end="\n\n")
                          
                else:
                    if recording:
                        ready_to_stop = True
                        
                    if recording == False and ready_to_stop == False:
                        recording_complete = True
                        
                    print("Hand Raised: False",end="\n\n")
                    
                self.center(result, total_time)
                   
                print("="*40)
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Bone recognize", annotated_frame)

            # if cv2.waitKey(1) & 0xFF == ord("q") or int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) >= total_frames:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                clip_counter = 0
                for clip_time in all_clip_time:
                    # 片段影片設定
                    clip_counter += 1
                    clip_filename = f'./videos/output/output_clip_{clip_counter}.mp4'
                    out = cv2.VideoWriter(clip_filename, cv2.VideoWriter_fourcc(*'mp4v'), avg_fps, (frame_width, frame_height))
                    
                    # 寫入片段影片
                    clip_cap = cv2.VideoCapture(self.video_path)
                    clip_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    current_frame = clip_time[0]
                    while current_frame < clip_time[1] + avg_fps * 1:
                        ret, frame = clip_cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        current_frame += 1
                    clip_cap.release()
        
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo = Bone_recognize("models/yolov8n-pose.pt", "videos/xain_2.mp4")
    # demo = Bone_recognize("models/yolov8n-pose.pt", 0)
    demo.run()
    
