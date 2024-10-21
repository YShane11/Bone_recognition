import cv2
from ultralytics import YOLO
import time


class Bone_recognize:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
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
                
   
                if left_hand_raised == True or right_hand_raised == True:
                    print("Hand Raised: True",end="\n\n")
                    cv2.putText(annotated_frame, "Hand Raised", (1050, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("Hand Raised: False",end="\n\n")
 
                    
                self.center(result, total_time)
                   
                print("="*40)

                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Bone recognize", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo = Bone_recognize("models/yolov8n-pose.pt", "videos/xain_2.mp4")
    demo.run()
    
