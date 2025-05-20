# Driver Drowsiness Detection

### Motivation
Drowsy driving is one of the leading causes of traffic accidents worldwide. Many drivers are unaware of their fatigue levels and may continue to drive until it’s too late. According to International Automobile Association (IAA) studies, nearly one in three drivers has fallen asleep at the wheel at some point, leading to dangerous and often fatal outcomes. 

This project aims to proactively monitor driver alertness and issue real-time warnings using visual cues such as eye closure, yawning, and head tilt — helping prevent accidents before they happen. 

### Main Feature
The system includes the following key features:

1. Eye closure detection

   - Detects if the driver's eyes are closed using a trained CNN.
   - Triggers alerts if:
     * Eyes remain closed for more than 3 seconds.
     * PERCLOS (Percentage of Eye Closure) ≥ 0.7 over 1 minute.

2. Yawn detection
   
   - A separate CNN detects yawns in real-time.
   - Issues a warning if 3+ yawns are detected within 15 minutes.

3. Head tilt detection 

   - Uses facial landmark tracking (MediaPipe + OpenCV).
   - Triggers alert if the driver’s head tilts downward beyond 20°.

4. Real-time alert system 
   
   - Built with Pygame to provide on-screen warnings and audio alarms.

### Methodology

1. Eye closed detection
   
   - Dataset: [Eye_dataset](https://www.kaggle.com/datasets/charunisa/eyes-dataset/code)
   - Model: CNN with 96.56% test accuracy.
   - Logic:
      - Detects closed/open state per frame.
      - Calculates PERCLOS in rolling 1-minute windows.
      - Triggers audio alarm if:
        * PERCLOS ≥ 0.7
        * Eyes closed > 3 seconds

2. Yawn detection

   - Dataset: [Yawning_dataset](https://www.kaggle.com/datasets/deepankarvarma/yawning-dataset-classification?select=yawn)
   - Model: CNN with 97.17% test accuracy.
   - Logic:
      - Detects yawns frame-by-frame.
      - If yawning lasts ≥ 6 seconds, count it.
      - Triggers alarm after 3+ yawns within 15 minutes.

3. Head down detection
   
   - Tool: MediaPipe FaceMesh + OpenCV solvePnP()
   - Steps: 
      - Use MediaPipe FaceMesh to detect and track 468 facial landmarks in real time from video frames. From these landmarks, six key facial points (nose tip, chin, left and right eye corners, left and right mouth corners) are extracted.
      - Estimate 3D head pose.
      - Convert pose to Euler angles:
         * Pitch (up/down)
         * Yaw (left/right)
         * Roll (tilt)
      - Alert if pitch > 20° downward
   
### How to use
1. Install dependencies: ```pip install -r requirements.txt```
2. Run the main program: ```main.py```










