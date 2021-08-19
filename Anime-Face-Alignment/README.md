
# FFHQ Face Image Alignment

**landmark detector** : 
  1. [`1adrianb/face-alignment`](https://github.com/1adrianb/face-alignment) : `pip install face-alignment`
  2. [`nagadomi/lbpcascade_animeface`](https://github.com/nagadomi/lbpcascade_animeface)
  3. [`kanosawa/anime_face_landmark_detection`](https://github.com/kanosawa/anime_face_landmark_detection)
  
  
**FFHQ alignment** : [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py)



## Result

| original images | Landmark | FFHQ-Alignment | 
| --- | --- | --- |
| <img src='raw_image/ex-1.jpg' width = '700'>  | <img src='landmark/landmark-0000.png' width = '700' > | <img src='align_image/align-0000.png' width = '700' > |
| <img src='raw_image/ex-2.jpg' width = '700'>  | <img src='landmark/landmark-0001.png' width = '700' > | <img src='align_image/align-0001.png' width = '700' > |
| <img src='raw_image/ex-4.jpg' width = '700'>  | <img src='landmark/landmark-0002.png' width = '700' > | <img src='align_image/align-0002.png' width = '700' > |
| <img src='raw_image/ex-7.jpg' width = '700'>  | <img src='landmark/landmark-0003.png' width = '700' > | <img src='align_image/align-0003.png' width = '700' > |
| <img src='raw_image/ex-5.jpg' width = '700'>  | <img src='landmark/landmark2-0000.png' width = '700' > | <img src='align_image/align2-0000.png' width = '700' > |


---

| landmark-68 | landmark-24 |
| --- | --- |
| <img src='asset/emma-landmark1.png' width = '700' > | <img src='asset/landmark-24.jpg' width = '700' ></p> |

### Comparision
<p align='center'><img src='asset/landmark-68-24.jpeg' width = '1000' ></p>

---
### 1. 68 landmark


```python
lm_chin          = lm[0  : 17, :2]  # left-right
lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
lm_eyebrow_right = lm[22 : 27, :2]  # left-right
lm_nose          = lm[27 : 31, :2]  # top-down
lm_nostrils      = lm[31 : 36, :2]  # top-down
lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

# Calculate auxiliary vectors.
eye_left     = np.mean(lm_eye_left, axis=0)
eye_right    = np.mean(lm_eye_right, axis=0)
eye_avg      = (eye_left + eye_right) * 0.5
eye_to_eye   = eye_right - eye_left
mouth_left   = lm_mouth_outer[0]
mouth_right  = lm_mouth_outer[6]
mouth_avg    = (mouth_left + mouth_right) * 0.5
eye_to_mouth = mouth_avg - eye_avg
```
**Success**

| landmark | ffhq-alignment |
| --- | --- |
| <img src='asset/landmark-68-success--1.png' width = '700' > | <img src='asset/landmark-68-success---1.png' width = '700' >|

|  |  |  |  |
| --- | --- | --- | --- |
| <img src='asset/landmark-68-2-success-1.png' > | <img src='asset/landmark-68-2-success-2.png' > | <img src='asset/landmark-68-2-success-3.png' > | <img src='asset/landmark-68-2-success-4.png' > |

---
### 2. 24 landmark

```python
lm = np.array(face_landmarks)
lm_chin          = lm[0  : 3, :2]  # left-right
lm_eyebrow_left  = lm[3  : 6, :2]  # left-right
lm_eyebrow_right = lm[6  : 9, :2]  # left-right
lm_nose          = lm[9  : 10, :2]  # top-down
lm_eye_left      = lm[10 : 15, :2]  # left-clockwise
lm_eye_right     = lm[15 : 20, :2]  # left-clockwise

# Calculate auxiliary vectors.
eye_left     = np.mean(lm_eye_left, axis=0)
eye_right    = np.mean(lm_eye_right, axis=0)
eye_avg      = (eye_left + eye_right) * 0.5
eye_to_eye   = eye_right - eye_left
mouth_left   = lm[20, :2]
mouth_right  = lm[22, :2]
mouth_avg    = (mouth_left + mouth_right) * 0.5
eye_to_mouth = mouth_avg - eye_avg
```


**success**
| landmark | ffhq-alignment |
| --- | --- |
| <img src='asset/landmark-68-success-1.png' width = '700' > | <img src='asset/landmark-68-success1.png' width = '700' >|
| <img src='asset/landmark-68-success-2.png' width = '700' > | <img src='asset/landmark-68-success2.png' width = '700' >|



**failure**
| landmark | ffhq-alignment |
| --- | --- |
| <img src='asset/landmark-24-failure--2.png' width = '700' > | <img src='asset/landmark-24-failure3.png' width = '700' >|
| <img src='asset/landmark-24-failure--1.png' width = '700' > | <img src='asset/landmark-24-failure2.png' width = '700' >|
| <img src='asset/landmark-68-success-3.png' width = '700' > | <img src='asset/landmark-68-success3.png' width = '700' >|
