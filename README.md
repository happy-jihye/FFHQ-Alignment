# FFHQ-Alignment


## 1. Face Alignment


| original images | alignment images | alignment (no padding) | 
| --- | --- | --- |
| <img src='FFHQ-Alignmnet/raw_images/celeb-1.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/aligned_images/align-celeb-1.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/no_padding_images/align-celeb-1.jpg' width = '700' >  |
| <img src='FFHQ-Alignmnet/raw_images/celeb-2.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/aligned_images/align-celeb-2.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/no_padding_images/align-celeb-2.jpg' width = '700' >  |
| <img src='FFHQ-Alignmnet/raw_images/celeb-3.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/aligned_images/align-celeb-3.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/no_padding_images/align-celeb-3.jpg' width = '700' >  |
| <img src='FFHQ-Alignmnet/raw_images/celeb-4.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/aligned_images/align-celeb-4.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/no_padding_images/align-celeb-4.jpg' width = '700' >  |
| <img src='FFHQ-Alignmnet/raw_images/celeb-5.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/aligned_images/align-celeb-5.jpg' width = '700' >  |  <img src='FFHQ-Alignmnet/no_padding_images/align-celeb-5.jpg' width = '700' >  |
| <img src='FFHQ-Alignmnet/raw_images/celeb-6.jpeg' width = '700' >  |  <img src='FFHQ-Alignmnet/aligned_images/align-celeb-6.jpeg' width = '700' >  |  <img src='FFHQ-Alignmnet/no_padding_images/align-celeb-6.jpeg' width = '700' >  |

---

## 2. Anime Alignment


| original images | Landmark | FFHQ-Alignment | 
| --- | --- | --- |
| <img src='Anime-Face-Alignment/raw_image/ex-1.jpg' width = '700'>  | <img src='Anime-Face-Alignment/landmark/landmark-0000.png' width = '700' > | <img src='Anime-Face-Alignment/align_image/align-0000.png' width = '700' > |
| <img src='Anime-Face-Alignment/raw_image/ex-2.jpg' width = '700'>  | <img src='Anime-Face-Alignment/landmark/landmark-0001.png' width = '700' > | <img src='Anime-Face-Alignment/align_image/align-0001.png' width = '700' > |
| <img src='Anime-Face-Alignment/raw_image/ex-4.jpg' width = '700'>  | <img src='Anime-Face-Alignment/landmark/landmark-0002.png' width = '700' > | <img src='Anime-Face-Alignment/align_image/align-0002.png' width = '700' > |
| <img src='Anime-Face-Alignment/raw_image/ex-7.jpg' width = '700'>  | <img src='Anime-Face-Alignment/landmark/landmark-0003.png' width = '700' > | <img src='Anime-Face-Alignment/align_image/align-0003.png' width = '700' > |
| <img src='Anime-Face-Alignment/raw_image/ex-5.jpg' width = '700'>  | <img src='Anime-Face-Alignment/landmark/landmark2-0000.png' width = '700' > | <img src='Anime-Face-Alignment/align_image/align2-0000.png' width = '700' > |

---

## 3. Landmark Detector

<p align='center'><img src='Landmark-detector/landmark/image_with_landmark.gif' width = '900' ></p> 


---


## Reference

**landmark detector** : 
  1. [`1adrianb/face-alignment`](https://github.com/1adrianb/face-alignment) : `pip install face-alignment`
  2. [`nagadomi/lbpcascade_animeface`](https://github.com/nagadomi/lbpcascade_animeface)
  3. [`kanosawa/anime_face_landmark_detection`](https://github.com/kanosawa/anime_face_landmark_detection)
  
**FFHQ alignment** : [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py)
