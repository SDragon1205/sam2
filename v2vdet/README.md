# YOLO-OO: An One-Shot Object-Oriented YOLO-based Detector

## Introduction
This paper presents YOLO-OO, an One-shot Object-oriented YOLO-based detector capable of identifying similar objects in other images after users provide exemplar items, without requiring additional training. While existing YOLO models excel in detection speed and accuracy, they still demand substantial labeled data for training, creating time and resource challenges for customized applications. Although Open-Vocabulary approaches attempt to address class limitation issues, they remain inherently too generic for precise identification of user-specific targets. Recently, with the rapid development of LLMs/VLMs, numerous high-quality vision encoders have emerged, generating superior image embeddings with excellent performance in downstream visual tasks. However, these encoders, due to their substantial parameter sizes, typically struggle to achieve real-time inference, limiting their practical application value.

## Key Contributions
Our research makes three primary contributions:

### One-shot Object Orientation Task
Unlike the ambiguity of open-vocabulary methods, our proposed task allows users to explicitly specify search targets by providing template images without requiring extensive pre-training samples, enabling more precise detection aligned with user requirements and accurately distinguishing provided objects even when traditionally considered similar

### Multi-Level Patch Visual Prompt Encoder (MAVPE)
 We develop a novel embedding generation method integrating multi-layer features from vision encoders with boundary box visual prompts to produce high-quality template embeddings, enhancing detection precision

### Object-Oriented Training Methodology
We train and validate by attaching images of the same object from different angles to both input and template images, enabling the model to transcend traditional categorical classification and achieve high-precision sample detection based on user-provided exemplars.
Code will be made available at a public repository upon publication.

## Demo

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/6a2c6dbe-3cdf-4d9d-9217-bbea6b350308" width="400">
  <img src="https://github.com/user-attachments/assets/a8caa49c-3af5-41e6-af1b-17b9ddd71ec3" width="400">

Youtube Demo Link: https://youtube.com/shorts/087Y0V9e6cA?feature=share