from PIL import Image, ImageDraw
import numpy as np
import random
import os

# 원본 이미지들이 저장된 디렉토리
original_images_dir = './mnt/data/X_'
original_images = [os.path.join(original_images_dir, f) for f in os.listdir(original_images_dir) if f.endswith('.png')]

# 변형 이미지를 저장할 디렉토리
output_dir = './mnt/data/X'
os.makedirs(output_dir, exist_ok=True)

# 이미지에 무작위 변형을 적용하는 함수
def create_variation(image):
    # numpy 배열로 변환
    img_array = np.array(image)
    
    # 무작위 변형
    angle = random.uniform(-5, 5)  # -5도에서 5도 사이의 무작위 회전
    translate_x = random.uniform(-5, 5)  # x축 방향의 무작위 평행 이동
    translate_y = random.uniform(-5, 5)  # y축 방향의 무작위 평행 이동
    scale = random.uniform(0.95, 1.05)  # 0.95배에서 1.05배 사이의 무작위 크기 조정
    
    # 동일한 크기와 흰색 배경의 새 이미지 생성
    new_image = Image.new('L', image.size, 'white')
    draw = ImageDraw.Draw(new_image)
    
    # 회전
    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=0)
    
    # 크기 조정
    w, h = rotated_image.size
    scaled_image = rotated_image.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
    
    # 평행 이동
    translated_image = Image.new('L', image.size, 'white')
    translated_image.paste(scaled_image, (int(translate_x), int(translate_y)))
    
    return translated_image

# 각 원본 이미지에 대해 1,000개의 변형 생성 및 저장
j = 0
for idx, original_image_path in enumerate(original_images):
    original_image = Image.open(original_image_path)
    for i in range(1000):
        variation = create_variation(original_image)
        variation.save(os.path.join(output_dir, f'X{j+1}.png'))
        j += 1
