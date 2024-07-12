import os
from pdf2image import convert_from_path

# 저장할 디렉토리 설정
#save_dir = 'C:/Users/erid3/Documents/Workspace/Python/ai/images'
save_dir = './images'
os.makedirs(save_dir, exist_ok=True)

# PDF 파일 경로
#pdf_path = 'C:/Users/erid3/Documents/Workspace/Python/ai/pdf/GOVCBR929V1.pdf'
pdf_path = './pdf/GOVCBR929V1.pdf'

# PDF를 이미지로 변환
images = convert_from_path(pdf_path)

# 변환된 이미지를 파일로 저장
for i, image in enumerate(images):
    image.save(os.path.join(save_dir, f'page_{i + 1}.png'), 'PNG')

print("PDF가 이미지로 성공적으로 변환되었습니다.")
