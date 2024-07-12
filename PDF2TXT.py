from tika import parser
pdf_path = "Lecture.생각의 근육 만들기 by 유훈 코치 - 교재用 (1).pdf"
parsed = parser.from_file(pdf_path)
txt = open('output.txt', 'w', encoding = 'utf-8')
print(parsed['content'], file = txt)
txt.close()