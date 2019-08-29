from PIL import Image
import pytesseract

path = "Wechat2.jpeg"

text = pytesseract.image_to_string(Image.open(path), lang='chi_sim')
print(text)
print("sss")