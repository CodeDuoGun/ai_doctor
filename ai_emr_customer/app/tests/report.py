import requests
import os
import json

from app.utils.log import logger

def download_img(img_url, img_type):
    """临时报错数据到本地"""

def cache_data(cache_dir, data, file_name):
    """cache image/audio/video"""
    os.makedirs(cache_dir, exist_ok=True)
    with open(file_name, 'wb') as img_file:
        img_file.write(data)
    return img_file

def read_data():
    with open("app/tests/person.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def handle_text(text):
    """处理问诊单文本内容"""
    # format json
    return text


def ocr_extract_text(image_path:str=None):
    reader = easyocr.Reader(['ch_tra','en'], gpu=False) # this needs to run only once to load the model into memory
    result = reader.readtext('EasyOCR/test_img.jpg')
    print(result)

def handle_image(image_url, cache_dir):
    """处理问诊单图片"""
    # 下载图片并保存到缓存目录
    response = requests.get(image_url)
    if response.status_code == 200:
        img_path = os.path.join(cache_dir, os.path.basename(image_url))
        cache_data(cache_dir, response.content, img_path)
    else:
        logger.error(f"Failed to download image from {image_url}")
        return
    # ocr 处理图片

    # 返回后处理的文本
    res = ""
    return res


def extract_text_from_image():
    # -*- coding:utf-8 -*-
    import pytesseract
    from PIL import Image
    import re
    import time


    tot_time = []
    with open('output1.txt', 'w', encoding='utf-8') as file:
        while True:
            if int(input("输入1继续，输入0退出：")) == 0:
                break
            start_time = time.time()
            image = Image.open("EasyOCR/test_img.jpg")
            gray_image = image.convert('L')
            text = pytesseract.image_to_string(gray_image, lang='chi_sim')  # 图片转字符串
            print(text)
            end_time = time.time()
            print(f'time = {end_time - start_time}')
            tot_time.append(end_time - start_time)
            file.write(text + '\n')
            file.write('-' * 50 + '\n')

    avg_time = sum(tot_time) / len(tot_time)
    print(f'平均解析时间: {avg_time}')

    with open('output1.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    text = re.sub(r' +', ' ', text)  # 空格处理

    with open('output1.txt', 'w', encoding='utf-8') as file:
        file.write(text)





def generate_report():
    data = read_data()
    logger.debug(f"json data: {data}")

    uid = data["patient_id"]
    cache_dir = f"app/cache/{uid}"

    # 问诊单文本内容
    
    
    # 问诊单图片处理 
    #   


if __name__ == "__main__":
    # ocr_extract_text()
    extract_text_from_image()
    # generate_report()