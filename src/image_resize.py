from PIL import Image
import os
from tqdm import tqdm

def resize_and_crop_image(input_path, output_path, size=(512, 512)):
    try:
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            target_width, target_height = size
            
            img_ratio = img.width / img.height #根据宽高比决定缩放方式
            target_ratio = target_width / target_height

            if img_ratio > target_ratio: #图片过宽，按高度缩放
                new_height = target_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = target_width #图片过高或比例相等，按宽度缩放
                new_height = int(new_width / img_ratio)

            img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

            left = (new_width - target_width) / 2
            top = (new_height - target_height) / 2
            right = left + target_width
            bottom = top + target_height

            img_cropped = img_resized.crop((left, top, right, bottom))

            img_cropped.save(output_path)
    except Exception as e:
        print(f"处理图片时出错: {input_path}, 错误信息: {e}")

def resize_images_in_folder(input_folder, output_folder, size=(512, 512)):
    
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    for filename in tqdm(files, desc="处理图片", unit="张"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        resize_and_crop_image(input_path, output_path, size)


if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = ""
    # 输出文件夹路径
    output_folder = ""

    resize_images_in_folder(input_folder, output_folder)
