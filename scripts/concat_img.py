from PIL import Image
import os

def concat_images_horizontally(image_paths, gap=10, gap_color=(255, 255, 255)):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    widths, heights = zip(*(img.size for img in images))
    
    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)
    
    new_img = Image.new('RGB', (total_width, max_height), color=gap_color)
    
    x_offset = 0
    for i, img in enumerate(images):
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
        if i != len(images) - 1:
            x_offset += gap 

    return new_img

image_names = [
    'original.png',
    'pretrain.png',
    'MAE-1-pixelnorm-soft-f0.png',
    'MAE-1-pixelnorm-soft-f2.png',
    'MAE-1-pixelnorm-soft-f5.png',
    'MAE-1-pixelnorm-soft-f8.png',
    'MAE-1-pixelnorm-soft-f11.png',
]

for dir in os.listdir("visualize"):
    if os.path.isdir(os.path.join("visualize", dir)):
        image_paths = [os.path.join("visualize", dir, name) for name in image_names]
        output_path = os.path.join("visualize", dir, "concat.png")
        
        concat_img = concat_images_horizontally(image_paths)
        concat_img.save(output_path)
        print(f"Saved concatenated image to {output_path}")