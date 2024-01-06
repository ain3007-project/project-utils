import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import openslide
import shutil
import pandas as pd
import huggingface_hub as hfh

def str2float(x):
    return float(x.replace(",", "."))
    
def get_mask_from_xml(xml_path, image_size, image_shrinking_factor):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image = Image.new("L", image_size, "black")
    draw = ImageDraw.Draw(image)
    draw.fill = True
    label2grayscale_color = {"bg": 0, "tissue": 255, "tisuue": 255}
    for i in root[0]:
        annotation_type = i.attrib["Type"]
        annotation_label = i.attrib["PartOfGroup"]
        # there is roi rectangle
        if annotation_type not in ["Spline", "Polygon", "Rectangle"]:
            print(f"Annotation type must be either Spline, Rectangle or Polygon but found: {annotation_type}")
            continue
            
        if annotation_label not in label2grayscale_color:
            print(f"Annotation label must be either tissue or bg but found: {annotation_label}")
            continue
        
        coordinates = [(i.attrib["X"], i.attrib["Y"]) for i in i[0]]
        coordinates = [(str2float(x), str2float(y)) for x, y in coordinates]
        coordinates = [(x*image_shrinking_factor, y*image_shrinking_factor) for x, y in coordinates]
        
        if annotation_type in ["Spline", "Polygon"]:
            draw.polygon(coordinates, fill=label2grayscale_color[annotation_label])
        elif annotation_type == "Rectangle":
            # ^
            # |         point 1 is bigger than point 3
            # | 0 1
            # | 3 2 
            # |------->
            draw.rectangle([coordinates[3], coordinates[1]], fill=label2grayscale_color[annotation_label])
#         if annotation_type == "Spline":
#             draw.line(coordinates, fill=label2grayscale_color[annotation_label], width=1)
#         elif annotation_type == "Polygon":
#             draw.polygon(coordinates, fill=label2grayscale_color[annotation_label])
    return image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



def process(file_list, image_idx, metadata_df, thumbnail_size=(2000, 2000), source_folder_name="Breast1__he", pad_images=False, raw_repo_id=None, image_padding_color="white", mask_padding_color="black"):
    for allow_pattern, image_path, xml_path in file_list:
        path = hfh.snapshot_download(
            repo_id=raw_repo_id, 
            repo_type="dataset", 
            allow_patterns=allow_pattern,
            cache_dir="hf_folder",
        )
#         print(os.path.join(path, image_path))
        try:
            slide_file = openslide.OpenSlide(os.path.join(path, image_path))
        except openslide.OpenSlideUnsupportedFormatError:
            print(f"image {image_path} didnt get recognized by OpenSlide. OpenSlideUnsupportedFormatError")
            image_idx += 1
            shutil.rmtree('hf_folder')
            continue
        except openslide.OpenSlideError:
            print(f"image {image_path} didnt get recognized by OpenSlide. OpenSlideError")
            image_idx += 1
            shutil.rmtree('hf_folder')
            continue


        thumbnail = slide_file.get_thumbnail(thumbnail_size)
        thumbnail_size = thumbnail.size
        if pad_images:
            thumbnail = expand2square(thumbnail, image_padding_color)
        
        thumbnail.save(f"images/{image_idx}.png")

        image_shrinking_factor = min(thumbnail_size) / min(slide_file.dimensions)
        
        if xml_path is not None:
            mask = get_mask_from_xml(
                os.path.join(path, xml_path),
                thumbnail_size,
                image_shrinking_factor
            )
            if pad_images:
                mask = expand2square(mask, mask_padding_color)
            mask.save(f"masks/{image_idx}.png")
            
            
        metadata_df.loc[len(metadata_df)] = [
            image_idx,
            image_path,
            source_folder_name,
            slide_file.dimensions[0],
            slide_file.dimensions[1],
            slide_file.level_count,
            slide_file.properties["openslide.vendor"],
            image_shrinking_factor,
        ]
            
        shutil.rmtree('hf_folder')
        print(f"finished {image_idx} image of {source_folder_name}")
        image_idx += 1
        
    return image_idx, metadata_df