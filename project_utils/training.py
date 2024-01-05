import torch
import os
import zipfile
import huggingface_hub as hfh
import shutil
from sklearn.model_selection import train_test_split

def download_and_extract_images(repo_id, target_dir):
        print("started downloading files")
        images_zip_path = hfh.hf_hub_download(
            repo_id=repo_id,
            filename="images.zip",
            repo_type="dataset",
        )
        
        masks_zip_path = hfh.hf_hub_download(
            repo_id=repo_id,
            filename="masks.zip",
            repo_type="dataset",
        )
        
        metadata_path = hfh.hf_hub_download(
            repo_id=repo_id,
            filename="metadata.csv",
            repo_type="dataset",
        )
        
        os.makedirs(target_dir, exist_ok=True)
        shutil.copyfile(metadata_path, target_dir + "/metadata.csv")
        
        print("started extracting")
        with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        with zipfile.ZipFile(masks_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("finished")



def prepare_image_lists(train_dir="data/train", test_dir="data/test", panda_dir="data/panda"):
    """
    Prepares the image lists for training and testing
    :param train_dir: path to the training directory
    :param test_dir: path to the testing directory
    :return: train_list, test_list
    """
    train_images = os.listdir(f"{train_dir}/images")
    train_masks = os.listdir( f"{train_dir}/masks")
    test_images = os.listdir(f"{test_dir}/images")
    test_masks = os.listdir(f"{test_dir}/masks")

    # in train dataset there are some files with no mask and we should filter those
    train_images = [i for i in train_images if i in train_masks]

    train_images = [f"{train_dir}/images/{image}" for image in train_images]
    train_masks = [f"{train_dir}/masks/{mask}" for mask in train_masks]
    test_images = [f"{test_dir}/images/{image}" for image in test_images]
    test_masks = [f"{test_dir}/masks/{mask}" for mask in test_masks]


    panda_images = list(zip(sorted(panda_images), sorted(panda_masks)))
    train_images = list(zip(sorted(train_images), sorted(train_masks)))
    test_images = list(zip(sorted(test_images), sorted(test_masks)))


    if panda_dir is not None:
        panda_images = os.listdir(f"{panda_dir}/images")
        panda_masks = os.listdir(f"{panda_dir}/masks")

        panda_images = [f"{panda_dir}/images/{image}" for image in panda_images]
        panda_masks = [f"{panda_dir}/masks/{mask}" for mask in panda_masks]

        panda_images = list(zip(sorted(panda_images), sorted(panda_masks)))
    
        # add panda images to train 
        train_images = train_images + panda_images


    # split train images into train and validation
    train_images, valid_images = train_test_split(train_images, test_size=0.2, random_state=42)

    return train_images, valid_images, test_images


