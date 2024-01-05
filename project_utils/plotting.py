import matplotlib.pyplot as plt




def plot_image_mask_by_path(image1_path, image2_path, figsize=(10, 5)):
    def imshow(image, mask):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fifgsize)
    ax1.imshow(plt.imread(image))
    ax1.set_title('Image')
    ax1.axis('off')
    ax2.imshow(plt.imread(mask))
    ax2.set_title('Mask')
    ax2.axis('off')
    plt.show()


