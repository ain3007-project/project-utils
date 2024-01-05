import matplotlib.pyplot as plt




def plot_image_mask_by_path(image_path, mask_path, figsize=(10, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(plt.imread(image_path))
    ax1.set_title('Image')
    ax1.axis('off')
    ax2.imshow(plt.imread(mask_path))
    ax2.set_title('Mask')
    ax2.axis('off')
    plt.show()

