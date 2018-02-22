"""
This script works by loading a bunch of images in the current directory,
cropping them using a clustering algorithm and placing patches randomly
in order to create a "collage".
The code is pretty straightforward to follow, don't hesitate to play
with the parameters!
"""

import os
import random
import sys

from skimage import io, segmentation, transform, img_as_float, color
import click
import numpy as np


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def approx_equal(a, b, eps=1e-6):
    return np.abs(a - b) < eps


def get_patches(img):
    """Get list of patches from image found with SLIC."""
    patches = []
    img_lab = color.rgb2lab(img)
    segmented = segmentation.slic(img_lab,
                                  n_segments=10,
                                  compactness=25.0,
                                  max_iter=10,
                                  sigma=20,
                                  convert2lab=False)
    for i in range(np.max(segmented) + 1):
        rows, cols = np.where(segmented == i)
        data = img[rows, cols]
        lab = np.mean(img_lab[rows, cols], axis=0)
        patches.append((rows, cols, data, lab))
    return patches


def resize(img, max_shape=(1280, 720)):
    """Resize while preserving aspect ratio, and fixing smallest dimension to
    `max_shape`."""
    fx = max_shape[0] / img.shape[0]
    fy = max_shape[1] / img.shape[1]
    f = min(fx, fy)
    if approx_equal(f, 1):
        return img_as_float(img)
    return transform.resize(img,
                            (int(f * img.shape[0]), int(f * img.shape[1])),
                            mode='constant', cval=0)


def paste(img, patch, pos):
    """Paste patch to image specifying center position for patch."""
    rows, cols, data, _ = patch
    height = np.ptp(rows)
    width = np.ptp(cols)
    xmin = np.min(cols)
    ymin = np.min(rows)

    for i, (x, y) in enumerate(zip(cols, rows)):
        x = x - xmin + pos[1] - width // 2
        if x < 0 or x > img.shape[1] - 1:
            continue
        y = y - ymin + pos[0] - height // 2
        if y < 0 or y > img.shape[0] - 1:
            continue
        if img[y, x, 0] == -1:
            img[y, x] = data[i]


def load_images(directory):
    """Return the list of images found in `directory`."""
    paths = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
        _, ext = os.path.splitext(filepath)
        if ext.lower() in IMG_EXTENSIONS:
            paths.append(filepath)
    if not paths:
        return []
    return io.imread_collection(paths)


@click.command()
@click.option('-d', '-directory',
              type=click.Path(exists=True, file_okay=False),
              help='Directory containing the images',
              default='images',
              show_default=True)
@click.option('-r', '--reuse', is_flag=True, default=False,
              help='Re-use already selected patches',
              show_default=True)
@click.option('-w', '--width', default=1280,
              help='Output image width',
              show_default=True)
@click.option('-h', '--height', default=720,
              help='Output image height',
              show_default=True)
def main(**kwargs):
    """Collotron: Automatic collage application."""

    print('Loading images...', end=' ')
    sys.stdout.flush()
    images = load_images(kwargs['directory'])
    if not images:
        print()
        print('Error: no images found')
        sys.exit(1)
    print('done ({:d} images loaded)'.format(len(images)))

    print('Resizing images...', end=' ')
    sys.stdout.flush()
    images = [resize(img) for img in images]
    print('done')

    print('Extracting patches...', end=' ')
    sys.stdout.flush()
    patches = [p for img in images for p in get_patches(img)]
    print('done ({} patches extracted)'.format(len(patches)))

    width = kwargs['width']
    height = kwargs['height']
    collage = np.ones((height, width, 3))
    collage[:] = -1
    while 1:
        if not kwargs['reuse']:
            if not patches:
                break
        patch_idx, patch = random.choice(list(enumerate(patches)))
        if not kwargs['reuse']:
            patches.pop(patch_idx)
        rows, cols = np.where(collage[:, :, 0] == -1)
        if rows.size == 0:
            break
        idx = random.choice(list(enumerate(cols)))[0]
        position = (rows[idx], cols[idx])
        print(' > Pasting patch {:d} at {:d}x{:d}'.format(
            patch_idx, position[0], position[1]))
        paste(collage, patch, position)

    collage[collage == -1] = 0
    print('Over! File written to collage.jpg')
    io.imsave('collage.jpg', collage)
    io.imshow(collage)
    io.show()


if __name__ == "__main__":
    main()
