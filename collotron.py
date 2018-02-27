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

from skimage import io, segmentation, transform, img_as_float, color, filters
from skimage.future import graph

import click
import numpy as np


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def weight_boundary(graph, src, dst, n):
    default = {'weight': 0.0, 'count': 0}
    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']
    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']
    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    pass


def approx_equal(a, b, eps=1e-6):
    """Compare two floating point number."""
    return np.abs(a - b) < eps


def init_collage(height, width):
    collage = np.ones((height, width, 3))
    collage[:, :, :] = -1
    return collage


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def paint_background(collage, patches):
    """Use main cluster patches to paint a homogeneous background."""
    patch_idxs = list(range(len(patches)))
    while 1:
        patch_idx = random.choice(patch_idxs)
        patch = patches[patch_idx]
        rows, cols = np.where(collage[:, :, 0] == -1)
        n = rows.size
        if n == 0:
            break
        idx = random.choice(list(range(n)))
        position = (rows[idx], cols[idx])
        # print(' > Pasting patch {:d} at {:d}x{:d}'.format(
        #     patch_idx, position[0], position[1]))
        paste(collage, patch, position)
    return


def get_patches(img, compactness=30, n_segments=200, rag_thresh=0.08):
    """Get list of patches from image found with SLIC."""
    patches = []
    img_lab = color.rgb2lab(img)
    edges = filters.sobel(color.rgb2gray(img))
    labels = segmentation.slic(img_lab, convert2lab=False,
                               compactness=compactness, n_segments=n_segments)
    g = graph.rag_boundary(labels, edges)
    segmented = graph.merge_hierarchical(labels, g, thresh=rag_thresh,
                                         rag_copy=False,
                                         in_place_merge=True,
                                         merge_func=merge_boundary,
                                         weight_func=weight_boundary)

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
    xmin = np.min(cols)
    ymin = np.min(rows)
    xmax = np.max(cols)
    ymax = np.max(rows)
    height = ymax - ymin
    width = xmax - xmin

    cols_im = cols - xmin + pos[1] - width // 2
    rows_im = rows - ymin + pos[0] - height // 2
    mask = np.where((cols_im > -1)
                    & (cols_im < img.shape[1])
                    & (rows_im > -1)
                    & (rows_im < img.shape[0]))[0]
    cols_im, rows_im = cols_im[mask], rows_im[mask]
    mask1 = np.where(img[rows_im, cols_im, 0] == -1)
    rows_im, cols_im = rows_im[mask1], cols_im[mask1]
    img[rows_im, cols_im] = data[mask][mask1]


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
@click.option('-w', '--width', default=1280,
              help='Output image width',
              show_default=True)
@click.option('-h', '--height', default=720,
              help='Output image height',
              show_default=True)
@click.option('-t', '--thresh', default=0.08)
@click.option('--show', is_flag=True, default=False)
@click.option('--save', is_flag=True, default=True)
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
    patches = [p for img in images
               for p in get_patches(img, rag_thresh=kwargs['thresh'])]
    print('done ({} patches extracted)'.format(len(patches)))

    collage = init_collage(kwargs['height'], kwargs['width'])

    print('Painting background...', end=' ')
    sys.stdout.flush()
    paint_background(collage, patches)
    print('done')

    print('Over! File written to collage.jpg')
    if kwargs['save']:
        io.imsave('collage.jpg', collage)
    if kwargs['show']:
        io.imshow(collage)
        io.show()


if __name__ == "__main__":
    main()
