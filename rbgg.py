import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(
    description='Generating random backgrounds for a given image.')
parser.add_argument('path', help='path to an image to process')
parser.add_argument('-o', '--out-dir', default='out', help='output directory')
parser.add_argument('-n', '--number', type=int, default=2,
                    help='number of images to generate')
parser.add_argument('-m', '--mask', action="store_true",
                    help='whether to save masks')
parser.add_argument('-mo', '--mask-out-dir', default='mask',
                    help='mask output directory')
parser.add_argument('--margin-low', type=float, help='margin low')
parser.add_argument('--margin-high', type=float, help='margin high')
parser.add_argument('--margins-equal', action="store_true",
                    help='equal margins')
parser.add_argument('--scale-low', type=float, help='scale low')
parser.add_argument('--scale-high', type=float, help='scale high')


def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else c[1]
        slices.append(slice(-c[0], e))
    return x[tuple(slices)]


def get_random_cutout(image, scale_low, scale_high, margin_low, margin_high, margins_equal):
    if scale_low or scale_high:
        if not scale_low:
            scale_low = scale_high
        if not scale_high:
            scale_high = scale_low
        scale = random.uniform(scale_low, scale_high)
        image = cv2.resize(image, (image.shape[1]*scale, image.shape[0]*scale))

    if margin_low or margin_high:
        if not margin_low:
            margin_low = margin_high
        if not margin_high:
            margin_high = margin_low
        if margins_equal:
            margin = random.unform(margin_low/2, margin_high/2)
            a, b, c, d = int(margin*image.shape[0]), int(margin*image.shape[0]), int(
                margin*image.shape[1]), int(margin*image.shape[1])
            image = np.pad(image, ((max(0, a), max(0, b)),
                                   (max(0, c), max(0, d)), (0, 0)))
            image = unpad(image, ((a, b), (c, d), (0, 0)))
        else:
            def rndm(): return random.uniform(margin_low/2, margin_high/2)
            a, b, c, d = int(rndm()*image.shape[0]), int(rndm()*image.shape[0]), int(
                rndm()*image.shape[1]), int(rndm()*image.shape[1])
            image = np.pad(image, ((max(0, a), max(0, b)),
                                   (max(0, c), max(0, d)), (0, 0)))
            image = unpad(image, ((min(0, a), min(0, b)),
                                  (min(0, c), min(0, d)), (0, 0)))

    # if random.random() > 0.8:
        # blur_rate = max(1, random.randrange(
        # min(cutout.shape[0], cutout.shape[1])//15))
        # cutout = cv2.blur(cutout, (blur_rate, blur_rate))
    return image


def get_random_solid_background(im):
    random_color = [random.randrange(
        256), random.randrange(256), random.randrange(256)]
    return np.full((im.shape[0], im.shape[1], 3), random_color)
    # return np.full((im.shape[0]+random.randrange(-im.shape[0]//4, im.shape[0]//3), im.shape[1]+random.randrange(-im.shape[1]//4, im.shape[1]//3), 3), random_color)


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]
    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1],
                         1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * \
        background[y:y+h, x:x+w] + mask * overlay_image
    full_mask = np.full(background.shape, 0.)
    full_mask[y:y+h, x:x+w] = mask
    full_mask *= 255
    return background, full_mask


if __name__ == '__main__':
    args = parser.parse_args()
    im = cv2.imread(args.path, cv2.IMREAD_UNCHANGED)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.mask_out_dir, exist_ok=True)

    for i in range(args.number):
        cutout = get_random_cutout(im, args.scale_low, args.scale_high,
                                   args.margin_low, args.margin_high, args.margins_equal)
        background = get_random_solid_background(cutout)
        # x = random.randrange(max(1, background.shape[1]-cutout.shape[1]))
        # y = random.randrange(max(1, background.shape[0]-cutout.shape[0]))
        added_image, mask = overlay_transparent(background, cutout, 0, 0)
        added_image = added_image.astype('uint8')

        cv2.imwrite(f'{args.out_dir}/{i}.jpg', added_image)
        if args.mask:
            cv2.imwrite(f'{args.mask_out_dir}/{i}.png', mask)
