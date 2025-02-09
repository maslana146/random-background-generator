import os
import random
import random as ran
import string
import argparse
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from nltk.corpus import words as wn


def get_random_string(length):
    return ''.join(ran.choice(string.ascii_letters) for i in range(length))


def random_color():
    return tuple([ran.randint(0, 256) for i in range(3)])


def get_fonts_from_dir(dir):
    return [i for i in os.listdir(dir) if i.endswith('.ttf')]


def get_fonts(n):
    drop_fonts = ['MTEXTRA', 'symbol', 'REFSPCL', 'wingding', 'WINGDNG3', 'segmdl2', 'OUTLOOK', 'holomdl2', 'marlett',
                  'webdings', 'WINGDNG2', 'BSSYM7']
    fonts = list(
        ran.sample(
            [i for i in fm.findSystemFonts(fontpaths=None, fontext='ttf') if not any(j in i for j in drop_fonts)], n))
    return [i.split('\\')[-1] for i in fonts]


def gen_fonts(output_dir, size_range, text, fonts_number):
    os.makedirs(output_dir, exist_ok=True)
    if args.system_fonts:
        function = get_fonts(fonts_number)
    else:
        function = get_fonts_from_dir('fonts_files')
    for font_name in function:
        if args.text is None:
            text = random.choice(wn.words())
        else:
            text = text
        for size in range(size_range[0], size_range[1], size_range[2]):
            color = random_color()
            filename = f'{output_dir}/{font_name[:len(font_name) - 4]}_{size}_{color}.png'
            if args.system_fonts:
                font = ImageFont.truetype(font_name, size)
            else:
                font = ImageFont.truetype(os.path.join('fonts_files', font_name), size)
            try:
                width, height = font.getsize(text)
            except OSError:
                break
            width, height = int(width), int(height)
            wm = Image.new('RGBA', (width + 10, height + 10), (0, 0, 0, 0))
            im = Image.new('RGBA', (width + 10, height + 10), (0, 0, 0, 0))
            draw = ImageDraw.Draw(wm)
            w, h = draw.textsize(text, font)
            x, y = (width - w) / 2, (height - h) / 2
            if args.border:
                border_color = random_color()
                border_size = 1
                draw.text((x - border_size, y), text, font=font, fill=border_color)
                draw.text((x + border_size, y), text, font=font, fill=border_color)
                draw.text((x, y - border_size), text, font=font, fill=border_color)
                draw.text((x, y + border_size), text, font=font, fill=border_color)

                # thicker border
                border_size += 2
                draw.text((x - border_size, y - border_size), text, font=font, fill=border_color)
                draw.text((x + border_size, y - border_size), text, font=font, fill=border_color)
                draw.text((x - border_size, y + border_size), text, font=font, fill=border_color)
                draw.text((x + border_size, y + border_size), text, font=font, fill=border_color)
            draw.text((x, y), text, color, font)

            en = ImageEnhance.Brightness(wm)
            mask = en.enhance(1)
            im.paste(wm, (0, 0), mask)
            im.save(filename)


parser = argparse.ArgumentParser(
    description='Generating random fonts png.')
parser.add_argument('-o', '--out-dir', default='out',
                    help='output directory. Default: out')
parser.add_argument('-s', '--size', type=int, nargs='+', default=(30, 35, 1),
                    help='(min, max, step) size of generated fonts. Default: (30, 35, 1)')
parser.add_argument('-t', '--text', type=str, default=None,
                    help='Text to generate. Default: random')
parser.add_argument('-f', '--fonts', type=int, default=5,
                    help='Number of fonts to generate. Default: 5')
parser.add_argument('-sf', '--system-fonts', action='store_true', default=False,
                    help='Use system fonts or fonts from folder. Default: False')
parser.add_argument('-b', '--border', action='store_true', default=False,
                    help='Add border to font. Default: False')

if __name__ == '__main__':
    # nltk.download()
    args = parser.parse_args()
    gen_fonts(args.out_dir, args.size, args.text, args.fonts)
    print(f"Fonts saved in {args.out_dir}.")
