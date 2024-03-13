import numpy as np
import re
import sys
from PIL import Image

def read_cam_file(filename):
    # TODO
    f = open(filename)
    extrinsics = np.zeros((0, 4))
    intrinsics = np.zeros((0, 3))
    f.readline()
    # Read and store the 4*4 matrix in extrinsics
    for _ in range(4):
        extrinsics = np.vstack([extrinsics, np.fromstring(f.readline(), sep=' ')])
    f.readline()

    f.readline()
    # Read and store the 3*3 matrix in intrinsics
    for _ in range(3):
        intrinsics = np.vstack([intrinsics, np.fromstring(f.readline(), sep=' ')])
    f.readline()

    depth_values = np.fromstring(f.readline(), sep=' ')
    depth_min = depth_values[0]
    depth_max = depth_values[1]
    return intrinsics, extrinsics, depth_min, depth_max

def read_img(filename):
    # TODO
    img = Image.open(filename)
    img_np = np.array(img)
    np_img = img_np / 255.
    img.close()
    return np_img

def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()
