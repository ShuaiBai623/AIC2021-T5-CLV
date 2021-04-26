import os
import cv2
import argparse


def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path


def main(args):
    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        path_data = os.path.join(args.data_root, seq_name)
        path_vdo = os.path.join(path_data, 'vdo.avi')
        path_images = os.path.join(path_data, 'img1')
        check_and_create(path_images)

        vidcap = cv2.VideoCapture(path_vdo)
        success, image = vidcap.read()

        count = 1
        while success:
            path_image = os.path.join(path_images, '%06d.jpg' % count)
            cv2.imwrite(path_image, image)
            success, image = vidcap.read()
            print('Data path: %s; Frame #%06d' % (path_data, count))
            count += 1


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Extract video frames')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)
