import csv
import glob
import os
import os.path

def extract_files():
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    dir = '/data3/moshan/cogs260_class_project/video_classification/data/moving_mnist/'
    folders = ['train', 'test']

    for folder in folders:
        path = dir + folder + '/'
        class_folders = glob.glob(path + '*')

        for vid_class in class_folders:

            # Get the parts of the file.
            video_parts = get_video_parts(vid_class)
            train_or_test = folder

            classname, filename_no_ext, filename = video_parts

            # Now get how many frames it is.
            nb_frames = get_nb_frames_for_video(video_parts, train_or_test)

            data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

            # if nb_frames < 40:
            print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('mnist_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts, train_or_test):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    classname, filename_no_ext, _ = video_parts

    path = '/data3/moshan/cogs260_class_project/video_classification/data/moving_mnist/'
    filename = filename_no_ext
    path = path + train_or_test + '/' + filename + '/'

    generated_files = glob.glob(path + '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[-1]
    filename_no_ext = filename
    classname = parts[-1]

    return classname, filename_no_ext, filename

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()

