import csv
import glob
import os
import os.path


def get_train_test_lists(version='01'):
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    test_file = './ucfTrainTestlist/testlist' + version + '.txt'
    train_file = './ucfTrainTestlist/trainlist' + version + '.txt'

    # Build the test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def extract_files(file_groups):
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

    for group, videos in file_groups.items():

        for video in videos:

            # Get the parts of the file.
            video_parts = get_video_parts(video)
            train_or_test = group

            classname, filename_no_ext, filename = video_parts

            # Now get how many frames it is.
            nb_frames = get_nb_frames_for_video(video_parts)

            data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

            # if nb_frames < 40:
            print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    classname, filename_no_ext, _ = video_parts

    path = '/data/yingwei/vidDB/UCF101/rgb/'
    filename = filename_no_ext
    filename = os.path.splitext(filename)[0]
    path = path + filename + '/'

    generated_files = glob.glob(path + '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[1]
    filename_no_ext = filename.split('.')[0]
    classname = parts[0]

    return classname, filename_no_ext, filename

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    group_lists = get_train_test_lists()
    extract_files(group_lists)

if __name__ == '__main__':
    main()

