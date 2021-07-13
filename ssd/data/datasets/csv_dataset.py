import os
from torch.utils.data import Dataset
import numpy as np
import csv
import sys
import skimage.io
import skimage.transform
import skimage.color
import skimage

sys.path.insert(0,os.path.abspath("../../.."))
from PIL import Image
from ssd.structures.container import Container

class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, transform=None,target_transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.class_names = ('__background__',
                   'product')
        self.train_file = train_file
        # self.class_list = class_list
        self.transform = transform
        self.target_transform = target_transform

        # parse the provided class file
        # try:
        #     with self._open_for_csv(self.class_list) as file:
        #         self.classes = self.load_classes(csv.reader(file, delimiter=','))
        # except ValueError as e:
        #     raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))
        self.class_dict = {class_name:i for i,class_name in enumerate(self.class_names)}

        self.labels = {}
        for key, value in self.class_dict.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.class_dict)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())
        # print("Check Image Names")
        # print(self.image_names)
        # print("CHeck is Labels")
        # print(self.labels)

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot,labels = self.load_annotations(idx)
        # print(annot.shape)
        # print(labels.shape)
        # sample = {'img': img, 'annot': annot}

        if self.transform:
            img,annot,labels = self.transform(img,annot,labels)
        if self.target_transform:
            annot,labels = self.target_transform(annot,labels)

        targets = Container(
            boxes=annot,
            labels=labels,
        )



        return img,targets,idx

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 4))
        labels = np.zeros((0,))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 4))
            label = np.zeros((1,))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            label[0]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)
            labels = np.append(labels,label,axis = 0)
            annotations = annotations.astype('float32')
            labels = labels.astype('int64')


        return annotations,labels

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.class_dict[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)

if __name__ == "__main__":
	cdata = CSVDataset("/Users/vaishnavp/Desktop/train.csv")
	print(cdata.__getitem__(7))