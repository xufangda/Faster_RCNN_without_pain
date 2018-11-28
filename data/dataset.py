from data.voc_dataset import VOCBboxDataset
voc_data_dir = 'D:\\Users\\XFD\\AnacondaProjects\\Dataset_VOC2012\\'


class Dataset:
    def __init__(self):
        self.db = VOCBboxDataset(voc_data_dir)

    def __getitem__(self, idx):
        img, bbox, label, difficult = self.db.get_example(idx)

        scale = 1.0
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

