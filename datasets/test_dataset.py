from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=W0611


import cv2, os
from torch.utils.data.dataset import Dataset


class TestDataset(Dataset):
    def __init__(self,
                 img_id_list,
                 transform=None,
                 **_):
        self.transform = transform
        self.images_dir = os.path.join('./data/', 'test_images')
        self.img_id_list = img_id_list
        
        self.size = len(self.img_id_list)

    def to_filepath(self, v):        
            return os.path.join(self.images_dir, v + '.jpg')
    
    def __getitem__(self, index):
        filename = self.to_filepath(self.img_id_list[index])
        print('filename ', filename)
        image = cv2.imread(filename)
        if self.transform is not None:
            image = self.transform(image)
        
       # print('img ', image)
        return {'image': image,'img_id':self.img_id_list[index]
                }

    def __len__(self):
        return self.size




