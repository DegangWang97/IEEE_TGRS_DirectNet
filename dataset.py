import math
import torch
import torch.utils.data as Data
import numpy as np
import scipy.io as sio
import copy

class Dataset_train(Data.Dataset):
    def __init__(self, opt):
        super(Dataset_train, self).__init__()
        
        self.opt = opt
        
        data_dir = './data/'
        self.image_file = data_dir + opt.dataset + '.mat'
        
        self.input_data = sio.loadmat(self.image_file)
        self.image = self.input_data['data']
        self.image = self.image.astype(np.float32)
        
        self.col = self.image.shape[0]
        self.row = self.image.shape[1]
        self.band = self.image.shape[2]

        self.mirror_image = get_mirror_hsi(self.row, self.col, self.band, self.image, self.opt.win_out)
        
        self.train_point = []
        self.train_data = np.zeros((self.col * self.row, self.opt.win_out, self.opt.win_out, self.band), dtype=float)
        self.size_data = (self.opt.win_out, self.opt.win_out, self.band)
        for i in range(self.row):
            for j in range(self.col):
                self.train_point.append([i,j])
        for k in range(len(self.train_point)):
            self.train_data[k,:,:,:] = get_neighborhood_pixel(self.mirror_image, self.train_point, k, self.opt.win_out)
        self.len = self.train_data.shape[0]
         
        self.label = self.train_data
        self.input, self.mask = generate_mask_plus(self.opt.win_in, self.size_data, copy.deepcopy(self.label))

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        
        label = self.label[index]
        input = self.input[index]
        mask = self.mask[index]
        
        data = {'label': label, 'input': input, 'mask': mask}
        
        data = ToTensor(data)
        
        return data


def DirectNetData(opt):
    
    # train dataloader
    dataset_train = Dataset_train(opt)
    loader_train = Data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    
    data_dir = './data/'
    image_file = data_dir + opt.dataset + '.mat'
    
    input_data = sio.loadmat(image_file)
    image = input_data['data']
    band = image.shape[2]
    
    print("The construction process of training patch pairs with blind-spots is done")
    print('-' * 50)
    
    return loader_train, band


def generate_mask_plus(win_in, size_data, input):
    
    #input row*col, win_out, win_out, band
    #size_data win_out win_out band
    #win_in inner window size
    
    num_sample = int(win_in * win_in)
    mask = np.ones(input.shape)
    output = input

    # point
    out_point = []
    for i_out in range(size_data[0]):
        for j_out in range(size_data[1]):
            out_point.append([i_out,j_out])
            
    in_point = []
    for i_in in range(win_in):
        for j_in in range(win_in):
            in_point.append([i_in + (size_data[0] // 2) - (win_in // 2), j_in + (size_data[1] // 2) - (win_in // 2)])

    bkg_point = []
    for i_bkg in out_point:
        if i_bkg not in in_point:
            bkg_point.append(i_bkg)

    idy_msks = []
    idx_msks = []

    for i_num_sample in range(num_sample):
        idy_msks.append(in_point[i_num_sample][0])
        idx_msks.append(in_point[i_num_sample][1])

    idy_msks = np.array(idy_msks)
    idx_msks = np.array(idx_msks)

    idy_neighs = []
    idx_neighs = []

    num_bkg = len(bkg_point)
    for i_num_bkg in range(num_bkg):
        idy_neighs.append(bkg_point[i_num_bkg][0])
        idx_neighs.append(bkg_point[i_num_bkg][1])

    for num in range(input.shape[0]):
        
        output_squeeze = output[num,:].squeeze() # win_out win_out band
        input_squeeze = input[num,:].squeeze()
        mask_squeeze = mask[num,:].squeeze()
        
        id_neigh = np.random.choice(num_bkg, num_sample, replace=False)
        
        idy_msk_neighs = []
        idx_msk_neighs = []
        for i_id_neigh in range(num_sample):
            idy_msk_neighs.append(idy_neighs[id_neigh[i_id_neigh]])
            idx_msk_neighs.append(idx_neighs[id_neigh[i_id_neigh]])
            
        idy_msk_neighs = np.array(idy_msk_neighs)
        idx_msk_neighs = np.array(idx_msk_neighs)
        
        for ich in range(size_data[2]):
            id_msk = (idy_msks, idx_msks, ich)
            id_msk_neigh = (idy_msk_neighs, idx_msk_neighs, ich)
            
            output_squeeze[id_msk] = input_squeeze[id_msk_neigh]
            mask_squeeze[size_data[0] // 2, size_data[1] // 2, ich] = 0.0
        
        output[num,:] = np.expand_dims(output_squeeze, axis=0)
        mask[num,:] = np.expand_dims(mask_squeeze, axis=0)
        
    return output, mask


def ToTensor(data):
    """Convert ndarrays in sample to Tensors."""
        # Swap color axis because numpy image: N x H x W x C
        #                         torch image: N x C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((0, 3, 1, 2)))
        # return data

    input, label, mask = data['input'], data['label'], data['mask']

    input = input.transpose((2, 0, 1)).astype(np.float32)
    label = label.transpose((2, 0, 1)).astype(np.float32)
    mask = mask.transpose((2, 0, 1)).astype(np.float32)
    
    return {'input': torch.from_numpy(input),
            'label': torch.from_numpy(label),
            'mask': torch.from_numpy(mask)}


def get_mirror_hsi(height, width, band, image, patch):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band), dtype=float)
    #central region
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=image
    #left region
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=image[:,padding-i,:]
    #right region
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=image[:,width-2-i,:]
    #top region
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i,:,:]
    #bottom region
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-2-i,:,:]

    print('-' * 50)
    print("The patch size is : [{0},{1}]".format(patch,patch))
    print("The mirror_data size : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print('-' * 50)
    return mirror_hsi


def get_neighborhood_pixel(mirror_image, train_point, i, patch):
    x = train_point[i][0]
    y = train_point[i][1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image