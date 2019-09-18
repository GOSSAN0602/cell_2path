

class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        train_controls = pd.read_csv(path_data+'/train_controls.csv')
        test_controls = pd.read_csv(path_data+'/test_controls.csv')
        self.controls = pd.concat([train_controls, test_controls])

    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, experiment, well, plate, channel):
        if self.mode == 'train':
            # pick one of the sites randomly
            site = np.random.randint(1, 3)
        else:
            site = self.site
        return '/'.join([self.img_dir, self.mode, experiment,
                        f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])

    def __getitem__(self, index):
        rec = self.records[index]
        experiment, well, plate = rec.experiment, rec.well, rec.plate
        paths = [self._get_img_path(experiment, well, plate, ch) for ch in self.channels]

        df = self.controls
        negs = df[(df.experiment == experiment) & (df.plate == plate) & (df.sirna == 1138)]
        well = negs.iloc[np.random.randint(0, len(negs))].well
        paths.extend([self._get_img_path(experiment, well, plate, ch) for ch in self.channels])

        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        tr_img = torch.empty((12, img_size, img_size), dtype=torch.float32)

        if self.mode == 'train':
            # randomly crop
            row, col = np.random.randint(0, 512 - img_size + 1, 2)
            tr_img[:6] = img[:6, row:row + img_size, col:col + img_size]
            # randomly crop the negative control image
            row, col = np.random.randint(0, 512 - img_size + 1, 2)
            tr_img[6:] = img[6:, row:row + img_size, col:col + img_size]
            return tr_img, int(self.records[index].sirna)

        # center crop
        row =  col = (512 - img_size) // 2
        tr_img[:] = img[:, row:row + img_size, col:col + img_size]
        return tr_img, rec.id_code

    def __len__(self):
        return self.len
