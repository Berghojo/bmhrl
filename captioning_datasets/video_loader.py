from torch.utils.data.dataset import Dataset
class ActivityNetCaptionsVidDataset(Dataset):

    def __init__(self, cfg, phase, get_full_feat):

        '''
            For the doc see the __getitem__.
        '''
        self.cfg = cfg
        self.phase = phase
        self.get_full_feat = get_full_feat

        self.feature_names = f'{cfg.video_feature_name}_{cfg.audio_feature_name}'

        if phase == 'train':
            self.meta_path = cfg.train_meta_path
            self.batch_size = cfg.train_batch_size
        elif phase == 'val_1':
            self.meta_path = cfg.val_1_meta_path
            self.batch_size = cfg.inference_batch_size
        elif phase == 'val_2':
            self.meta_path = cfg.val_2_meta_path
            self.batch_size = cfg.inference_batch_size
        elif phase == 'learned_props':
            self.meta_path = cfg.val_prop_meta_path
            self.batch_size = cfg.inference_batch_size
        else:
            raise NotImplementedError

        # caption dataset *iterator*
        self.train_vocab, self.caption_loader, self.word_counter = caption_iterator(cfg, self.batch_size, self.phase)

        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]

        if cfg.modality == 'video':
            self.features_dataset = I3DFeaturesDataset(
                cfg.video_features_path, cfg.video_feature_name, self.meta_path,
                torch.device(cfg.device), self.pad_idx, self.get_full_feat, cfg
            )
        elif cfg.modality == 'audio':
            self.features_dataset = VGGishFeaturesDataset(
                cfg.audio_features_path, cfg.audio_feature_name, self.meta_path,
                torch.device(cfg.device), self.pad_idx, self.get_full_feat, cfg
            )
        elif cfg.modality == 'audio_video':
            self.features_dataset = AudioVideoFeaturesDataset(
                cfg.video_features_path, cfg.video_feature_name, cfg.audio_features_path,
                cfg.audio_feature_name, self.meta_path, torch.device(cfg.device), self.pad_idx,
                self.get_full_feat, cfg
            )
        else:
            raise Exception(f'it is not implemented for modality: {cfg.modality}')

        # initialize the caption loader iterator
        self.caption_loader_iter = iter(self.caption_loader)

    def __getitem__(self, dataset_index):
        caption_data = next(self.caption_loader_iter)
        to_return = self.features_dataset[caption_data.idx]
        to_return['caption_data'] = caption_data

        return to_return

    def __len__(self):
        return len(self.caption_loader)

    def update_iterator(self):
        '''This should be called after every epoch'''
        self.caption_loader_iter = iter(self.caption_loader)

    def dont_collate(self, batch):
        return batch[0]
