try:
    from .rvq import *
except:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rvq import *

try:
    from ..modules.random_quantizer import RandomProjectionQuantizer
    from ..modules.features import MelSTFT
    from ..modules.conv import Conv2dSubsampling
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modules.random_quantizer import RandomProjectionQuantizer
    from modules.features import MelSTFT
    from modules.conv import Conv2dSubsampling

import fairseq

CLIPSECS = 30 # 5 for rvq, 30 for model

class RVQDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        normalize: bool = False,
    ):
        self.sample_rate = sample_rate
        self.datas,inds,tot,self.sizes = load_audio_by_json(manifest_path, None, None, self.sample_rate)
        self.dataset_len = len(self.datas)

        self.reader = Read_and_PadCrop_Normalized_T(n_samples=CLIPSECS*sample_rate,sample_rate = self.sample_rate)
        self.normalize = normalize
    

    def __getitem__(self, i):
        # WORLD_SIZE = int(torch.distributed.get_world_size())
        # WORLD_RANK = int(torch.distributed.get_rank())
        # np.random.seed(1337 + self.epoch * WORLD_SIZE + WORLD_RANK + i)
        # index = random.randint(0,len(self.sizes) - 1)
        index = i
        item = None
        while item is None:
            try:
                wav = self.get_audio_by_slice(index)
                item = {"id": index, "source": wav}
            except Exception as e:
                # print(e)
                traceback.print_exc()
                print(f'skip damaged data {index}')
                index = np.random.randint(0,len(self.sizes)-1)
        return item

    def __len__(self):
        return self.dataset_len
    
    def get_audio_by_slice(self,index):
        
        # wav_path = os.path.join('/apdcephfs/share_1316500/cloudezhou/MERT/MERT/converted', self.audio_names[index])
        wav_path = self.datas[index]['path']
        # wav_path = '/apdcephfs_cq7/share_1297902/speech_data/' + wav_path[wav_path.index('Music4All'):]
        # print(wav_path)
        audio_info =  torchaudio.info(wav_path)
        origin_sample_rate = audio_info.sample_rate
        origin_duration = audio_info.num_frames / origin_sample_rate

        wav, *ignored = self.reader(wav_path, origin_duration,origin_sample_rate)
        wav = wav.float()
        
        # _path, slice_ptr = parse_path(wav_path)
        # original way
        # if len(slice_ptr) == 0:
        #     wav, cur_sample_rate = sf.read(_path)
        # else:
        #     assert _path.endswith(".zip")
        #     data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        #     f = io.BytesIO(data)
        #     wav, cur_sample_rate = sf.read(f)
        # wav = torch.from_numpy(wav).float()
        # print(wav.shape)
        wav = wav.permute(1,0)
        wav = self.postprocess(wav, self.sample_rate)
        # print(wav.shape)

        # wav = wav.squeeze(0)
        return wav
    
    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

class Preprocessor(nn.Module):
    def __init__(self, 
            codebook_dim=16,
            codebook_size=4096,
            hop_length=240,
            n_mels=128,
            stat_path=None,
            is_spec_wise=False,
            s=4,
        ) -> None:
        super().__init__()

        self.features=["melspec_2048"]
        self.s = s

        # load feature mean / std stats
        import os
        if stat_path is not None and os.path.exists(stat_path):
            with open(stat_path, "r") as f:
                self.stat = json.load(f)
        else:
            # print("No stats file found at `{}`, use default from msd.".format(stat_path))
            self.stat = {"spec_256_cnt": 14394344256, "spec_256_mean": -23.34296658431829, "spec_256_std": 26.189295587132637, "spec_512_cnt": 28677104448, "spec_512_mean": -21.31267396860235, "spec_512_std": 26.52644536245769, "spec_1024_cnt": 57242624832, "spec_1024_mean": -18.852271129208273, "spec_1024_std": 26.443154583585663, "spec_2048_cnt": 114373665600, "spec_2048_mean": -15.638743433896792, "spec_2048_std": 26.115825961611545, "spec_4096_cnt": 228635747136, "spec_4096_mean": -11.715532502794836, "spec_4096_std": 25.763972210234062, "melspec_256_cnt": 14282760192, "melspec_256_mean": -26.962600400166156, "melspec_256_std": 36.13614100912126, "melspec_512_cnt": 14282760192, "melspec_512_mean": -9.108344167718862, "melspec_512_std": 24.71910937988429, "melspec_1024_cnt": 14282760192, "melspec_1024_mean": 0.37302579246531126, "melspec_1024_std": 18.684082325919388, "melspec_2048_cnt": 14282760192, "melspec_2048_mean": 6.768444971712967, "melspec_2048_std": 18.417922652295623, "melspec_4096_cnt": 14282760192, "melspec_4096_mean": 13.617164614990036, "melspec_4096_std": 18.08552130124525, "cqt_cnt": 9373061376, "cqt_mean": 0.46341379757927165, "cqt_std": 0.9543998080910191, "mfcc_256_cnt": 1339008768, "mfcc_256_mean": -11.681755459447485, "mfcc_256_std": 29.183186444668316, "mfcc_512_cnt": 1339008768, "mfcc_512_mean": -2.540581461792183, "mfcc_512_std": 31.93752185832081, "mfcc_1024_cnt": 1339008768, "mfcc_1024_mean": 6.606636263169779, "mfcc_1024_std": 34.151644801729624, "mfcc_2048_cnt": 1339008768, "mfcc_2048_mean": 5.281600844245184, "mfcc_2048_std": 33.12784541220003, "mfcc_4096_cnt": 1339008768, "mfcc_4096_mean": 4.7616569480166095, "mfcc_4096_std": 32.61458906894133, "chromagram_256_cnt": 1339008768, "chromagram_256_mean": 55.15596556703181, "chromagram_256_std": 73.91858278719991, "chromagram_512_cnt": 1339008768, "chromagram_512_mean": 175.73092252759895, "chromagram_512_std": 248.48485148525953, "chromagram_1024_cnt": 1339008768, "chromagram_1024_mean": 589.2947481634608, "chromagram_1024_std": 913.857929063196, "chromagram_2048_cnt": 1339008768, "chromagram_2048_mean": 2062.286388327397, "chromagram_2048_std": 3458.92657915397, "chromagram_4096_cnt": 1339008768, "chromagram_4096_mean": 7673.039107997085, "chromagram_4096_std": 13009.883158267234}

        # feature extractor
        self.preprocessor_melspec_2048 = MelSTFT(
            n_fft=2048, hop_length=hop_length, is_db=True
        )

        self.is_spec_wise = is_spec_wise
        

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key] # {'melspec_2048_cnt': 14282760192, 'melspec_2048_mean': 6.768444971712967}
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """rearrange the batch to flatten every 4 steps"""
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=self.s)
        return x
    
    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        for key in x.keys():
            layer = getattr(self, "quantizer_%s" % key)
            out[key] = layer(x[key])
        return out

    def to_spec_wise(self, x):
        Batch, Spec, Time = x.shape
        SubSpec, N_SubSpec = 16, 8
        assert SubSpec * N_SubSpec == Spec == 128
        x = rearrange(x, "b (n s) t -> b s (n t)", n=N_SubSpec, s=SubSpec)
        return x # [Batch, SubSpec=16, N_SubSpec*Time=8*100Hz]

    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x, features=self.features) # -> {'melspec_2048': Tensor{Size([3, 128, 3000]) cuda:0 f32}}
        x = self.normalize(x)
        if self.is_spec_wise:
            x = {k:self.to_spec_wise(v) for k,v in x.items()}
        x = self.rearrange(x) # -> {'melspec_2048': Tensor{Size([3, 750, 512]) cuda:0 f32}}
        return x['melspec_2048'].permute((0, 2, 1))


class CQTPreprocessor(nn.Module):
    def __init__(self, 
            sr=24000,
            hop=960,
            nb=84,
            to_db = True,
        ) -> None:
        super().__init__()

        from nnAudio.features.cqt import CQT
        import torchaudio
        self.cqt_fn = CQT(
                            sr=sr, 
                            hop_length=hop,
                            n_bins=nb, 
                            fmin=32.7 if nb == 84 else 27.5, # 84 or 88
                            bins_per_octave=12, 
                            filter_scale=1, 
                            norm=1, 
                            window='hann', 
                            center=True, 
                            pad_mode='constant', 
                            trainable=False, 
                            output_format='Magnitude', 
                            verbose=True,
                        )
        if to_db:
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        else:
            self.amplitude_to_db = lambda x:x

    @torch.no_grad()
    def __call__(self, x):
        return self.amplitude_to_db(self.cqt_fn(x))


from dataclasses import dataclass

@dataclass
class UserDirModule:
    user_dir: str

def load_model(model_dir, checkpoint_dir):
    '''Load Fairseq SSL model'''

    if model_dir is not None:
        model_path = UserDirModule(model_dir)
        fairseq.utils.import_user_module(model_path)
    
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir], strict=False)
    model = model[0]

    return model



class PreprocessorWithModel(nn.Module):
    def __init__(self, model_dir, checkpoint_dir, use_layer_idx=9) -> None:
        super().__init__()
        self.model = load_model(model_dir=model_dir, checkpoint_dir=checkpoint_dir)
        self.model.eval()
        self.use_layer_idx = use_layer_idx

    def forward(self, x):
        with torch.no_grad():
            self.model.eval()
            res = self.model(x, features_only = True)
            layer_results = res['layer_results']
            return layer_results[self.use_layer_idx].permute(0,2,1)


def Music_ModelTarget_Config():
    config = dict(
        train_dataset = dict(
            manifest_path = 'path/to/data/music4all/train.json',
            sample_rate = 24000,
            normalize = False,
        ),
        valid_dataset = dict(
            manifest_path = 'path/to/data/music4all/valid.json',
            manifest_path = None,
            sample_rate = 24000,
            normalize = False,
        ),
        model = dict(
            input_dim = 1024, 
            n_codebooks = 8, 
            codebook_size = 1024, 
            codebook_dim = 16, 
            quantizer_dropout = 0.0,
        ),
        train = dict(
            batch_size = 32,
            num_workers = 6,
            valid_interval = 10,
            save_interval = 100,
            max_updates = 500000,
            lr = 1e-4,
            device = 'cuda:0',
            loss = 'commitment_loss * 0.25 + codebook_loss * 1.0 + (x - quantized_prompt_embeds).abs().mean()',
            preprocess = PreprocessorWithModel(
                model_dir= 'path/to/muq/muq_fairseq',
                checkpoint_dir='path/to/muq_m4a_75K.pt',
                use_layer_idx=9,
            )
        )
    )
    return config



def main(config):
    train_dataset = RVQDataset(**config['train_dataset'])
    if config['valid_dataset']['manifest_path'] is None:
        # split train and valid dataset
        from torch.utils.data import random_split
        train_dataset, valid_dataset = random_split(
            train_dataset, lengths=[len(train_dataset) - 500, 500]
        )
    else:
        valid_dataset = RVQDataset(**config['valid_dataset'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['train']['batch_size'], drop_last=True, num_workers=config['train']['num_workers'])
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=config['train']['batch_size'], drop_last=True, num_workers=config['train']['num_workers'])
    model = ResidualVectorQuantize(**config['model'])

    device = config['train']['device']
    preprocess = config['train']['preprocess'].to(device)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    cur_updates = 0
    is_running = True
    result = {}
    from tqdm import tqdm
    from tensorboardX import SummaryWriter 
    writer = SummaryWriter()
    from collections import defaultdict
    import os
    from logging import getLogger
    logger = getLogger()
            
    while is_running:
        results = defaultdict(lambda:0)
        for item in tqdm(train_dataloader, desc='train'): 
            wavs = item['source']
            optimizer.zero_grad()
            wavs = wavs.to(device)
            x = preprocess(wavs)
            model.train()
            quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = model(x)
            loss = eval(config['train']['loss'])
            loss.backward()
            optimizer.step()

            results['loss/train'] += loss.item()
            results['commitment_loss/train'] += commitment_loss.item()
            results['codebook_loss/train'] += codebook_loss.item()
            results['rvq_usage/train'] += rvq_usage.float().mean().item()

            if cur_updates % config['train']['valid_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    for item in tqdm(valid_dataloader, desc='valid'): 
                        wavs = item['source']
                        wavs = wavs.to(device)
                        x = preprocess(wavs)
                        quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = model(x)
                        valid_loss = eval(config['train']['loss'])
                        
                        results['loss/valid'] += valid_loss.item()
                        results['commitment_loss/valid'] += commitment_loss.item()
                        results['codebook_loss/valid'] += codebook_loss.item()
                        results['rvq_usage/valid'] += rvq_usage.float().mean().item()

                    results['cur_updates'] = cur_updates
                    results['loss/train'] /= config['train']['valid_interval'] 
                    results['commitment_loss/train'] /= config['train']['valid_interval']
                    results['codebook_loss/train'] /= config['train']['valid_interval']
                    results['rvq_usage/train'] /= config['train']['valid_interval']

                    results['loss/valid'] /= len(valid_dataloader) 
                    results['commitment_loss/valid'] /= len(valid_dataloader)
                    results['codebook_loss/valid'] /= len(valid_dataloader)
                    results['rvq_usage/valid'] /= len(valid_dataloader)

                    print('')
                    logger.info(str(results))
                    for k,v in results.items():
                        writer.add_scalar(k, v, cur_updates)
                    
                    results.clear()

            if cur_updates % config['train']['save_interval'] == 0:
                os.makedirs(f'{writer.logdir}/ckpt/', exist_ok=True)
                logger.info(f'saving checkpoint to {writer.logdir}/ckpt/RVQ_{cur_updates}.pth')
                torch.save(model.state_dict(), f'{writer.logdir}/ckpt/RVQ_{cur_updates}.pth')

            
            if cur_updates < config['train']['max_updates']:
                cur_updates += 1
            else:
                is_running = False
                break
            


if __name__ == '__main__':
    config = Music_ModelTarget_Config()
    main(config)