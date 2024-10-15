import json
import random
import torch
from torch import nn
from einops import rearrange
import os
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
from fairseq.modules import LayerNorm

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


class MuQ(nn.Module):
    """
    MuQ

    Input: 128-band mel spectrogram
    Frontend: 2-layer Residual convolution
    Backend: 12-layer Conformer
    Quantizer: a codebook for mel spectrogram
    """

    def __init__(
        self,
        num_codebooks=1,
        codebook_dim=16,
        codebook_size=4096,
        features=["melspec_2048"],
        hop_length=240,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=12,
        mask_hop=0.4,
        mask_prob=0.6,
        is_flash=False,
        stat_path=None, #"./data/fma_stats.json",
        model_path=None, #"./data/pretrained_fma.pt",
        w2v2_config_path=None, #"facebook/wav2vec2-conformer-rope-large-960h-ft",
        use_rvq_target=False,
        use_vq_target=False,
        rvq_ckpt_path=None,
        recon_loss_ratio=None,
        label_rate=25,
        use_hubert_masking_strategy=False,
        use_hubert_featurizer=False,
        hubert_conv_feature_layers="[(512,10,5)] + [(512,3,2)] * 3  + [(512,3,3)] + [(512,2,2)] * 2",
        use_hubert_nce_loss=False,
        hubert_final_dim=256,
        rvq_n_codebooks=8,
        rvq_multi_layer_num=1,
        use_encodec_target=False,
    ):
        super(MuQ, self).__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.features = features
        self.recon_loss_ratio = recon_loss_ratio
        self.n_fold = int(100//label_rate)
        self.label_rate = label_rate
        self.use_hubert_masking_strategy = use_hubert_masking_strategy
        self.use_hubert_featurizer = use_hubert_featurizer
        self.use_hubert_nce_loss = use_hubert_nce_loss

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

        # random quantizer
        self.use_rvq_target = use_rvq_target
        self.use_vq_target = use_vq_target
        self.use_encodec_target = use_encodec_target
        
        seed = 142
        if self.use_rvq_like_target:
            if use_rvq_target:
                try:
                    from .rvq_muq import ResidualVectorQuantize
                except:
                    import sys, os
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from rvq_muq import ResidualVectorQuantize
        
                inp_dim = 128*self.n_fold
                self.rvq = ResidualVectorQuantize(
                    input_dim = inp_dim, 
                    n_codebooks = rvq_n_codebooks, 
                    codebook_size = 1024, 
                    codebook_dim = 16, 
                    quantizer_dropout = 0.0,
                    use_multi_layer_num = rvq_multi_layer_num,
                    )
            elif use_vq_target:
                try:
                    from .rvq_muq import VectorQuantize
                except:
                    import sys, os
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from rvq_muq import VectorQuantize
                
                self.rvq = VectorQuantize(
                    input_dim = 128*self.n_fold,
                    codebook_size = 1024,
                    codebook_dim = 8,
                    stale_tolerance = 1000,
                    mfcc_clustering = False
                )
            elif use_encodec_target:
                from encodec import EncodecModel
                self.rvq = EncodecModel.encodec_model_24khz()
                self.rvq.set_target_bandwidth(6.0)
                for param in self.rvq.parameters():
                    param.requires_grad = False
                
            import os
            if rvq_ckpt_path is not None and os.path.exists(rvq_ckpt_path):
                state_dict = torch.load(rvq_ckpt_path, map_location="cpu")
                self.rvq.load_state_dict(state_dict)
            else:
                print(f'Checkpoint for rvq `{rvq_ckpt_path}` not found. Using random initialization.')
        else:
            for feature in self.features:
                for i in range(num_codebooks):
                    setattr(
                        self,
                        f"quantizer_{feature}", # _{i}
                        RandomProjectionQuantizer(
                            n_mels * self.n_fold, codebook_dim, codebook_size, seed=seed + i
                        ),
                    )

        if use_hubert_masking_strategy:
            self.mask_emb = nn.Parameter(
                torch.FloatTensor(encoder_dim).uniform_()
            )

        if use_hubert_featurizer:
            feature_enc_layers = eval(hubert_conv_feature_layers)  # noqa
            hubert_feat_embed = feature_enc_layers[-1][0]
            self.hubert_feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode='default', #cfg.extractor_mode,
                conv_bias=False, #cfg.conv_bias,
            )
            self.post_extract_proj = (
                nn.Linear(hubert_feat_embed, encoder_dim)
                if hubert_feat_embed != encoder_dim
                else None
            )
            self.layer_norm = LayerNorm(hubert_feat_embed)
        else:
            # two residual convolution layers + one projection layer
            strides_factory = {
                4: [2, 2],
                2: [2, 1]
            }
            self.conv = Conv2dSubsampling(
                1, conv_dim, encoder_dim, strides=strides_factory.get(self.n_fold), n_bands=n_mels
            )

        # Conformer
        if is_flash:
            from modules.flash_conformer import (
                Wav2Vec2ConformerEncoder,
                Wav2Vec2ConformerConfig,
            )
        else:
            from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
                Wav2Vec2ConformerEncoder,
                Wav2Vec2ConformerConfig,
            )
        import os
        if w2v2_config_path is None or not os.path.exists(w2v2_config_path):
            w2v2_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w2v2_config.json")
        print("load w2v2 config from:", w2v2_config_path)
        config = Wav2Vec2ConformerConfig.from_pretrained(
            w2v2_config_path
        )
        config.num_hidden_layers = encoder_depth
        config.hidden_size = encoder_dim

        self.conformer = Wav2Vec2ConformerEncoder(config)

        if self.use_hubert_nce_loss:
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(codebook_size, hubert_final_dim)
            ) # embeddings of codes
            nn.init.uniform_(self.label_embs_concat)
            self.linear = nn.Linear(encoder_dim, hubert_final_dim) # final_proj
        else:
            # projection
            self.linear = nn.Linear(encoder_dim, codebook_size) # N_SubSpec=8

        # reconstruct melspec
        if self.recon_loss_ratio is not None and self.recon_loss_ratio > 0:
            self.recon_proj = nn.Linear(encoder_dim, n_mels * self.n_fold)
            self.recon_loss = nn.MSELoss()

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # cls token (used for sequence classification)
        random.seed(seed)
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))

        # load model
        if model_path:
            S = torch.load(model_path)["state_dict"]
            SS = {k[6:]: v for k, v in S.items()}
            SS['quantizer_melspec_2048.random_projection'] = SS['quantizer_melspec_2048_0.random_projection']
            SS['quantizer_melspec_2048.codebook'] = SS['quantizer_melspec_2048_0.codebook']
            del SS['quantizer_melspec_2048_0.random_projection']
            del SS['quantizer_melspec_2048_0.codebook']
            unmatch = self.load_state_dict(SS, strict=False)
            if len(unmatch.missing_keys) > 0:
                print(f'Missing keys: {unmatch.missing_keys}')

    @property
    def use_rvq_like_target(self):
        return self.use_rvq_target or self.use_vq_target or self.use_encodec_target


    def apply_hubert_mask(self, x, padding_mask=None, target_list=None):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_length = int(self.mask_hop / (1/self.label_rate))
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                mask_length, # self.mask_length,
                "static", #self.mask_selection,
                0, #self.mask_other,
                min_masks=2,
                no_overlap=False, #self.no_mask_overlap,
                min_space=1, #self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
            mask_indices = torch.nonzero(mask_indices)
        else:
            mask_indices = None

        return x, mask_indices

    def masking(self, x, attention_mask=None):
        """random masking of 400ms with given probability"""
        if self.use_hubert_masking_strategy:
            return x, None
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop) # 9600 = 24000 * 0.4
        len_masking_token = int(24000 / self.hop_length / 2 / 2 * self.mask_hop) # 10 = 25Hz * 0.4

        # get random mask indices
        start_indices = torch.rand(b, t // len_masking_raw) < self.mask_prob
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_raw, dim=1) 
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)

        return mx, token_domain_masked_indices


    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            layer.to(x.device)
            dtype = x.dtype
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
            if out[key].dtype != dtype:
                out[key].to(dtype=dtype)
        return out

    def encoder(self, x, *, attention_mask=None, is_features_only=False):
        """2-layer conv + w2v-conformer"""
        if not self.use_hubert_featurizer:
            x = self.conv(x) # [3, 128, 3000] -> [3, 750, 1024]
        if self.training and self.use_hubert_masking_strategy and not is_features_only:
            x, mask_indices = self.apply_hubert_mask(x)
        else:
            mask_indices = None
        if attention_mask is None:
            out = self.conformer(x, output_hidden_states=True)
        else:
            attention_mask = attention_mask.bool()
            skip_n = int(attention_mask.size(-1) / x.size(1))
            attention_mask = attention_mask[:, ::skip_n]
            attention_mask = attention_mask[:, :x.size(1)]
            out = self.conformer(x, attention_mask=attention_mask, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        interval = self.codebook_size
        logits = {
            key: logits[:, :, i * interval : (i + 1) * interval]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb, mask_indices

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
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=self.n_fold)
        return x

    def get_rvq_codes(self, inp, raw_wav):
        if self.use_rvq_target:
            quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = self.rvq(inp)
            return codes
        if self.use_vq_target:
            quantized_prompt_embeds, commitment_loss, codebook_loss, codes, _ = self.rvq(inp)
            return codes.unsqueeze(1)
        if self.use_encodec_target:
            encoded_frames = self.rvq.encode(raw_wav.unsqueeze(1)) #list, B,[ 8,T ]
            codes = torch.cat([encoded[0].detach() for encoded in encoded_frames], dim=-1)
            if self.label_rate == 25:
                codes = codes[:, :, ::3]
            return codes

    @torch.no_grad()
    def tokenize(self, x, raw_wav):
        out = {}
        for key in x.keys():
            if self.use_rvq_like_target:
                self.rvq.eval()
                inp = x[key].permute((0, 2, 1))
                codes = self.get_rvq_codes(inp, raw_wav)
                out[key] = torch.cat([codes[:, idx, ...] for idx in range(int(self.codebook_size//1024))], dim=-1) # (when use freq mask)->[Batch, N_SubSpec, SeqLen=8*750]
            else:
                layer = getattr(self, "quantizer_%s" % key)
                out[key] = layer(x[key])
        return out

    def to_spec_wise_quad(self, x):
        Batch, QuadSpec, Time = x.shape
        SubSpec, N_SubSpec = 16, 8
        assert 4 * SubSpec * N_SubSpec == QuadSpec == 4*128
        x = rearrange(x, "b (q n s) t -> b (q s) (n t)", q=4, n=N_SubSpec, s=SubSpec)
        return x # [Batch, SubSpec=16, N_SubSpec*Time=8*100Hz]

    def get_targets(self, x, label=None):
        if self.use_encodec_target:
            raw_x = x.clone()
        else:
            raw_x = None
        x = self.preprocessing(x, features=self.features) # -> {'melspec_2048': Tensor{Size([3, 128, 3000]) cuda:0 f32}}
        x = self.normalize(x)
        x = self.rearrange(x) # -> {'melspec_2048': Tensor{Size([3, 750, 512]) cuda:0 f32}}
        melspec = x['melspec_2048']
        if label is None:
            target_tokens = self.tokenize(x, raw_x) # -> {'melspec_2048': Tensor{Size([3, 750]) cuda:0 i64}}
        else:
            # print("use_target from label")
            target_tokens = {'melspec_2048': rearrange(label, "b n s -> b (n s)").long()}
        return target_tokens, melspec

    def get_predictions(self, x, *, mask=None, attention_mask=None, return_new_mask=False, is_features_only=False):
        # preprocessing
        if not self.use_hubert_featurizer:
            x = self.preprocessing(x, features=["melspec_2048"])
            x = self.normalize(x) # -> {'melspec_2048': Tensor{Size([3, 128, 3000]) cuda:0 f32}}
        else:
            features = self.hubert_feature_extractor(x)
            features = self.layer_norm(features.transpose(1, 2))
            if self.post_extract_proj is not None:
                features = self.post_extract_proj(features)
            x = {"melspec_2048": features}

        # encoding
        logits, hidden_emb, new_mask = self.encoder(x["melspec_2048"], attention_mask=attention_mask, is_features_only=is_features_only)

        if return_new_mask:
            return logits, hidden_emb, mask if new_mask is None else new_mask
        else:
            return logits, hidden_emb

    def get_latent(self, x, layer_ix=12):
        _, hidden_states = self.get_predictions(x)
        emb = hidden_states[layer_ix]
        return emb

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= 0.1
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def compute_hubert_nce_loss(self, proj_xs, targets):

        label_embs_list = self.label_embs_concat.split(self.codebook_size, 0) # (self.num_classes, 0)
        
        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            return self.compute_nce(proj_x, y, negs)
    
        logit_list = [
                    compute_pred(proj_x, t, label_embs_list[i])
                    for i, (proj_x, t) in enumerate(zip(proj_xs, targets))
                ]
        
        return sum(logit_list)

    
    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            if not self.use_rvq_like_target:
                masked_logits = logits[key][tuple(masked_indices.t())]
                masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            else:
                Batch, SeqLen, N_Codebook_x_CodebookSize = logits[key].shape # CodebookSize=4096
                Batch, N_Codebook_x_SeqLen = target_tokens[key].shape # N_Codebook*SeqLen=4*750
                N_Codebook = int(N_Codebook_x_SeqLen // SeqLen)
                # print("not use_virtual, n codebook = ", N_Codebook)
                target_tokens[key] = rearrange(target_tokens[key], "b (n s) -> b s n", n=N_Codebook) # Batch, SeqLen=750, N_Codebook=4
                masked_logits = logits[key][tuple(masked_indices.t())] 
                masked_tokens = target_tokens[key][tuple(masked_indices.t())] 
                masked_logits = rearrange(masked_logits, "b (n c) -> (b n) c", n=N_Codebook)
                masked_tokens = rearrange(masked_tokens, "b n -> (b n)", n=N_Codebook) 

            if self.use_hubert_nce_loss:
                losses[key] = self.compute_hubert_nce_loss(masked_logits, masked_tokens)
            else:
                losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def get_recon_loss(self, last_hidden_emb, melspec, masked_indices):
        pred_melspec = self.recon_proj(last_hidden_emb[tuple(masked_indices.t())])
        target_melspec = melspec[tuple(masked_indices.t())]
        recon_loss = self.recon_loss(pred_melspec, target_melspec)
        return recon_loss

    def forward(self, x, attention_mask=None, label=None):
        dtype = x.dtype
        # get target feature tokens
        target_tokens, melspec = self.get_targets(x, label=label) 

        # masking
        x, masked_indices = self.masking(x, attention_mask=attention_mask) 

        # forward
        logits, hidden_emb, masked_indices = self.get_predictions(x, mask=masked_indices, attention_mask=attention_mask, return_new_mask=True) 

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)

        if self.recon_loss_ratio:
            losses["recon_loss"] = self.get_recon_loss(hidden_emb[-1], melspec, masked_indices) * self.recon_loss_ratio

        return logits, hidden_emb, losses, accuracies
