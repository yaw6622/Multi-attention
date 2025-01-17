import torch
import torch.nn as nn
import torch.nn.functional as F
from My_Model.utils.ViTmodules import VisionTransformer, DecoderLinear, padding, unpadding
class ViTSeg(nn.Module):
    def __init__(self, image_size=[224,224], patch_size = 16, n_layers=4, d_model=96, mlp_ratio=4, n_heads=16, n_cls=9,
                 dropout=0.1, drop_path_rate=0.0, distilled=False, channels=3, fill_value=0):
        super(ViTSeg, self).__init__()

        # 参数存储
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = mlp_ratio * d_model
        self.n_heads = n_heads
        self.n_cls = n_cls
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.distilled = distilled
        self.channels = channels
        self.fill_value = fill_value

        # Encoder部分
        self.encoder = self.build_encoder()

        # Decoder部分
        self.decoder = self.build_decoder()
        self.segmenter = Segmenter(
            self.encoder, self.decoder, n_cls, patch_size = self.patch_size, fill_value=self.fill_value,distilled=self.distilled
        )

        # 输出层


    def build_encoder(self):
        model = VisionTransformer(
            image_size=self.image_size,
            patch_size=self.patch_size,
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            n_cls=self.n_cls,
            dropout=self.dropout,
            drop_path_rate=self.drop_path_rate,
            distilled=self.distilled,
            channels=self.channels
        )
        return model

    def build_decoder(self):
        # 在这里定义Decoder部分
        # 例如，使用Transformer Decoder结构
        decoder = DecoderLinear(
            n_cls = self.n_cls,
            patch_size = self.patch_size,
            d_encoder = self.d_model,
        )
        return decoder



    def forward(self, x):
        # 定义前向传播逻辑
        # 处理输入数据
        # 通过encoder和decoder
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        # output = self.output_layer(decoded)
        # return output
        return self.segmenter(x)

class Segmenter(nn.Module):
    def __init__(self, encoder, decoder, n_cls, patch_size,fill_value, distilled):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.fill_value = fill_value
        self.distilled = distilled

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params


    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size,self.fill_value)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
