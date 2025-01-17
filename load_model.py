from My_Model import  ViT, Segformer, AptSegV2
# from My_Model import My_model1, APTATT
import segmentation_models_pytorch as smp
def get_model(config, mode="semantic"):
    if mode == "semantic":
        if config.model == "Unet":
            model = smp.Unet(
                encoder_name=config.PR_Backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config.num_classes,  # model output channels (number of classes in your dataset)
            )
        # elif config.model == "APTATT":
        #     model = APTATT.AdaptATT(
        #         img_size=[config.img_size,config.img_size],
        #         num_classes=config.num_classes,
        #         patch_size=config.patch_size,
        #         stride=config.patch_size,
        #         in_chans=config.channels,
        #         embed_dim=[config.emd_dim],
        #         Transformer_depth=[1],
        #         Mamba_depth=[4]
        #     )
        #
        # elif config.model == "ViM":
        #     model = My_model1.VimWU(
        #         img_size=config.img_size,
        #         num_classes=config.num_classes,
        #         patch_size=config.patch_size,
        #         stride=config.patch_size,
        #         in_chans=config.channels,
        #         embed_dim=config.emd_dim,
        #         depth= 4,
        #     )
        elif config.model == "Segformer":
            model = Segformer.Segformer(
                image_size=[config.img_size,config.img_size],
                patch_size=config.patch_size,
                num_classes = config.num_classes,
                in_chans=config.channels,
                d_model = [config.emd_dim],
                depth=[4],

            )
        elif config.model == "AptSegV2":
            model = AptSegV2.AptSegV2(
                image_size=[config.img_size,config.img_size],
                patch_size=config.patch_size,
                num_classes = config.num_classes,
                channels=config.channels,
                d_model = [config.emd_dim],
                depth=[1],

            )

        elif config.model == "ViTSeg":
            model = ViT.ViTSeg(
                image_size=[config.img_size,config.img_size],
                patch_size=config.patch_size,
                n_cls = config.num_classes,
                channels=config.channels,
                d_model = config.emd_dim,

            )
        elif config.model == "FPN":
            model = smp.FPN(
                encoder_name=config.PR_Backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config.num_classes,  # model output channels (number of classes in your dataset)
            )
        elif config.model == "Linknet":
            model = smp.Linknet(
                encoder_name=config.PR_Backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config.num_classes,  # model output channels (number of classes in your dataset)
            )
        elif config.model == "DeeplabV3+":
            model = smp.DeepLabV3Plus(
                encoder_name=config.PR_Backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config.num_classes,  # model output channels (number of classes in your dataset)
            )




        # elif config.model == "CF":
        #     model = DBL.DBL(
        #         input_dim=10,
        #         num_classes=config.num_classes,
        #         inconv=[32, 64],
        #         sequence=18,
        #         hidden_size=88,
        #         input_shape=(128, 128),
        #         mid_conv=True,
        #         pad_value=config.pad_value,
        #     )
        return model
    else:
        raise NotImplementedError
