import torch
import torch.nn as nn
import modeling.ECGEncoder_co as ECGEncoder
from modeling.swin_transformer import build_swin
from modeling.greenMIM_CL_models import ClipLoss
from modeling.Swin_Config import build_swin_config
from util.extract_backbone import load_pretrained_
class ECCL_single(nn.Module):
    def __init__(self,args=None) -> None:
        super().__init__()
        self.device = args.device
        self.args = args
        
        ######################  ECG Model Set   ###############################
        self.ECG_encoder = ECGEncoder.__dict__[args.ecg_model](
            img_size=args.ecg_input_size,
            patch_size=args.ecg_patch_size,
            in_chans=args.ecg_input_channels,
            num_classes=args.latent_dim,
            drop_rate=args.ecg_drop_out,
            args=args,
        )
        if args.ecg_pretrained:
            print("load pretrained ecg_model")
            ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
            ecg_checkpoint_model = ecg_checkpoint['model']
            msg = self.ECG_encoder.load_state_dict(ecg_checkpoint_model, strict=False)
            print('load ecg model')
            print(msg)
        

        ######################  CMR Model Set   ###############################
        config = build_swin_config(args.cmr_model_config)
        self.CMR_encoder1 = build_swin(config)
        if args.cmr_pretrained_model:
            print("load pretrained swin cmr_model")
            cmr_checkpoint = torch.load(args.cmr_pretrained_model, map_location='cpu')
            cmr_checkpoint_model = cmr_checkpoint['model']
            if args.cmr_ind == True:
                load_pretrained_(ckpt_model=cmr_checkpoint_model, model=self.CMR_encoder1)
            else:
                if self.args.CL_which_cmr == 'cmr':
                    cmr_checkpoint_model1 = {k: v for k, v in cmr_checkpoint_model.items() if k.startswith('mask_model1')}
                    cmr_checkpoint_model1 = {k.replace('mask_model1.', ''): v for k, v in cmr_checkpoint_model1.items()}
                elif self.args.CL_which_cmr == 'cmr_la':
                    cmr_checkpoint_model1 = {k: v for k, v in cmr_checkpoint_model.items() if k.startswith('mask_model2')}
                    cmr_checkpoint_model1 = {k.replace('mask_model2.', ''): v for k, v in cmr_checkpoint_model1.items()}
                load_pretrained_(ckpt_model=cmr_checkpoint_model1, model=self.CMR_encoder1)
            
            
            
        ######################  Loss Set   ###############################
        self.loss_fn = ClipLoss(temperature=args.temperature, args=args)
        
    

    def forward(self, ecg, cmr, cmr_la, cond):

        
        ecg_inter,ecg_feature = self.ECG_encoder(ecg,cond)
        if self.args.CL_which_cmr == 'cmr':
            with torch.no_grad():
                cmr_feature = self.CMR_encoder1(cmr)
            loss_ecgcmr = self.loss_fn(ecg_feature, cmr_feature)
            loss = {
                "loss_ecgcmr":loss_ecgcmr
            }
        elif self.args.CL_which_cmr == 'cmr_la':
            with torch.no_grad():
                cmr_feature = self.CMR_encoder1(cmr_la)
            loss_ecgcmr = self.loss_fn(ecg_feature, cmr_feature)
            loss = {
                "loss_ecglacmr":loss_ecgcmr
            }
        else:
            print('Error: CL_which_cmr should be cmr or cmr_la')
            exit()
        return loss
