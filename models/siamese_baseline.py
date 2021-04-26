import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet34
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from models.senet import se_resnext50_32x4d
from .efficientnet import EfficientNet


supported_img_encoders = ["se_resnext50_32x4d","efficientnet-b2","efficientnet-b3"]

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        # self.init_weight()
    def init_weight(self):
        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.q_proj.weight, std=0)
        nn.init.normal_(self.k_proj.weight, std=std)
        nn.init.normal_(self.v_proj.weight, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class SiameseBaselineModelv1(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = resnet50(pretrained=False,
                                 num_classes=model_cfg.OUTPUT_SIZE)
        state_dict = torch.load(self.model_cfg.RESNET_CHECKPOINT,
                                map_location=lambda storage, loc: storage.cpu())
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        self.backbone.load_state_dict(state_dict, strict=False)

        self.bert_model = BertModel.from_pretrained(model_cfg.BERT_NAME)
        for p in  self.bert_model.parameters():
            p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        lang_dim = 768
        img_dim = 1024
        self.domian_fc1 = nn.Sequential(nn.Linear(lang_dim, lang_dim), nn.ReLU(), nn.Linear(lang_dim, 1024))
        self.domian_fc2 = nn.Sequential(nn.ReLU(),nn.Linear(img_dim, img_dim))
        self.domian_cls = nn.Sequential(nn.Linear(img_dim, img_dim), nn.ReLU(),nn.Linear(img_dim, 2498))

    def encode_text(self,nl_input_ids,nl_attention_mask):
        outputs = self.bert_model(nl_input_ids,
                                  attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_fc1(lang_embeds)
        lang_embeds = F.normalize(lang_embeds, p = 2, dim = -1)
        return lang_embeds

    def encode_images(self,crops):
        visual_embeds = self.backbone(crops)
        visual_embeds = self.domian_fc2(visual_embeds)
        visual_embeds = F.normalize(visual_embeds, p = 2, dim = -1)
        return visual_embeds

    def forward(self, nl_input_ids,nl_attention_mask,crops):

        outputs = self.bert_model(nl_input_ids,
                                  attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_fc1(lang_embeds)
        visual_embeds = self.backbone(crops)
       
        cls_logits = self.domian_cls(visual_embeds)
        visual_embeds = self.domian_fc2(visual_embeds)
        visual_embeds, lang_embeds = map(lambda t: F.normalize(t, p = 2, dim = -1), (visual_embeds, lang_embeds))
        return [(visual_embeds, lang_embeds)], self.logit_scale,[cls_logits]



class SiameseLocalandMotionModelBIG(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        embed_dim = self.model_cfg.EMBED_DIM
        if self.model_cfg.IMG_ENCODER in  supported_img_encoders:
            if self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.vis_backbone = se_resnext50_32x4d()
                self.vis_backbone_bk = se_resnext50_32x4d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
                self.domian_vis_fc_bk = nn.Conv2d(self.img_in_dim, embed_dim,kernel_size=1)
            else:
                self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_bk = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)

        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        for p in  self.bert_model.parameters():
            p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        
        self.domian_vis_fc_merge = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim),nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.vis_car_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.vis_motion_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))
        self.lang_motion_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.ReLU(),nn.Linear(embed_dim, embed_dim//2))

        self.domian_lang_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        if self.model_cfg.car_idloss:
            self.id_cls = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.mo_idloss:   
            self.id_cls2 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))
        if self.model_cfg.share_idloss:  
            self.id_cls3 = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.BatchNorm1d(embed_dim), nn.ReLU(),nn.Linear(embed_dim, self.model_cfg.NUM_CLASS))

    def encode_text(self,nl_input_ids,nl_attention_mask):
        outputs = self.bert_model(nl_input_ids,
                                  attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        lang_embeds = F.normalize(lang_embeds, p = 2, dim = -1)
        return lang_embeds

    def encode_images(self,crops,motion):
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)
        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds,visual_mo_embeds],dim=-1))
        visual_embeds = F.normalize(visual_merge_embeds, p = 2, dim = -1)
        return visual_embeds

    def forward(self, nl_input_ids,nl_attention_mask,crops,motion):

        outputs = self.bert_model(nl_input_ids,attention_mask=nl_attention_mask)
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.domian_lang_fc(lang_embeds)
        visual_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_embeds = visual_embeds.view(visual_embeds.size(0), -1)
        motion_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        motion_embeds = motion_embeds.view(motion_embeds.size(0), -1)        
        visual_car_embeds = self.vis_car_fc(visual_embeds)
        visual_mo_embeds = self.vis_motion_fc(motion_embeds)
        visual_merge_embeds = self.domian_vis_fc_merge(torch.cat([visual_car_embeds,visual_mo_embeds],dim=-1))
        cls_logits_results = []
        if self.model_cfg.car_idloss:
            cls_logits = self.id_cls(visual_embeds)
            cls_logits_results.append(cls_logits)
        if self.model_cfg.mo_idloss:
            cls_logits2 = self.id_cls2(motion_embeds)
            cls_logits_results.append(cls_logits2)
        lang_car_embeds = self.lang_car_fc(lang_embeds)
        lang_mo_embeds = self.lang_motion_fc(lang_embeds)
        if self.model_cfg.share_idloss:  
            merge_cls_t = self.id_cls3(lang_embeds)
            merge_cls_v = self.id_cls3(visual_merge_embeds)
            cls_logits_results.append(merge_cls_t)
            cls_logits_results.append(merge_cls_v)


        visual_merge_embeds, lang_merge_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds = map(lambda t: F.normalize(t, p = 2, dim = -1), (visual_merge_embeds, lang_embeds,visual_car_embeds,lang_car_embeds,visual_mo_embeds,lang_mo_embeds))

        return [(visual_car_embeds,lang_car_embeds),(visual_mo_embeds,lang_mo_embeds),(visual_merge_embeds, lang_merge_embeds)],self.logit_scale,cls_logits_results
