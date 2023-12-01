import torch
import torch.nn as nn
import numpy as np

import model.clip.clip
from model.clip.dino import vit_base
from utils import utils_dino

from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.cluster_check = cfg.DATASETS.CLUSTER
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        if cfg.MODEL.DINO_TEACHER:
            # self.classifier = nn.Linear(self.in_planes*2, self.num_classes, bias=False)
            # self.classifier.apply(weights_init_classifier)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_self = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_self.apply(weights_init_classifier)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        if cfg.MODEL.DINO_TEACHER:
            self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
            self.classifier_self_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
            self.classifier_proj.apply(weights_init_classifier)
            self.classifier_self_proj.apply(weights_init_classifier)

        else :
            self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
            self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        # Load Clip
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        if cfg.MODEL.VISUAL_MODEL == 'dino_vit':
            self.image_encoder = clip_model.dino
        elif cfg.MODEL.VISUAL_MODEL == 'clipreid_vit':
            self.image_encoder = clip_model.visual
            if cfg.MODEL.DINO_TEACHER:

                self.dino_encoder = vit_base(patch_size=16, num_classes=0,
                                     img_size=[self.h_resolution * self.vision_stride_size, self.w_resolution * self.vision_stride_size])
                utils_dino.load_pretrained_weights(self.dino_encoder, cfg.MODEL.DINO_PRETRAIN_PATH, 'teacher',
                                                   'vit_base', 16)

                for param in self.dino_encoder.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("The visual model is not predefined in the project.")

        self.config=cfg
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        if cfg.DATASETS.CLUSTER:
            self.prompt_learner = PromptLearnerCluster(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        else :
            self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding,ctx_init_cfg=cfg.INPUT.PROMPT_BASE)
        self.tokenizer = clip.tokenize
        self.text_encoder = TextEncoder(clip_model)

        self.token_embedding = clip_model.token_embedding

        self.unembedder = clip.unembedding
        self.untokenizer = clip.untokenize

    # The code for interpretation is adapted from the approach described in the paper available at
    # https://github.com/hila-chefer/Transformer-MM-Explainability.
    # paper : https://arxiv.org/abs/2103.15679
    def interpret_(self,image, texts, device, start_layer, start_layer_text):
        batch_size = image.shape[0]
        #images = image.repeat(batch_size, 1, 1, 1)

        logits_per_image = self.classifier(self.forward(image)[:,:self.classifier.weight.shape[1]])
        # logits_per_text = self.text_encoder(texts)

        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]

        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)

        self.image_encoder.zero_grad()
        self.text_encoder.zero_grad()
        if 'DINO' in str(type(self.image_encoder)):
            image_attn_blocks = list(dict(self.image_encoder.blocks.named_children()).values())
            num_tokens = image_attn_blocks[0].attn.shape[-1]
        else:
            image_attn_blocks = list(dict(self.image_encoder.transformer.resblocks.named_children()).values())
            num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        if start_layer == -1:
            # calculate index of last layer
            start_layer = len(image_attn_blocks) - 1

        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            if 'DINO' in str(type(self.image_encoder)):
                grad = torch.autograd.grad(one_hot, [blk.attn], retain_graph=True)[0].detach()
                qcam = blk.attn.detach()
            else:
                grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
                qcam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]

        # text_attn_blocks = list(dict(self.text_encoder.resblocks.named_children()).values())
        #
        # if start_layer_text == -1:
        #     # calculate index of last layer
        #     start_layer_text = len(text_attn_blocks) - 1
        #
        # num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        # R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
        # R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        # for i, blk in enumerate(text_attn_blocks):
        #     if i < start_layer_text:
        #         continue
        #     grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        #     cam = blk.attn_probs.detach()
        #     cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        #     grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        #     cam = grad * cam
        #     cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        #     cam = cam.clamp(min=0).mean(dim=1)
        #     R_text = R_text + torch.bmm(cam, R_text)
        # text_relevance = R_text

        return image_relevance
    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            # decode here
            # text_out=[]
            # for i in range(prompts.shape[0]):
            #     unembedd = self.unembedder(prompts[i],self.token_embedding)
            #     text_out.append(self.untokenizer(unembedd))
            # _tokenizer.decode(self.prompt_learner.tokenized_prompts.tolist()[0])
            return text_features#,text_out

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]
            if self.config.MODEL.DINO_TEACHER:
                image_features_last_dino, image_features_dino, image_features_proj_dino = self.dino_encoder(x, cv_embed)
                img_feature_last_dino = image_features_last_dino[:,0]
                img_feature_dino = image_features_dino[:,0]
                img_feature_proj_dino = image_features_proj_dino[:,0]



        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        if self.config.MODEL.DINO_TEACHER:
            feat_dino = self.bottleneck(img_feature_dino)
            feat_proj_dino = self.bottleneck_proj(img_feature_proj_dino)
        if self.training:
            # DINO teacher
            if self.config.MODEL.DINO_TEACHER:
                # Concat strategy
                # cls_score = self.classifier(torch.cat((feat, feat_dino), dim=1))
                # cls_score_proj = self.classifier_proj(torch.cat((feat_proj,feat_proj_dino),dim=1))
                cls_score = self.classifier(feat)
                cls_score_proj = self.classifier_proj(feat_proj)
                return [cls_score, cls_score_proj], [img_feature_last, img_feature,
                                                     img_feature_proj], img_feature_proj, img_feature_proj_dino
            else:
                cls_score = self.classifier(feat)
                cls_score_proj = self.classifier_proj(feat_proj)
                return [cls_score, cls_score_proj], [img_feature_last, img_feature,
                                                         img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
                # return self.classifier(feat)

    def forward_clustered(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None,cluster=None):
        if get_text == True:
            if self.cluster_check:
                prompts = self.prompt_learner((label,cluster))
            else:
                prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            # decode here
            # text_out=[]
            # for i in range(prompts.shape[0]):
            #     unembedd = self.unembedder(prompts[i],self.token_embedding)
            #     text_out.append(self.untokenizer(unembedd))
            # _tokenizer.decode(self.prompt_learner.tokenized_prompts.tolist()[0])
            return text_features  # ,text_out

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
                # return self.classifier(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # dino model has student teacher naming
            if 'student' in i:
                continue
            if 'dino_vit' in i:
                continue
            #     i=i.replace('dino_vit','image_encoder') # edit dino_vit to image_encoder
            if 'classifier'in i:
                # continue
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding,ctx_init_cfg):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        elif dataset_name == 'msmt17clustered':
            ctx_init = "A photo of a X X X X person."
            ctx_init_clustered = "A photo of two X X X X people."
        else:
            # ctx_init = "A photo of a X X X X person." #ctx_init_cfg
            ctx_init = ctx_init_cfg  # ctx_init_cfg

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()

        if dataset_name == 'msmt17clustered':
            tokenized_prompts_clustered = clip.tokenize(ctx_init_clustered).cuda()


        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
            if dataset_name == 'msmt17clustered':
                embedding_clustered = token_embedding(tokenized_prompts_clustered).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        if dataset_name == 'msmt17clustered':
            self.tokenized_prompts_clustered = tokenized_prompts_clustered

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)  # n_classes, 4, 512


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])
        if dataset_name == 'msmt17clustered':
            self.register_buffer("token_prefix_cluster", embedding[:, :n_ctx + 1, :])
            self.register_buffer("token_suffix_cluster", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] # b, 4, 512
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) # b 77 512

        return prompts


class PromptLearnerCluster(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        elif dataset_name == 'msmt17clustered':
            ctx_init = "A photo of a X X X X person."
            ctx_init_clustered = "A photo of two X X X X people."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()

        if dataset_name == 'msmt17clustered':
            tokenized_prompts_clustered = clip.tokenize(ctx_init_clustered).cuda()

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
            if dataset_name == 'msmt17clustered':
                embedding_clustered = token_embedding(tokenized_prompts_clustered).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        if dataset_name == 'msmt17clustered':
            self.tokenized_prompts_clustered = tokenized_prompts_clustered

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)  # n_classes, 4, 512

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.register_buffer("token_prefix_cluster", embedding_clustered[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix_cluster", embedding_clustered[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, input_):

        if len(input_)==2:
            label,cluster = input_
        else:
            label=input_
            # cluster
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]

        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        cluster_ind = []
        if len(input_) == 2:

            for i in range(b):
                if cluster[i]==True:
                    cluster_ind.append(i)
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        if len(cluster_ind)!=0:
            for i in cluster_ind:
                prefix_cluster = self.token_prefix_cluster.squeeze()
                suffix_cluster = self.token_suffix_cluster.squeeze()
                prompts[i]=torch.cat(
                    [
                        prefix_cluster, # 1,dim
                        cls_ctx[i],# n_ctx,dim
                        suffix_cluster# *,dim
                ],
                dim=0,
                )

        return prompts

