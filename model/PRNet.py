import torch
from torch import nn
import torch.nn.functional as F

from model.resnet import *
from model.loss import WeightedDiceLoss
from model.PRT_transformer import CyCTransformer as PRTransformer
from model.ops.modules import MSDeformAttn
from model.backbone_utils import Backbone


def masked_average_pooling(feature, mask):
    masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
    return masked_feature




class PRnetwork(nn.Module):
    def __init__(self, layers=50, classes=2, shot=1, reduce_dim=384, \
                 criterion=WeightedDiceLoss(), with_transformer=True,with_LPF=True, trans_multi_lvl=1):
        super(PRnetwork, self).__init__()
        assert layers in [50, 101]
        assert classes > 1
        self.layers = layers
        self.criterion = criterion
        self.shot = shot
        self.with_transformer = with_transformer
        self.with_LPF=with_LPF
        if self.with_transformer:
            self.trans_multi_lvl = trans_multi_lvl
        self.reduce_dim = reduce_dim

        self.print_params()

        in_fea_dim = 1024 + 512

        drop_out = 0.5

        self.adjust_feature_supp = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )
        self.adjust_feature_qry = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )

        self.high_avg_pool = nn.AdaptiveAvgPool1d(reduce_dim)

        prior_channel = 1
        self.qry_merge_feat = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + prior_channel, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        if self.with_transformer:
            self.supp_merge_feat = nn.Sequential(
                nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )

            self.transformer = PRTransformer(embed_dims=reduce_dim, num_points=9, shot=self.shot,with_LPF=self.with_LPF)
            self.merge_multi_lvl_reduce = nn.Sequential(
                nn.Conv2d(reduce_dim * 1, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.merge_multi_lvl_sum = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

        self.merge_res = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.ini_cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.init_weights()
        self.backbone = Backbone('resnet{}'.format(layers), train_backbone=False, return_interm_layers=True,
                                 dilation=[False, True, True])

        self.aux_loss = nn.BCEWithLogitsLoss()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def print_params(self):
        repr_str = self.__class__.__name__
        repr_str += f'(backbone layers={self.layers}, '
        repr_str += f'reduce_dim={self.reduce_dim}, '
        repr_str += f'shot={self.shot}, '
        repr_str += f'with_transformer={self.with_transformer})'
        print(repr_str)
        return repr_str

    def forward(self, x, s_x, s_y, y):
        # print(f"s_y:{s_y.shape}")      batch X shot X 473 X 473
        # print(f"s_x:{s_x.shape}")      batch X shot X 3 X 473 X 473
        #print(f"x:{x.shape}")            batch X 3 X 473 X 473
        #print(f"y:{y.shape}")            batch X 473 X 473
        batch_size, _, h, w = x.size()
        assert (h - 1) % 8 == 0 and (w - 1) % 8 == 0
        img_size = x.size()[-2:]

        # backbone feature extraction
        qry_bcb_fts = self.backbone(x)
        supp_bcb_fts = self.backbone(s_x.view(-1, 3, *img_size))
        query_feat = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        supp_feat = torch.cat([supp_bcb_fts['1'], supp_bcb_fts['2']], dim=1)
        query_feat = self.adjust_feature_qry(query_feat)
        supp_feat = self.adjust_feature_supp(supp_feat)

        fts_size = query_feat.shape[-2:]
        query_feat_high = qry_bcb_fts['3']
        supp_feat_high = supp_bcb_fts['3'].view(batch_size, -1, 2048, fts_size[0], fts_size[1])
        # supp_mask = F.interpolate((s_y == 1).view(-1, *img_size).float().unsqueeze(1), size=(fts_size[0], fts_size[1]),
        #                           mode='bilinear', align_corners=True)

        # global feature extraction
        supp_feat_list = []
        supp_feat_high_list=[]
        FP_list=[]
        r_supp_feat = supp_feat.view(batch_size, self.shot, -1, fts_size[0], fts_size[1])
        for st in range(self.shot):
            mask = (s_y[:, st, :, :] == 1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            tmp_supp_feat = r_supp_feat[:, st, ...]
            tmp_supp_feat_high=supp_feat_high[:, st, ...]
            tmp_supp_feat = masked_average_pooling(tmp_supp_feat, mask).unsqueeze(-1).unsqueeze(-1)
            tmp_supp_feat_high = masked_average_pooling(tmp_supp_feat_high, mask)
            SSFP = self.SSP_func(query_feat, tmp_supp_feat)
            FP=0.5*SSFP+0.5*tmp_supp_feat
            FP_list.append(FP)
            supp_feat_list.append(tmp_supp_feat)
            supp_feat_high_list.append(tmp_supp_feat_high)
        if self.shot > 1:
            global_supp_pp = torch.mean(torch.stack(supp_feat_list, dim=0), dim=0)
            SSFP = self.SSP_func(query_feat, global_supp_pp)
            global_supp_pp=0.5*SSFP+0.5*global_supp_pp
            multi_supp_pp = torch.cat(FP_list,dim=0)
        else:
            multi_supp_pp = FP_list[0]
            global_supp_pp = FP_list[0]


        # prior generation
        global_supp_pp_high=torch.mean(torch.stack(supp_feat_high_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)  # batch*1024*1*1
        SSFP = self.SSP_func(query_feat_high, global_supp_pp_high)
        global_supp_pp_high=0.5*SSFP+0.5*global_supp_pp_high
        corr_query_mask=F.cosine_similarity(query_feat_high, global_supp_pp_high, dim=1).unsqueeze(1)

        prior=corr_query_mask

        # feature mixing
        query_cat_feat = [query_feat, global_supp_pp.expand(-1, -1, fts_size[0], fts_size[1]), corr_query_mask]
        query_feat = self.qry_merge_feat(torch.cat(query_cat_feat, dim=1))

        query_feat_out = self.merge_res(query_feat) + query_feat
        init_out = self.ini_cls(query_feat_out)
        init_mask = init_out.max(1)[1]
        to_merge_fts = [supp_feat, multi_supp_pp.expand(-1, -1, fts_size[0], fts_size[1])]
        aug_supp_feat = torch.cat(to_merge_fts, dim=1)
        aug_supp_feat = self.supp_merge_feat(aug_supp_feat)

        query_feat_list, qry_outputs_mask_list = self.transformer(query_feat, y.float(),aug_supp_feat,s_y.clone().float(),global_supp_pp,init_mask.detach())
        # fused_query_feat = torch.cat(query_feat_list, dim=1)
        fused_query_feat = query_feat_list
        fused_query_feat = self.merge_multi_lvl_reduce(fused_query_feat)
        fused_query_feat = self.merge_multi_lvl_sum(fused_query_feat) + fused_query_feat

        # Output Part
        out = self.cls(fused_query_feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:

            # calculate loss
            main_loss = self.criterion(out, y.long())

            aux_loss_q = torch.zeros_like(main_loss)

            init_out = F.interpolate(init_out, size=(h, w), mode='bilinear', align_corners=True)
            main_loss2 = self.criterion(init_out, y.long())

            for qy_id, qry_out in enumerate(qry_outputs_mask_list):
                q_gt = F.interpolate(((y == 1) * 1.0).unsqueeze(1), size=qry_out.size()[2:], mode='nearest')
                aux_loss_q = aux_loss_q + self.aux_loss(qry_out, q_gt)
            aux_loss_q = aux_loss_q / len(qry_outputs_mask_list)

            aux_loss = aux_loss_q

            return out.max(1)[1], 0.7 * main_loss + 0.3 * main_loss2, aux_loss
        else:
            return out,prior

    def SSP_func(self, feature_q, proto_fg):
        #q_mask=F.interpolate(y.float().unsqueeze(0), size=(feature_q.shape[2], feature_q.shape[3]), mode='bilinear', align_corners=True)
        bs,c = feature_q.shape[0:2]
        pred_fg = F.cosine_similarity(feature_q, proto_fg, dim=1).view(bs,-1)  # batch*3600
        fg_ls = []
        for epi in range(bs):
            fg_thres = 0.7  # 0.9 #0.6        #这是前景阀值
            cur_feat = feature_q[epi].view(c, -1)  # 1024*3600
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]  # .mean(-1)   1024*N1    根据阀值进行前景特征选择
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]  # .mean(-1)

            # global proto
            fg_proto = fg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))


        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        return new_fg

