import torch
import torch.nn.functional as F
import global_placeholder
import numpy as np
import math
# def amplification_function(values):
#     return values ** 2 + 1

# def my_equation_add_contrast_loss(reg_loss, conf_values, pos_conf_values, ious, pos_mask):

#         conf_greater_iou_flag = pos_conf_values > ious
#         iou_greater_conf_flag = pos_conf_values < ious
#         contrast_values = torch.zeros_like(ious)
#         zeros = torch.zeros_like(ious)
#         contrast_values[conf_greater_iou_flag] = (pos_conf_values.detach() - ious)[conf_greater_iou_flag]  # 说明要拉iou    # 0.4942  0.7875
#         contrast_values[iou_greater_conf_flag] = (ious.detach() - pos_conf_values)[iou_greater_conf_flag]  # 说明要拉conf
        
        
#     #!! cls的话，要是有Conf大于IOU的情况，也不会去拉低Conf。 或者说，还是得拉低Conf？？？
#     # loss = (1 + 0.15 * torch.unsqueeze(torch.square(maximum * delta), dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
    

#     #!! cls的话，要是有Conf大于IOU的情况，也不会去拉低Conf。 或者说，还是得拉低Conf？？？
#     # loss = (1 + 0.15 * torch.unsqueeze(torch.square(maximum * delta), dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
    
#         maximum = torch.max(pos_conf_values, ious)

#         # contrast_values = torch.abs(pos_conf_values - ious)  # 0.48974 0.78683 拉
#         margin = 0.2
#         # reg_loss = reg_loss + torch.max(zeros, contrast_values - margin)  # 泰拉了
#         # reg_loss = reg_loss + contrast_values  # 0.4907  0.7888
#         # reg_loss = reg_loss + torch.sin(contrast_values * 3.1415926 / 2) ** 3  # 0.4901  0.7882
#         # reg_loss = reg_loss + torch.sin(contrast_values * 3.1415926 / 2)  # 0.4942  0.7875  就这个
#         reg_loss = reg_loss + torch.sin(contrast_values * 3.1415926 / 2) * pos_mask.float()
#         # reg_loss = reg_loss + info_factor * torch.sin(contrast_values * 3.1415926 / 2) * pos_mask.float()
#         # reg_loss = reg_loss + info_factor * contrast_values * pos_mask.float()
        
#         # reg_loss = reg_loss + maximum * contrast_values # 
    
#     return reg_loss


def my_equation_add_task_contrast_loss(reg_loss, conf_values, pos_conf_values, ious, pos_mask):

    margin = 0.2
    zeros = torch.zeros_like(ious)
    distance = torch.abs(pos_conf_values - ious)
    # 原文写法
    # delta = torch.max(zeros, distance - margin)
    # 原码写法
    delta = torch.where(distance < 0.2, zeros, distance)
    information_entropy = - conf_values * conf_values.log() # e为底
    beta = torch.exp(information_entropy.sum(dim=1))

    reg_loss = reg_loss + (1 / (1 + beta)) * delta * pos_mask.float()
    
    return reg_loss



# def my_equation_similarity_loss(kl_loss, pos_conf_values, ious, pos_mask): # JS散度
#     batch_distribution_losses = []
#     for conf, iou, mask in zip(pos_conf_values, ious, pos_mask):  # 这是多batch情况
        
#         # softmax_pos_conf_values = F.softmax(pos_conf_values[it, pos_mask[it]], dim=0)  # 化为概率分布 dim 表示batch之间不影响
#         # softmax_ious = F.softmax(ious[it, pos_mask[it]], dim=0)
        
#         conf = conf[mask]
#         iou = iou[mask]
#         elem_num = conf.numel()
#         similarity_loss = 1 - (conf * iou).mean()
#         # similarity_loss = F.pairwise_distance(conf, iou, dim=-1)
#         # cosine_loss = 1 - F.cosine_similarity(pos_conf_values[it, pos_mask[it]], ious[it, pos_mask[it]], dim=0)
        
        
#         # means = (softmax_pos_conf_values + softmax_ious) / 2
#         # distribution_loss = 0.5 * kl_loss(means.log(), softmax_pos_conf_values) + 0.5 * kl_loss(means.log(), softmax_ious) # P、Q 指代target和input
#         batch_distribution_losses.append(similarity_loss)
#     # p || q
#     # target || input
#     return sum(batch_distribution_losses) / len(batch_distribution_losses)  # NOTE 不平均是 0.4944  0.7929  不应该平均，就是每张图片都有自己的loss！注意后期实现
    
# def my_equation_js_loss(kl_loss, pos_conf_values, ious, pos_mask): # JS散度

#         batch_distribution_losses = []
#         for it in range(pos_conf_values.size()[0]):
            
#             softmax_pos_conf_values = F.softmax(pos_conf_values[it, pos_mask[it]], dim=0)  # 化为概率分布 dim 表示batch之间不影响
#             softmax_ious = F.softmax(ious[it, pos_mask[it]], dim=0)
            
#             means = (softmax_pos_conf_values + softmax_ious) / 2
#             distribution_loss = 0.5 * kl_loss(means.log(), softmax_pos_conf_values) + 0.5 * kl_loss(means.log(), softmax_ious) # P、Q 指代target和input
#             batch_distribution_losses.append(distribution_loss.sum())
#         # p || q
#         # target || input
#         return sum(batch_distribution_losses)  # NOTE 不平均是 0.4944  0.7929  不应该平均，就是每张图片都有自己的loss！注意后期实现
#     return 0 

# def my_equation_add_correlation_loss(reg_loss, conf_values, pos_conf_values, ious, pos_mask): # TODO 注意一个问题，新算出来的loss代表的是全集！还得筛选一次正样本

#         # correlation = pos_conf_values * ious + 0.00001
#         # correlation_loss = - correlation * torch.log10(correlation) - (1 - correlation) * torch.log10((1 - correlation))  # 二项分布信息熵，0.4931  0.7859
#         # correlation_loss = 0.5 * (- correlation * torch.log10(correlation) - (1 - correlation) * torch.log10((1 - correlation)))  # 二项分布信息熵，
#         # correlation_loss = (- correlation * torch.log2(correlation) - (1 - correlation) * torch.log2((1 - correlation)))  # 二项分布信息熵，拉
        
                
#         info_entropys = (- conf_values * conf_values.log()).sum(dim=1).exp()
#         info_factor = 1 / (info_entropys)  # 牛
#         # info_factor = 1.2 / (info_entropys)  # 这里的常量可以去调一调  拉
#         # info_factor = 0.8 / (info_entropys)  # 这里的常量可以去调一调   拉
        
#         minimum = torch.min(pos_conf_values, ious)
#         maximum = torch.max(pos_conf_values, ious)
        
#         correlation = pos_conf_values * ious
#         # correlation_loss = 1 - correlation  # 还行
        
#         correlation_loss = info_factor * (1 - correlation)  # 还行
        
#         # correlation_loss = (1 - maximum)(1 - correlation)  # 
#         # correlation_loss = torch.cos(correlation * 3.1415926 / 2)
#         reg_loss = reg_loss + correlation_loss * pos_mask.float()
#     return reg_loss

def my_equation_add_correlation_loss(reg_loss, cls_branch_factor, reg_branch_factor, conf_values, pos_conf_values, ious, mask): # TODO 注意一个问题，新算出来的loss代表的是全集！还得筛选一次正样本

    # correlation = pos_conf_values * ious + 0.00001
    # correlation_loss = - correlation * torch.log10(correlation) - (1 - correlation) * torch.log10((1 - correlation))  # 二项分布信息熵，0.4931  0.7859
    # correlation_loss = 0.5 * (- correlation * torch.log10(correlation) - (1 - correlation) * torch.log10((1 - correlation)))  # 二项分布信息熵，
    # correlation_loss = (- correlation * torch.log2(correlation) - (1 - correlation) * torch.log2((1 - correlation)))  # 二项分布信息熵，拉
    
    # info_factor = 1.2 / (info_entropys)  # 这里的常量可以去调一调  拉
    # info_factor = 0.8 / (info_entropys)  # 这里的常量可以去调一调   拉
    
    # minimum = torch.min(pos_conf_values, ious)
    # maximum = torch.max(pos_conf_values, ious)
    
    # correlation = torch.pow(pos_conf_values, cls_trade_off) * torch.pow(ious, reg_trade_off)
    eps = 2.220446049250313e-16# -10貌似可以，但是我狠一点
    # correlation = torch.pow(pos_conf_values+eps, 0.5) * torch.pow(ious+eps, 0.5) - eps
    correlation = torch.pow(pos_conf_values+eps, ious) * torch.pow(ious+eps, pos_conf_values) - eps
    # correlation = torch.pow(pos_conf_values+eps, ious.detach()) * torch.pow(ious+eps, pos_conf_values.detach()) - eps  # 这真不行
    # correlation = torch.pow(pos_conf_values * ious * matched_obj+eps, 1/3) - eps ** 1/3  # 还是不该加这个
    
    
    
    # correlation = torch.pow(pos_conf_values, 0.5) * torch.pow(ious, 0.5)
    # correlation = torch.pow(pos_conf_values, 0.5) * ious  # 这个就没事
    # correlation = pos_conf_values * torch.pow(ious, 0.5)  # 但是还是nan
    # correlation = (pos_conf_values ** 0.5) * (ious ** 0.5)  一样
    # correlation = pos_conf_values * ious
    
    # correlation = pos_conf_values * ious  #
    # correlation_loss =(1 - correlation) ** 2  
    # correlation_loss = torch.exp(-correlation) - torch.exp(-torch.tensor(1.))   # 实在不行，再加一个cls entropy，只对置信度大的进行处理。
    correlation_loss = (1 + torch.abs(pos_conf_values-ious).detach())*(torch.exp(-correlation) - torch.exp(-torch.tensor(1.)))   # 实在不行，再加一个cls entropy，只对置信度大的进行处理。
    
    
    
    # correlation_loss = 1.6 * (torch.exp(-correlation) - torch.exp(-torch.tensor(1.)))   # 实在不行，再加一个cls entropy，只对置信度大的进行处理。
    # correlation_loss = (1 - correlation) * ((1 + correlation) ** 0.8) # 确实这种设计相比于1-iou有用
    
    # correlation_loss = info_factor * (1 - correlation)  # 还行
    
    # correlation_loss = (1 - maximum)(1 - correlation)  # 
    # correlation_loss = torch.cos(correlation * 3.1415926 / 2)
    # information_entropy = - conf_values * conf_values.log() # e为底
    # beta = torch.exp(information_entropy.sum(dim=1)).detach()   # 所以这个玩意就应该detach
    # reg_loss = reg_loss + (1 + 1 / (beta)) * correlation_loss * pos_mask.float()  # 所以应不应该有+1呢
    
    reg_loss = reg_loss + correlation_loss * mask.float()  
    return reg_loss


def my_equation_add_entropy_iou_loss(reg_loss, conf_values, pos_conf_values, ious, pos_mask): # 轻微提点，但是实际效果就是，TP抬高了，但是TP量反而少了

    beta = 1.2
    alpha = 1.5
    gamma = 0.8
    # iou_loss = alpha * (1 - ious) * ((1 + ious) ** gamma) # 确实这种设计相比于1-iou有用
    # iou_loss = alpha * (1 - ious)
    # iou_loss = 1 - ious
    eps_ious = ious + 0.000001
    iou_loss = - eps_ious * torch.log2(eps_ious) - (1 - eps_ious) * torch.log2((1 - eps_ious))  # 1# 0.4856  0.7770- NOTE 信息熵这个的底，可以做一个消融实验
    # iou_loss = - eps_ious * torch.log(eps_ious) - (1 - eps_ious) * torch.log((1 - eps_ious))  # 烂
    

    
    reg_loss = reg_loss + iou_loss * pos_mask.float()
    return reg_loss

# def my_equation_add_gtcls_loss(cls_loss, conf_values, pos_conf_values, ious, pos_mask): # 轻微提点，但是实际效果就是，TP抬高了，但是TP量反而少了

#         # beta = 1.2
#         # alpha = 1.5
#         # gamma = 0.8
#         eps = 1e-10
#         epsp_pos_conf_values = pos_conf_values + eps
#         epsm_pos_conf_values = pos_conf_values - eps
        
#         gtcls_loss = - epsp_pos_conf_values * torch.log(epsp_pos_conf_values) - (1 - epsm_pos_conf_values) * torch.log((1 - epsm_pos_conf_values))  # cls_3
#         # gtcls_loss = 1 - pos_conf_values  # cls_1
        
#         # gtcls_loss = 0.5 * (1 - pos_conf_values) ** 2 # cls_2
#         cls_loss = cls_loss + gtcls_loss * pos_mask.float()
#     return cls_loss


def my_equation_add_harmonic_iou_loss(reg_loss, conf_values, pos_conf_values, ious, pos_mask): # 轻微提点，但是实际效果就是，TP抬高了，但是TP量反而少了

    beta = 1.2
    alpha = 1.5
    gamma = 0.8
    iou_loss = alpha * (1 - ious) * ((1 + ious) ** gamma) # 确实这种设计相比于1-iou有用

    reg_loss = reg_loss + iou_loss
    return reg_loss


def my_equation_add_harmonic_conf_loss(cls_loss, conf_values, pos_conf_values, pos_gtconf_values, pos_mask): # 轻微提点，但是实际效果就是，TP抬高了，但是TP量反而少了

    beta = 1.2
    alpha = 1.5
    gamma = 0.8
    conf_loss = alpha * (1 - pos_gtconf_values) * ((1 + pos_gtconf_values) ** gamma)

    cls_loss = cls_loss + conf_loss * pos_mask.float()
    return cls_loss

def HarDet_loss(pos_reg_loss, pos_cls_loss, conf_values, pos_gtconf_values, pos_conf_values, matched_iou_vals, pos_mask):
    pos_reg_loss = my_equation_add_harmonic_iou_loss(pos_reg_loss, None, None, matched_iou_vals, pos_mask)
    pos_reg_loss = (1 + torch.exp( - pos_cls_loss)) * pos_reg_loss * pos_mask.float() # 0.4856  0.7770-
    pos_cls_loss = (1 + torch.exp( - pos_reg_loss)) * pos_cls_loss * pos_mask.float()
    pos_reg_loss = my_equation_add_task_contrast_loss(pos_reg_loss, conf_values, pos_gtconf_values, matched_iou_vals, pos_mask)
    return pos_reg_loss, pos_cls_loss

def HQOD_loss(pos_reg_loss, pos_cls_loss, conf_values, pos_gtconf_values, pos_conf_values, matched_iou_vals, cls_branch_factor, reg_branch_factor, mask):  # 以前那个真不行，就是a/(b+1)
    # pos_reg_loss = my_equation_add_harmonic_iou_loss(pos_reg_loss, None, None, matched_iou_vals, pos_mask)
    # pos_reg_loss = (1 + pos_gtconf_values) * pos_reg_loss * pos_mask.float() # 0.4856  0.7770-
    # pos_cls_loss = (1 + matched_iou_vals) * pos_cls_loss * pos_mask.float()
    # pos_reg_loss = torch.exp( 1 - matched_iou_vals) * pos_reg_loss * pos_mask.float() # 0.4856  0.7770-
    # pos_cls_loss = torch.exp( 1 - pos_gtconf_values) * pos_cls_loss * pos_mask.float()
    # pos_reg_loss = my_equation_add_task_contrast_loss(pos_reg_loss, conf_values, pos_gtconf_values, matched_iou_vals, pos_mask)

    pos_reg_loss = my_equation_add_harmonic_iou_loss(pos_reg_loss, None, None, matched_iou_vals, mask)
    # pos_cls_loss  = my_equation_add_harmonic_conf_loss(pos_cls_loss, conf_values, pos_conf_values, pos_gtconf_values, pos_mask)
    pos_reg_loss = my_equation_add_correlation_loss(pos_reg_loss, cls_branch_factor, reg_branch_factor, conf_values, pos_gtconf_values, matched_iou_vals, mask)
    
    return pos_reg_loss, pos_cls_loss
    
# def HQOD_loss(pos_reg_loss, pos_cls_loss, conf_values, pos_gtconf_values, pos_conf_values, matched_iou_vals, pos_mask):
#     # pos_reg_loss = (1 + pos_gtconf_values) * (1 + pos_conf_values ** 1.7) * pos_reg_loss * pos_mask.float() # 0.4895  -0.7779?
#     # pos_cls_loss = (1 + matched_iou_vals ** 1.7) * pos_cls_loss * pos_mask.float() # 
    
#     # pos_reg_loss = ((1 + pos_gtconf_values) * (1 + pos_conf_values)).sqrt() * pos_reg_loss * pos_mask.float() # 好 1
#     # pos_cls_loss = (1 + matched_iou_vals) * pos_cls_loss * pos_mask.float() #     
#     # pos_reg_loss = my_equation_add_entropy_iou_loss(pos_reg_loss, None, None, matched_iou_vals, pos_mask)
    
#     # pos_reg_loss = (1 + pos_conf_values) * pos_reg_loss * pos_mask.float() # 这一个真不行
#     # pos_reg_loss = my_equation_add_harmonic_iou_loss(pos_reg_loss, None, None, matched_iou_vals, pos_mask)
        
#     pos_reg_loss = (1 + pos_gtconf_values) * pos_reg_loss * pos_mask.float() # 好 2
#     pos_cls_loss = (1 + matched_iou_vals) * pos_cls_loss * pos_mask.float() #  
#     # pos_reg_loss = (1 + pos_gtconf_values ** 1.7) * pos_reg_loss * pos_mask.float() # 差
#     # pos_cls_loss = (1 + matched_iou_vals ** 1.7) * pos_cls_loss * pos_mask.float() #  
#     # pos_reg_loss = (1 + pos_conf_values) * pos_reg_loss * pos_mask.float() # 差
#     # pos_cls_loss = (1 + matched_iou_vals) * pos_cls_loss * pos_mask.float() #  
#     # pos_reg_loss = ((1 + pos_gtconf_values) * (1 + pos_conf_values)) * pos_reg_loss * pos_mask.float()  # 差
#     # pos_cls_loss = (1 + matched_iou_vals) ** 2 * pos_cls_loss * pos_mask.float() # 
#     # pos_reg_loss = (1 + pos_conf_values * pos_gtconf_values) * pos_reg_loss * pos_mask.float() # 差
#     # pos_cls_loss = (1 + matched_iou_vals) * pos_cls_loss * pos_mask.float() # 
#     pos_reg_loss = my_equation_add_entropy_iou_loss(pos_reg_loss, None, None, matched_iou_vals, pos_mask)
    
#     # pos_reg_loss = my_equation_add_correlation_loss(pos_reg_loss, pos_conf_values, matched_iou_vals, pos_mask) # 不要
#     # pos_reg_loss = my_equation_add_harmonic_iou_loss(pos_reg_loss, None, None, matched_iou_vals, pos_mask)
#     # pos_reg_loss = my_equation_add_iou_loss(pos_reg_loss, None, None, matched_iou_vals, pos_mask)
#     # pos_reg_loss = my_equation_add_task_contrast_loss(pos_reg_loss, conf_values, pos_gtconf_values, matched_iou_vals, pos_mask)
    
#     return pos_reg_loss, pos_cls_loss

# def my_equation_add_correlation_loss(reg_loss, pos_conf_values, ious, pos_mask): # TODO 注意一个问题，新算出来的loss代表的是全集！还得筛选一次正样本
#     correlation = pos_conf_values * ious + 0.000001
#     correlation_loss = - correlation * torch.log10(correlation) - (1 - correlation) * torch.log10((1 - correlation))  # 二项分布信息熵，0.4931  0.7859
#     # correlation_loss = 0.5 * (- correlation * torch.log10(correlation) - (1 - correlation) * torch.log10((1 - correlation)))  # 二项分布信息熵，
#     # correlation_loss = (- correlation * torch.log2(correlation) - (1 - correlation) * torch.log2((1 - correlation)))  # 二项分布信息熵，拉
#     # correlation_loss = 1 - correlation  # 拉了
#     reg_loss = reg_loss + correlation_loss * pos_mask.float()
#     return reg_loss
# def my_equation_add_intensive_iou_loss(reg_loss, iou):
#     reg_loss = reg_loss + (1 - iou)
#     return reg_loss

# def my_equation_add_kl_between_loss(reg_loss, )



# def my_equation_for_cls(amp_values, delta, loss): # IOU拉Conf
#     # loss = (1 + 0.5 * torch.unsqueeze(maximum * delta, dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance
#     # loss = (1 + 0.8 * torch.unsqueeze(torch.square(maximum) * delta, dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
#     # loss = (1 + 0.8 * torch.unsqueeze(maximum * torch.square(delta), dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
    
#     #!! cls的话，要是有Conf大于IOU的情况，也不会去拉低Conf。 或者说，还是得拉低Conf？？？
#     # loss = (1 + 0.15 * torch.unsqueeze(torch.square(maximum * delta), dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
    
#     # temp_weight = torch.square(maximum * delta)
#     # temp_weight = maximum * delta  # 不行
#     # temp_weight = 0.5 * delta

#         qbit = global_placeholder.quant_bit
#         if qbit == None: qbit = 8 # 避免fp32时
#         qbit = float(qbit)
#         # amp_values = amplification_function(amp_values)
#         # temp_weight = (qbit / 8)**2 * delta
        
#         # temp_weight = (qbit / 8)**2 * torch.sin(delta * 3.1415926 / 2)
        
#         # temp_weight = (qbit / 8)**2 * amp_values * torch.sin(delta * 3.1415926 / 2)
#         temp_weight = delta  # 
#         # temp_weight = torch.sin(delta * 3.1415926 / 2)  # 目前最优
#         if temp_weight.shape != loss.shape:
#             # 如果有问题的话
#             temp_weight = torch.unsqueeze(temp_weight, dim=1)
#         loss = (1 + temp_weight) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
#         # loss = (loss + 1) ** (1 + temp_weight) - 1
    
#     # 回归的话，Conf带不了他
#     return loss

# def my_equation_for_reg(amp_values, delta, loss):  # Conf拉IOU
#     # loss = (1 + 0.5 * torch.unsqueeze(maximum * delta, dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance
#     # loss = (1 + 0.8 * torch.unsqueeze(torch.square(maximum) * delta, dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
#     # loss = (1 + 0.8 * torch.unsqueeze(maximum * torch.square(delta), dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
    
#     # !! cls的话，要是有Conf大于IOU的情况，也不会去拉低Conf。 或者说，还是得拉低Conf？？？
#     # loss = (1 + 0.8 * torch.unsqueeze(torch.square(maximum * delta), dim=1)) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
    
#     # temp_weight = torch.square(maximum * delta)
#     # temp_weight = maximum * delta
#     # temp_weight = 0.5 * delta

#         qbit = global_placeholder.quant_bit
#         if qbit == None: qbit = 8 # 避免fp32时
#         qbit = float(qbit)
#         # amp_values = amplification_function(amp_values)
#         # temp_weight = (qbit / 8)**2 * delta
        
#         # temp_weight = (qbit / 8)**2 * torch.sin(delta * 3.1415926 / 2) # sin^2?
        
#         # temp_weight = (qbit / 8)**2 * amp_values * torch.sin(delta * 3.1415926 / 2)
#         temp_weight = delta # 
#         # temp_weight = torch.sin(delta * 3.1415926 / 2) # 目前最优
#         if temp_weight.shape != loss.shape:
#             # 如果有问题的话
#             temp_weight = torch.unsqueeze(temp_weight, dim=1)
#         loss = (1 + temp_weight) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
#         # loss = (loss + 1) ** (1 + temp_weight) - 1
#     # 回归的话，Conf带不了他
#     return loss



# def my_equation_for_cls(amp_values, delta, loss): # IOU拉Conf

#         qbit = global_placeholder.quant_bit
#         if qbit == None: qbit = 8 # 避免fp32时
#         qbit = float(qbit)
#         temp_weight = delta  # 
#         if temp_weight.shape != loss.shape:
#             # 如果有问题的话
#             temp_weight = torch.unsqueeze(temp_weight, dim=1)
#         loss = (1 + temp_weight) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
    
#     # 回归的话，Conf带不了他
#     return loss

# def my_equation_for_reg(amp_values, delta, loss):  # Conf拉IOU

#         qbit = global_placeholder.quant_bit
#         if qbit == None: qbit = 8 # 避免fp32时
#         qbit = float(qbit)
#         temp_weight = delta # 
#         if temp_weight.shape != loss.shape:
#             # 如果有问题的话
#             temp_weight = torch.unsqueeze(temp_weight, dim=1)
#         loss = (1 + temp_weight) * loss  # TODO 还有一个情况就是，数据分布也是imbalance  就应该是更关注高值情况下的差异
#     return loss


# def harmonic_loss(loss)