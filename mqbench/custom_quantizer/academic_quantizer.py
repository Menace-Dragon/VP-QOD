import copy
from collections import OrderedDict
from distutils.log import warn
from typing import List
import global_placeholder
import operator
import mmcv
import torch
from torch.fx import GraphModule
from torch.quantization import propagate_qconfig_
from torch.quantization.fx.qconfig_utils import get_flattened_qconfig_dict
import torch.nn.functional as F
import torch.nn as nn
from mqbench.utils import is_symmetric_quant, getitem2node
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer
from mqbench.fake_quantize.tqt import TqtFakeQuantize
from torch.quantization.quantize_fx import _fuse_fx
import mqbench.nn.intrinsic as qnni 
import mqbench.nn.intrinsic.qat as qnniqat
import torch.nn.intrinsic as nni
from torch.nn.parameter import Parameter

@register_model_quantizer(BackendType.Academic)  # 装饰器，妙
class AcademicQuantizer(ModelQuantizer):
    """Academic setting mostly do not merge BN and leave the first and last layer to higher bits.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.io_module = {}
        self.post_act_8bit_node_name = []
        # self.additional_qat_module_mapping = {
        #     # Intrinsic modules:
        #     nni.ConvBn2d: qnniqat.ConvBn2d,
        #     nni.ConvBnReLU2d: qnniqat.ConvBnReLU2d,
        #     nni.ConvReLU2d: qnniqat.ConvReLU2d,
        # }
        

    def prepare(self, model: GraphModule, qconfig, further_detail=dict(), testing=False):
        if global_placeholder.fold_bn_flag:
            # 进行bn fuse  TODO 验证一下，量化插入逻辑对不对；影响的optimi;   weight是成功插入了
            model = _fuse_fx(model, self.extra_fuse_dict)
        
        specified_general_quantizers = further_detail.get('specified_general_quantizers', [])
        last_8bit_module = further_detail.get('last_8bit_module', [])  # TODO 要主动给出I！！
        self.exclude_module_name = further_detail.get('exclude_prefixes', [])  # 不进行量化
        self.removed_quantizer_names = further_detail.get('removed_quantizer_names', [])  # 不进行量化
        qloss_flag = further_detail.get('qloss_flag', False)
    
        self._get_io_module(model, last_8bit_module)  # XXX 找出真正意义的首node和尾node（可不止一个），存到self的dict中。理论上来说，就应该是实实在在的layer！！！！
        self._get_post_act_8bit_node_name(model)  # 根据首尾layer！，找出各自前一个node!，存到self的dict中.这些node后会插入8bitact quant
        model = self._weight_quant(model, qconfig, testing=testing)  # 利用io_module，为module layer插入 weight quantizer
        model, node_to_quantize_output = self._insert_fake_quantize_for_act_quant(model, qconfig, specified_general_quantizers, testing=testing)  # 同时利用post_act ，插入act quantizer
        if qloss_flag:
            self.open_qloss(model)
        # if global_placeholder.quant_algorithm == 'tqt':
        #     logger.info(f'\nNow initialize type of TQT quantizers \n')
        # 这里是针对pot scale形式的量化。关键是fold bn 后的bias就是pot scale形式的
        self._set_quant_type(model, node_to_quantize_output)
        return model

    def open_qloss(self, model):
        qloss_flag = global_placeholder.qloss_flag
        if qloss_flag:
            for name, module in model.named_modules():
                if hasattr(module, 'compute_qloss'):
                    # 说明是quantizer
                    module.compute_qloss = True 
                    # 111111111111111   and 'getitem' not in name
                    
                    # if 'post_act' in name:
                    #     module.compute_qloss = True # 22222222222
                    #     module.regular_margin = Parameter(torch.tensor([1.]))
                        
                    #     # # 说明是act量化器
                    #     # module.identity = 2
                    
                    # # else:
                    # #     # 说明是quantizer
                    # #     module.compute_qloss = True # 22222222222
                    # #     # 说明是weight量化器
                    # #     # module.identity = 1
                    # #     # module.compute_qloss = True
                    # #     # module.regular_margin = Parameter(torch.tensor([1.]))
                

    def _weight_quant(self, model: GraphModule, qconfig, testing=False):  # 为每个layer标上qconfig
        logger.info("Replace module to qat module.")
        
        wq_sign = qconfig.weight.p.keywords.pop('sign')
        wq_bit = qconfig.weight.p.keywords.pop('bit')
        wqconfig_8bit = copy.deepcopy(qconfig)
        wq_symmetry = True if is_symmetric_quant(qconfig.weight.p.keywords['qscheme']) else False
        wqconfig_8bit.weight.p.keywords['quant_min'] = -2 ** (8 - 1) if wq_symmetry else 0
        wqconfig_8bit.weight.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if wq_symmetry else 2 ** 8 - 1
        wqconfig_8bit.weight.p.keywords['dtype'] = torch.qint8 if wq_symmetry else torch.quint8
        
        for name, module in model.named_modules():  # XXX 原来GraphModule也储存着原先torch.nn
            if name in self.io_module.keys():
                logger.info("Set layer {} to 8 bit.".format(name))
                module.qconfig = wqconfig_8bit
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        if not testing:
            propagate_qconfig_(model, flattened_qconfig_dict)  # XXX 这是torch官方的函数，就是绑定qconfig。为所有的层或着叫节点绑定qconfig属性。
        else:
            warn('只量化首尾！！')
        self._qat_swap_modules(model, self.additional_qat_module_mapping)  # 为layer插入weight quantizer
        return model

    @property
    def function_type_to_quant_input(self) -> list:
        return self.additional_function_type + [
            # operator.add,
            # operator.mul,
            # torch.nn.functional.adaptive_avg_pool2d,
            # torch.nn.functional.max_pool2d,
            # torch.nn.functional.avg_pool2d,
            # torch.flatten,
            # 'mean',
            # 'sum',
            # # torch.nn.functional.interpolate,
            
            # mmcv.cnn.bricks.swish.Swish,
            # mmcv.cnn.bricks.activation.Clamp,
            # mmcv.cnn.bricks.hsigmoid.HSigmoid
        ]

    def _set_quant_type(self, model: GraphModule, tensor_type_set):
        # tensor_type_set = self._find_act_quants(model) # 可以复用输入
        params_type_set = self._find_weight_quants(model)
        inputs_type_set = self._find_input_quants(model)
        module_dict = dict(model.named_modules())
        quantizer_prefix = "_post_act_fake_quantizer"

        for node in tensor_type_set:
            if isinstance(node.name, str) and (node.name + quantizer_prefix) in module_dict:
                next_op = module_dict[node.name + quantizer_prefix]
                if isinstance(next_op, TqtFakeQuantize):
                    next_op.set_quant_type('tensor') # 就是指定act量化节点的类型
                    logger.info(f'{node.name + quantizer_prefix} has been set to quant type <tensor>')
        for node in params_type_set:
            if isinstance(node.target, str) and node.target in module_dict:
                op = module_dict[node.target]
                if hasattr(op, 'weight_fake_quant'):
                    if isinstance(op.weight_fake_quant, TqtFakeQuantize):
                        op.weight_fake_quant.set_quant_type('param')
                        logger.info(f'{node.target} has been set to quant type <param/weight>')
                if hasattr(op, 'bias_fake_quant'):  # NOTE TODO 有趣，其实是可以给出bias_fake_quant。其实在本文，就是走academic quantization。
                    if isinstance(op.bias_fake_quant, TqtFakeQuantize):
                        op.bias_fake_quant.set_quant_type('param')
                        logger.info(f'{node.target} has been set to quant type <param/bias>')
        for node in inputs_type_set:
            if isinstance(node.target, str) and node.target in module_dict:
                next_op = module_dict[node.target]    
                if isinstance(next_op, TqtFakeQuantize):
                    next_op.set_quant_type('input')
                    logger.info(f'{node.target} has been set to quant type <input>')
    
    def _find_input_quants(self, model) -> List:
        node_need_to_quantize_weight = []
        nodes = list(model.graph.nodes)
        for node in nodes:
            if node.op == 'placeholder' and node.all_input_nodes == []:
                node_need_to_quantize_weight.append(list(node.users)[0])
        return node_need_to_quantize_weight

    def _find_weight_quants(self, model) -> List:
        node_need_to_quantize_weight = []
        nodes = list(model.graph.nodes)
        module_dict = dict(model.named_modules())
        for node in nodes:
            if node.target in module_dict:
                if hasattr(module_dict[node.target], 'weight_fake_quant') or hasattr(module_dict[node.target], 'bias_fake_quant'):
                    node_need_to_quantize_weight.append(node)
        return node_need_to_quantize_weight

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (  # 也就是说，带有weight quantizer的都是属于此
            # Conv            # Conv
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
            torch.nn.intrinsic.qat.modules.linear_relu.LinearReLU,
            torch.nn.qat.modules.conv.Conv2d,
            qnniqat.ConvBnReLU2d,
            qnniqat.ConvBn2d,
            qnniqat.ConvReLU2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
            # # Pooling
            # torch.nn.modules.pooling.AvgPool2d,
            # torch.nn.modules.pooling.AdaptiveAvgPool2d,
            # torch.nn.modules.pooling.MaxPool2d,
            
            # mmcv.cnn.bricks.swish.Swish,
            # mmcv.cnn.bricks.activation.Clamp,
            # mmcv.cnn.bricks.hsigmoid.HSigmoid
            
        ) + self.additional_module_type

    def _get_post_act_8bit_node_name(self, model):
        for nodes in self.io_module.values():
            for node in nodes:
                for _arg in node.args:
                    if isinstance(_arg, list):
                        print('{} node 的输入比较多（反复使用）！'.format(node.name))
                        for a_arg in _arg:
                            if isinstance(a_arg, torch.fx.node.Node):
                                self.post_act_8bit_node_name.append(a_arg.name)
                    elif isinstance(_arg, torch.fx.node.Node):
                        self.post_act_8bit_node_name.append(_arg.name)
                    
        # # 原写法
        # for node in self.io_module.values():
        #     for _arg in node.args:
        #         if isinstance(_arg, list):
        #             print('{} node 的输入比较多（反复使用）！'.format(node.name))
        #             for a_arg in _arg:
        #                 if isinstance(a_arg, torch.fx.node.Node):
        #                     self.post_act_8bit_node_name.append(a_arg.name)
        #         elif isinstance(_arg, torch.fx.node.Node):
        #             self.post_act_8bit_node_name.append(_arg.name)


    def _get_io_module(self, model, bit8_last_module_names):
        
        # 导入全局设置
        # model_type = global_placeholder.model_type
        # model_type = model_type.split('_')[0]  # 直取大类
        # bit8_last_module_names = last_module_names[model_type]
        # bit8_last_module_names = []  # dummy
        
        nodes = list(model.graph.nodes)
        for node in nodes:
            total_args = []
            the_first_layer = False  # NOTE 这个first layer 找法其实也有点问题，因为palceholder node 的下一个node不一定是module
            for _arg in node.args:
                if isinstance(_arg, torch.fx.node.Node):
                    if _arg.op == 'placeholder' and isinstance(node.target, str):
                        the_first_layer = True
                    total_args.append(_arg.name)
            if the_first_layer:
                self.io_module[node.target] = [node]  # 找到首，这倒没啥问题
            
            if node.target in bit8_last_module_names:
                # 在想要保留成8bit的list里的话，则成功保存
                if node.target in self.io_module.keys():
                    # 如果已经创建过键值对了的话,添加新的相关node
                    self.io_module[node.target].append(node)
                else:
                    # 如果还没有创建键值对
                    self.io_module[node.target] = [node]
                # bit8_last_module_names.remove(node.target)  # TODO 这样好像有问题？因为node是可重复的！
            
            
            
            continue
            # 下面写得太冗余了！
            if node.op == 'output':
                for _arg in node.args[0]:  # XXX _arg还会出现多个，根据你模型定义了几个输出
                    if isinstance(_arg, dict):
                        for out in _arg.values():# 遍历一下
                            if isinstance(out, list):
                                for arg_node in out:
                                    if arg_node.target in bit8_last_module_names:
                                        # 在想要保留成8bit的list里的话，则成功保存
                                        # 弹出
                                        self.io_module[arg_node.target] = arg_node
                                        bit8_last_module_names.remove(arg_node.target)
                            elif out.target in bit8_last_module_names:
                                # 在想要保留成8bit的list里的话，则成功保存
                                # 弹出
                                self.io_module[out.target] = out
                                bit8_last_module_names.remove(out.target)
                            else:
                                raise NotImplementedError
                    elif isinstance(_arg, list):
                        pass
                            
                    
                    # if isinstance(_arg, tuple):
                    #     # 说明是更复杂的情况
                    #     print('\n!!find complex output!!!!!接下来取最后一个输出来处理尾量化问题')
                    #     if isinstance(_arg[-1], dict):
                    #         for _value in _arg[-1].values():
                    #             if isinstance(_value, list):
                    #                 for it in _value:
                    #                     self.io_module[it.target] = it
                    #             elif isinstance(_value, torch.fx.node.Node):
                    #                 self.io_module[_value.target] = _value
                    #     else:
                    #         raise NotImplementedError

                    # elif isinstance(_arg, torch.fx.node.Node):
                    #     self.io_module[_arg.target] = _arg  # XXX 准确地来说不应该叫module
                    # else:
                    #     raise NotImplementedError

    def _find_act_quants(self, model: GraphModule) -> List:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []  # TODO 意思是输出量化？
        g2node = getitem2node(model)  # TODO 这是干啥的
        for node in nodes:  # 两个筛选条件，一个是用来确认该node是否是不允许量化，一个用来确认是否满足量化并整理（C or FC）其输入node
            if ((node.op == "call_module" and node.target in self.exclude_module_name) or
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or
                    node.name in self.exclude_node_name) and node.name not in self.additional_node_name:
                logger.info("Exclude skip: {}".format(node.name))
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input) or node.name in self.additional_node_name:  # XXX 是layer且属于module_type_to_quant_input、是函数且属于function_type_to_quant_input、name属于additional_node_name
                input_node_list = self._flatten_args(node.args)  # XXX 将node 的输入node dict reorg成list 可以利用这个
                # Means this is not Tensor + Tensor. 接下来检查，输入是否是存粹的node
                if not all([isinstance(_node, torch.fx.node.Node) for _node in input_node_list]):
                    continue
                for _node in input_node_list:
                    if self._is_implicit_merge(modules, (node, _node)):  # TODO 这个是拿来检验，父子关系是否为mul或add，会被fused？
                        logger.info("Implicit merge: {} + {}".format(_node.name, node.name))
                        continue
                    if _node in g2node:
                        _node = g2node[_node]
                    node_need_to_quantize_output.append(_node)  # XXX 总结来说，就是找到需要前置插入量化节点的node，然后找到他父节点，在所有父节点后面插入act 量化节点
        return node_need_to_quantize_output  # 意思就是Conv 或Linear之前肯定会有act 量化节点

    def _insert_fake_quantize_for_act_quant(self, model: GraphModule, qconfig, specified_general_quantizers, testing=False):  # 在conv前插入
        graph = model.graph
        nodes = list(model.graph.nodes)
        # self.exclude_node_name = ['backbone_fpn_extra_blocks_p7']  # p7的输入不进行量化！！是这个意思
        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)  # 找到那些输出act需要被量化的node TODO 这里有问题
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()

        aq_sign = qconfig.activation.p.keywords.pop('sign')
        aq_bit = qconfig.activation.p.keywords.pop('bit')
        
        # 先造8bit量化的config，因为尾保持8bit量化会需要这个. 8bit 对称 unsign量化
        aqconfig_8bit = copy.deepcopy(qconfig.activation)
        aq_symmetry = True if is_symmetric_quant(qconfig.activation.p.keywords['qscheme']) else False
        aqconfig_8bit.p.keywords['quant_min'] = -2 ** (8 - 1) if (aq_symmetry and aq_sign) else 0
        aqconfig_8bit.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if (aq_symmetry and aq_sign)  else 2 ** 8 - 1
        aqconfig_8bit.p.keywords['dtype'] = torch.qint8 if (aq_symmetry and aq_sign)  else torch.quint8
        # 再造8bit量化的特殊config，因为首保持8bit量化会需要这个. 8bit 对称 sign量化
        aqconfig_8bit_special = copy.deepcopy(qconfig.activation)
        aq_symmetry = True if is_symmetric_quant(qconfig.activation.p.keywords['qscheme']) else False
        aqconfig_8bit_special.p.keywords['quant_min'] = -2 ** (8 - 1) if aq_symmetry else 0
        aqconfig_8bit_special.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if aq_symmetry else 2 ** 8 - 1
        aqconfig_8bit_special.p.keywords['dtype'] = torch.qint8 if aq_symmetry else torch.quint8
        # 再造特殊config，因为一些非ReLU后面的quantizer若symmetric则置为sign。  同bit 对称 sign量化
        aqconfig_special = copy.deepcopy(qconfig.activation)
        aqconfig_special.p.keywords['quant_min'] = -2 ** (aq_bit - 1) if aq_symmetry else 0
        aqconfig_special.p.keywords['quant_max'] = 2 ** (aq_bit - 1) - 1 if aq_symmetry else 2 ** aq_bit - 1
        aqconfig_special.p.keywords['dtype'] = torch.qint8 if aq_symmetry else torch.quint8
        
        module_dict = dict(model.named_modules())
        
        for node in node_to_quantize_output:  # 开始遍历，插入act量化节点
            quantizer_name = node.name + quantizer_prefix
            
            # 检查是否为需要跳过的quantizer。需要跳过。比如backbone输出其实已经被量化过了，那么neck的输入就不需要被量化。
            if quantizer_name in self.removed_quantizer_names:
                logger.info("Remove {} quantizer".format(quantizer_name))
                continue
            
            if node.name in self.post_act_8bit_node_name:  # TODO 尾巴不应该是sign！也应该走下面那一套  都块函数化，然后现在的8bitconfig其实就是special情况
                logger.info("Set {} post act quantize to 8 bit.".format(node.name))  # 确实说明的是，在该node后面加上fakequant用于act
                
                fake_quantizer = self._execute_act_quantizer(node, module_dict, quantizer_name, aqconfig_8bit, aqconfig_8bit_special, specified_general_quantizers)
                
                # # NOTE！因为共享头的输入会被插入多个量化器
                # quantizer_name = node.name + quantizer_prefix
                # logger.info("Insert act quant {}".format(quantizer_name))
                # fake_quantizer = aqconfig_8bit()  # 直接生成量化器  NOTE 这玩意就是量化器，走公式的那种，是layer！
                
                # fake_quantizer.compute_qloss = True  # 置True，表示act的quantizer要计算qloss
                # setattr(model, quantizer_name, fake_quantizer)  # 绑定layer到model中
                # with graph.inserting_after(node):  # XXX 确实是在node后面插入 act fquantizer  但其实这个node就是act或者其他函数
                #     inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})  # 定义node
                #     for _node in nodes:  #  遍历graph，想把原来接着的node的arg重定向到inserted_node上。
                #         _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)
            else:  
                fake_quantizer = self._execute_act_quantizer(node, module_dict, quantizer_name, qconfig.activation, aqconfig_special, specified_general_quantizers)
                
                # fake_quantizer = None
                # if (node.op == 'call_function' or node.op == 'call_method'):
                #     target_module = None
                # else:
                #     target_module = module_dict[node.target]  # TODO  要判断一下是不是call function，然后解决        add、interp处的actquantizer一样的道理，mqbench的逻辑是兼容的
                # quantizer_name = node.name + quantizer_prefix
                # if ('quantizer' not in node._prev.name and  # 这个就能筛大部分的了
                #     (isinstance(target_module, (nn.ReLU, nn.MaxPool2d)) or node.target in (F.relu, F.max_pool2d))):  # 如果说target module是relu、maxpool、那就遵守quantizer
                #     fake_quantizer = qconfig.activation()
                #     logger.info("Insert act quant {} with general config".format(quantizer_name))
                # else:  # 如果为conv、bn、或者op=placeholder，
                #     # 说明不是Relu后面的quantizer，要改为特制的quantizer    这个分支对应了mobilenetv2的情况\add、interp处的情况
                #     fake_quantizer = aqconfig_special()
                #     logger.info("Insert act quant {} with special config".format(quantizer_name))
                    
                # # 二次检查，把非ReLU后的actquantizer搞成若symmetric则sign
                # fake_quantizer.compute_qloss = True  # 置True，表示act的quantizer要计算qloss
                # setattr(model, quantizer_name, fake_quantizer)
                # with graph.inserting_after(node):  # XXX 确实是在node后面插入 act fquantizer  但其实这个node就是act或者其他函数
                #     inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})  # 定义node
                #     for _node in nodes:  #  遍历graph，想把原来接着的node的arg重定向到inserted_node上。
                #         _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)
            
            
            # fake_quantizer.compute_qloss = compute_qloss_flag  # 置False，表示act的quantizer不计算qloss；置true, 表示act的quantizer算qloss
            setattr(model, quantizer_name, fake_quantizer)
            with graph.inserting_after(node):  # XXX 确实是在node后面插入 act fquantizer  但其实这个node就是act或者其他函数
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})  # 定义node
                for _node in nodes:  #  遍历老graph，想把原来接着的node的arg重定向到inserted_node上。  注意是老nodes集合，很妙！
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)
    
            # else:
            #     warn('只量化首尾！！')

        model.recompile()
        model.graph.lint()
        return model, node_to_quantize_output
    def _execute_act_quantizer(self, node, module_dict, quantizer_name, config, special_config, specified_general_quantizers):
        def is_node_names_have_word(in_nodes, word):
            # 这里可能会有问题，因为有些relu node的名字可没有“relu”
            for in_node in in_nodes:
                if word in in_node.name:
                    return True
            return False
        
        fake_quantizer = None
        if (node.op == 'call_function' or node.op == 'call_method' or node.op == 'placeholder'):
            target_module = None
        else:
            target_module = module_dict[node.target]  # TODO  要判断一下是不是call function，然后解决        add、interp处的actquantizer一样的道理，mqbench的逻辑是兼容的
        if (
            (
            quantizer_name in specified_general_quantizers 
                )
            or 
            (
                # not is_node_names_have_word(node.all_input_nodes, 'quantizer')  # 这个就能筛大部分的了  'quantizer' not in node._prev.name
                # and 
                (
                    isinstance(target_module, (qnniqat.ConvReLU2d, qnniqat.ConvBnReLU2d, nn.intrinsic.qat.modules.conv_fused.ConvReLU2d, nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d, torch.nn.intrinsic.qat.modules.linear_relu.LinearReLU, nn.ReLU, nn.ReLU6, nn.MaxPool2d)) 
                    or node.target in (F.relu, F.relu6, F.max_pool2d)
                )
                ) 
            or 
            (
                is_node_names_have_word(node.all_input_nodes, 'relu') 
                and 'flatten' in node.name
                )):  # 如果说target module是relu、maxpool、那就遵守quantizer
            
            fake_quantizer = config()
            logger.info("Insert act quant {} with general config".format(quantizer_name))
        else:  # 如果为conv、bn、或者op=placeholder，
            # 说明不是Relu或relu6后面的quantizer，要改为特制的quantizer    这个分支对应了mobilenetv2的情况\add、interp处的情况
            fake_quantizer = special_config()  # special唯一的意义就是作为对称量化setting
            logger.info("Insert act quant {} with special config".format(quantizer_name))
        
        return fake_quantizer