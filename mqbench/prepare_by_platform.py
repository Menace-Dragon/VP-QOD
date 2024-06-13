from copy import deepcopy
from enum import Enum
from typing import Any, Dict
import types
import inspect

import torch
from torch.fx import Tracer
from torch.fx.graph_module import GraphModule
from torch.quantization.quantize_fx import _swap_ff_with_fxff
from torch.quantization import QConfig
import mmcv
import mmdet

from mqbench.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize,
    TqtFakeQuantize,
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
    PureHooker
)
from mqbench.observer import (
    ClipStdObserver,
    LSQObserver,
    MinMaxFloorObserver,
    MinMaxObserver,
    EMAMinMaxObserver,
    PoTModeObserver,
    EMAQuantileObserver,
    MSEObserver,
    EMAMSEObserver,
)
from mqbench.fuser_method_mappings import fuse_custom_config_dict
from mqbench.utils.logger import logger
from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER
from mqbench.scheme import QuantizeScheme

__all__ = ['prepare_by_platform']

class BackendType(Enum):
    Academic = 'Academic'
    Tensorrt = 'Tensorrt'
    SNPE = 'SNPE'
    PPLW8A16 = 'PPLW8A16'
    NNIE = 'NNIE'
    Vitis = 'Vitis'
    ONNX_QNN = 'ONNX_QNN'
    PPLCUDA = 'PPLCUDA'
    OPENVINO = 'OPENVINO'
    Tengine_u8 = "Tengine_u8"
    Tensorrt_NLP = "Tensorrt_NLP"
    Academic_NLP = "Academic_NLP"


ParamsTable = {
    BackendType.Academic:   dict(qtype='affine'),    # noqa: E241
    BackendType.NNIE:       dict(qtype='nnie',       # noqa: E241
                                 # NNIE actually do not need w/a qscheme. We add for initialize observer only.
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=NNIEFakeQuantize,
                                 default_act_quantize=NNIEFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.Tensorrt:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8, symmetric_range=True),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.OPENVINO:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.SNPE:       dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.PPLW8A16:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=16),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.Vitis:      dict(qtype='vitis',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=True, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=True, bit=8),
                                 default_weight_quantize=TqtFakeQuantize,
                                 default_act_quantize=TqtFakeQuantize,
                                 default_weight_observer=MinMaxFloorObserver,
                                 default_act_observer=PoTModeObserver),
    BackendType.ONNX_QNN:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=MinMaxObserver),
    BackendType.PPLCUDA:    dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=MinMaxObserver),
    BackendType.Tengine_u8: dict(qtype="affine",
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
}
ParamsTable[BackendType.Tensorrt_NLP] = ParamsTable[BackendType.Tensorrt]
ParamsTable[BackendType.Academic_NLP] = ParamsTable[BackendType.Academic]

ObserverDict = {  # Obeser 映射到类头
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'EMAMinMaxObserver':        EMAMinMaxObserver,        # More general choice.   # noqa: E241
    'MinMaxFloorObserver':      MinMaxFloorObserver,      # For Vitis HW           # noqa: E241
    'PoTModeObserver':          PoTModeObserver,   # For Vitis HW           # noqa: E241
    'EMAQuantileObserver':      EMAQuantileObserver,      # Quantile observer.     # noqa: E241
    'ClipStdObserver':          ClipStdObserver,          # Usually used for DSQ.  # noqa: E241
    'LSQObserver':              LSQObserver,              # Usually used for LSQ.  # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'EMAMSEObserver':           EMAMSEObserver,                                    # noqa: E241
}

FakeQuantizeDict = {  # 量化器映射到类头
    'FixedFakeQuantize': FixedFakeQuantize,      # Unlearnable scale/zeropoint  # noqa: E241
    'LearnableFakeQuantize': LearnableFakeQuantize,  # Learnable scale/zeropoint    # noqa: E241
    'NNIEFakeQuantize':      NNIEFakeQuantize,       # Quantize function for NNIE   # noqa: E241
    'DoReFaFakeQuantize':    DoReFaFakeQuantize,     # Dorefa                       # noqa: E241
    'DSQFakeQuantize':       DSQFakeQuantize,        # DSQ                          # noqa: E241
    'PACTFakeQuantize':      PACTFakeQuantize,       # PACT                         # noqa: E241
    'TqtFakeQuantize':       TqtFakeQuantize,        # TQT                          # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,   # AdaRound                     # noqa: E241
    'QDropFakeQuantize':     QDropFakeQuantize,      # BRECQ & QDrop                # noqa: E241
    'PureHooker': PureHooker
}


def get_qconfig_by_platform(deploy_backend: BackendType, extra_qparams: Dict):  # 拉取默认的config，再修改成自定义的config
    """

    Args:
        deploy_backend (BackendType):
        extra_qparams (dict):

    >>> extra params format: {
            'w_observer': str, weight observer name,
            'a_observer': str, activation observer name,
            'w_fakequantize': str, weight fake quantize function name,
            'w_fakeq_params": dict, params for weight quantize function,
            'a_fakequantize': str, activation fake quantize function name,
            'a_fakeq_params': dict, params for act quantize function,
            if deploy_backend == BackendType.Academic keys below will be used:
            'w_qscheme': {
                'bit': bitwidth,
                'symmetry': whether quantize scheme is symmetric,
                'per_channel': whether quantize scheme is perchannel,
                'pot_scale': whether scale is power of two.
            }
            'a_qscheme': {
                same with w_qscheme.
            }
        }
    """
    w_observer = extra_qparams.get('w_observer', None)
    if w_observer:
        assert w_observer in ObserverDict, \
            'Do not support observer name: {}'.format(w_observer)
        w_observer = ObserverDict[w_observer]
    a_observer = extra_qparams.get('a_observer', None)
    if a_observer:
        assert a_observer in ObserverDict, \
            'Do not support observer name: {}'.format(a_observer)
        a_observer = ObserverDict[a_observer]
    w_fakequantize = extra_qparams.get('w_fakequantize', None)
    if w_fakequantize:
        assert w_fakequantize in FakeQuantizeDict, \
            'Do not support fakequantize name: {}'.format(w_fakequantize)
        w_fakequantize = FakeQuantizeDict[w_fakequantize]
    a_fakequantize = extra_qparams.get('a_fakequantize', None)
    if a_fakequantize:
        assert a_fakequantize in FakeQuantizeDict, \
            'Do not support fakequantize name: {}'.format(a_fakequantize)
        a_fakequantize = FakeQuantizeDict[a_fakequantize]
    backend_params = ParamsTable[deploy_backend]  # 拉取默认config

    # NNIE backend must use NNIEFakeQuantize but leave observer adjustable.
    if backend_params['qtype'] == 'nnie':
        if not w_observer:
            w_observer = backend_params['default_weight_observer']
        if not a_observer:
            a_observer = backend_params['default_act_observer']
        w_qscheme = backend_params['w_qscheme']
        a_qscheme = backend_params['a_qscheme']
        w_config = backend_params['default_weight_quantize'].with_args(observer=w_observer,
                                                                       **w_qscheme.to_observer_params())
        a_config = backend_params['default_act_quantize'].with_args(observer=a_observer,
                                                                    **a_qscheme.to_observer_params())
        return QConfig(activation=a_config, weight=w_config)

    # Academic setting should specific quant scheme in config.
    if deploy_backend in [BackendType.Academic, BackendType.Academic_NLP]:
        w_qscheme = QuantizeScheme(**extra_qparams['w_qscheme'])  # qscheme就是量化有关参数，bit 对称 POT
        a_qscheme = QuantizeScheme(**extra_qparams['a_qscheme'])
    else:
        w_qscheme = extra_qparams.get('w_qscheme', None)
        if w_qscheme is None:
            w_qscheme = backend_params['w_qscheme']
        else:
            logger.info("Weight Quant Scheme is overrided!")
            w_qscheme = QuantizeScheme(**w_qscheme)
        a_qscheme = extra_qparams.get('a_qscheme', None)
        if a_qscheme is None:
            a_qscheme = backend_params['a_qscheme']
        else:
            logger.info("Activation Quant Scheme is overrided!")
            a_qscheme = QuantizeScheme(**a_qscheme)

    # Set extra args for observers.
    w_observer_extra_args = extra_qparams.get('w_observer_extra_args', {})
    a_observer_extra_args = extra_qparams.get('a_observer_extra_args', {})
    w_qscheme.kwargs.update(w_observer_extra_args)
    a_qscheme.kwargs.update(a_observer_extra_args)
    # Get weight / act fake quantize function and params. And bias fake quantizer if needed(Vitis)
    if not w_fakequantize:
        w_fakequantize = backend_params['default_weight_quantize']
    w_fakeq_params = extra_qparams.get('w_fakeq_params', {})  # TODO 这是干嘛的
    if not a_fakequantize:
        a_fakequantize = backend_params['default_act_quantize']
    a_fakeq_params = extra_qparams.get('a_fakeq_params', {})
    # Get default observer type.
    if not w_observer:
        w_observer = backend_params['default_weight_observer']
    if not a_observer:
        a_observer = backend_params['default_act_observer']

    # Create qconfig.
    # here, rewrited by with_args
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())  # TODO 这是干嘛的
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    logger.info('Weight Qconfig:\n    FakeQuantize: {} Params: {}\n'
                '    Oberver:      {} Params: {}'.format(w_fakequantize.__name__, w_fakeq_params,
                                                         w_observer.__name__, str(w_qscheme)))
    logger.info('Activation Qconfig:\n    FakeQuantize: {} Params: {}\n'
                '    Oberver:      {} Params: {}'.format(a_fakequantize.__name__, a_fakeq_params,
                                                         a_observer.__name__, str(a_qscheme)))
    if backend_params['qtype'] == 'vitis':
        logger.info('Bias Qconfig:\n    TqtFakeQuantize with MinMaxObserver')

    return QConfig(activation=a_qconfig, weight=w_qconfig)  # TODO 这是啥意思


class CustomedTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

def duplicate_reused_nodes(graph: torch.fx.Graph, modules: Dict[str, Any] = {}, not_duplicated_prefixes=[]):  # TODO 这是怎么找出哪个call module是重复的？
    _dup_prefix = '_dup'
    target_dict = dict()  # TODO node.target指的是实际layer/function的名字
    dup_modules = dict()  # 就是新增的复制的layer
    
    for node in graph.nodes:  # XXX 这里就和ONNX gs一摸一样，遍历节点，但是更细粒度，包括一些参数
        if node.op == "call_module":  # call_function就是符合torch规范的函数，如torch.add
            if node.target not in target_dict:  # target_dict其实就是想找出所有的layer
                target_dict[node.target] = [node]  # XXX node名字是唯一的，重复使用的layer会为node名字加后缀。
            else:
                target_dict[node.target].append(node)
    for key in target_dict:
        exclude_flag = False# XXX 定义排除机制
        for exclude_prefix in not_duplicated_prefixes:
            if exclude_prefix in key:
                exclude_flag = True
                break
        if exclude_flag:
            continue  # 直接退出？可取吗
        if len(target_dict[key]) > 1:  # 这里就是看哪个名字重复出现了，那就是说该算子共享了，就需要处理
            for idx, node in enumerate(target_dict[key]):
                if idx == 0:  # 第一个的node就是其本身，原汁原味，不变。
                    continue
                module = deepcopy(modules[node.target])  # 索引出对应的layer，深度拷贝
                node.target += _dup_prefix + str(idx)  # 直接改名字
                dup_modules[node.target] = module
    graph.lint()
    return graph, dup_modules

def prepare_constant_dict(graph: torch.fx.Graph, model: torch.nn.Module):
    def _get_attrs(target, attrs):
        attrs = attrs.split('.')
        for att in attrs:
            target = getattr(target, att)
        return target
    constant_dict = dict()
    for node in graph.nodes:
        if node.op == 'get_attr':
            constant_dict[node.target] = _get_attrs(model, node.target)
    return constant_dict

def eliminate_dead_node(graph: torch.fx.Graph):
    """
        Remove all dead code from the graph, based on each node's number of
        users, and whether the nodes have any side effects. The graph must be
        topologically sorted before calling.
    """
    def is_impure(node):
        """
        从高版本抄过来的，懒得升级torch版本了
        https://pytorch.org/docs/1.9.0/_modules/torch/fx/node.html#Node.is_impure
        https://pytorch.org/docs/1.9.0/_modules/torch/fx/graph.html#Graph.eliminate_dead_code
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Returns:

            bool: If the op is impure or not.
        """
        if node.op in {"placeholder", "output"}:
            return True

        # Check if an impure function.
        # if node.op == "call_function":
        #     return node.target in _side_effectful_functions

        # Check if an impure module.
        # if node.op == "call_module":
        #     assert (
        #         node.graph.owning_module is not None
        #     ), "self.graph.owning_module not set for purity check"
        #     target_mod = node.graph.owning_module.get_submodule(node.target)
        #     assert (
        #         target_mod is not None
        #     ), f"Did not find expected submodule target {node.target}"
        #     return getattr(target_mod, "_is_impure", False)

        return False
    
    
    graph.lint()

    # Reverse iterate so that when we remove a node, any nodes used as an
    # input to that node have an updated user count that no longer reflects
    # the removed node.
    changed = False
    for node in reversed(graph.nodes):
        if not is_impure(node) and len(node.users) == 0:
            graph.erase_node(node)
            logger.info("Erase node {}. ".format(node.name))  # 确实说明的是，在该node后面加上fakequant用于act
            changed = True

    return changed


def prepare_by_platform(
        model: torch.nn.Module,
        deploy_backend: BackendType,
        structure_detail,
        prepare_custom_config_dict: Dict[str, Any] = {},
        custom_tracer: Tracer = None):
    """
    Args:
        model (torch.nn.Module):
        deploy_backend (BackendType):

    >>> prepare_custom_config_dict : {
            extra_qconfig_dict : Dict, Find explanations in get_qconfig_by_platform,
            extra_quantizer_dict: Extra params for quantizer.
            preserve_attr: Dict, Specify attribute of model which should be preserved
                after prepare. Since symbolic_trace only store attributes which is
                in forward. If model.func1 and model.backbone.func2 should be preserved,
                {"": ["func1"], "backbone": ["func2"] } should work.
            Attr below is inherited from Pytorch.
            concrete_args: Specify input for model tracing.
            extra_fuse_dict: Specify extra fusing patterns and functions.
        }

    """
    model_mode = 'Training' if model.training else 'Eval'
    logger.info("Quantize model Scheme: {} Mode: {}".format(deploy_backend, model_mode))

    # XXX Get Qconfig，该阶段只是在收集和整理信息，没有定义实质性的东西，就是写config
    extra_qconfig_dict = prepare_custom_config_dict.get('extra_qconfig_dict', {})
    qconfig = get_qconfig_by_platform(deploy_backend, extra_qconfig_dict)

    _swap_ff_with_fxff(model)  # XXX 替换fx不支持的节点。几乎很少
    # # Preserve attr.  XXX 
    # preserve_attr_dict = dict()
    # if 'preserve_attr' in prepare_custom_config_dict:
    #     for submodule_name in prepare_custom_config_dict['preserve_attr']:
    #         cur_module = model
    #         if submodule_name != "":
    #             cur_module = getattr(model, submodule_name)
    #         # preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
    #         preserve_attr_dict[submodule_name] = cur_module
    #         # for attr in preserve_attr_list:
    #         #     preserve_attr_dict[submodule_name][attr] = getattr(cur_module, attr)
    # Symbolic trace
    concrete_args = structure_detail.input_concrete_args  # XXX trace 的定制
    not_duplicated_prefixes = structure_detail.not_duplicated_prefixes
    customed_leaf_module = prepare_custom_config_dict.get('leaf_module', [])  # XXX trace 的定制 leaf module 就是我们正常定义的层结构，这里是手动说明自定义的层
    customed_leaf_module.append(mmcv.cnn.bricks.swish.Swish)
    customed_leaf_module.append(mmcv.cnn.bricks.activation.Clamp)
    customed_leaf_module.append(mmcv.cnn.bricks.hsigmoid.HSigmoid)
    customed_leaf_module.append(mmcv.cnn.bricks.scale.Scale)
    customed_leaf_module.append(mmdet.models.necks.ssd_neck.L2Norm)
    
    tracer = CustomedTracer(customed_leaf_module=tuple(customed_leaf_module))
    if custom_tracer is not None:
        tracer = custom_tracer
    
    if len(concrete_args) == 1 and 'in_num' in concrete_args:
        model.in_num = concrete_args['in_num']
    elif len(concrete_args) == 0:
        pass
    else:
        raise NotImplementedError
    
    graph = tracer.trace(model)  # XXX graph的node是允许“重复”的，指反复使用一个算子（层）时会反复记录成node。
    name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
    modules = dict(model.named_modules())  # XXX 这个操作直接提取出所有的子layer，及其名字，作为name：layer dict
    # TODO 删除死节点。torch1.9之后才有。此为低版本的滥用
    eliminate_dead_node(graph)
    
    graph, duplicated_modules = duplicate_reused_nodes(graph, modules, not_duplicated_prefixes)  # XXX 意思是，需要把共享的call module复制出来，取消共享。注意，只复制module！在共享head机制下不能复制head！该功能更关注的是串行上的复用问题
    constant_nodes = prepare_constant_dict(graph, model)  # 确实是用来获取constant。一般出自model的attr
    # TODO 下面这两步更新到一起？为啥？
    modules.update(duplicated_modules)  # XXX 确实，必须得复制，因为有的relu是共享的！
    modules.update(constant_nodes)
    
    graph_module = GraphModule(modules, graph, name)  # TODO 这就搞到一起了？？？
    # Model fusion.
    extra_fuse_dict = prepare_custom_config_dict.get('extra_fuse_dict', {})  # XXX 指定哪些需要fuse
    # extra_fuse_dict.update(fuse_custom_config_dict)  # NOTE 加载MQBEnch自定义的fuse，
    # Prepare
    import mqbench.custom_quantizer  # noqa: F401
    extra_quantizer_dict = prepare_custom_config_dict.get('extra_quantizer_dict', {})
    quantizer = DEFAULT_MODEL_QUANTIZER[deploy_backend](extra_quantizer_dict, extra_fuse_dict)
    prepared = quantizer.prepare(graph_module, qconfig, structure_detail.further_detail, testing=prepare_custom_config_dict.get('testing', False))  # 返回已经插入quantizer的模型结构
    # TODO Restore attr.
    if 'preserve_attr' in structure_detail:
        for attr_name in structure_detail['preserve_attr']:
            cur_module = model
            try:
                attr = getattr(model, attr_name)
                
            except AttributeError:
                pass
            else:
                logger.info("Preserve attr: {}".format(attr_name))
                _type = type(model)
                if inspect.ismethod(attr):
                    attr = types.MethodType(getattr(_type, attr_name), prepared)
                
                # preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
                setattr(prepared, attr_name, attr)
                
    # # Restore attr.
    # if 'preserve_attr' in prepare_custom_config_dict:
    #     for submodule_name in prepare_custom_config_dict['preserve_attr']:
    #         cur_module = prepared
    #         _type = type(model)
    #         if submodule_name != "":
    #             cur_module = getattr(prepared, submodule_name)
    #             _type = type(getattr(model, submodule_name))
    #         preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
    #         for attr_name in preserve_attr_list:
    #             logger.info("Preserve attr: {}.{}".format(submodule_name, attr_name))
    #             _attr = preserve_attr_dict[submodule_name][attr_name]
    #             if inspect.ismethod(_attr):
    #                 _attr = types.MethodType(getattr(_type, attr_name), cur_module)
    #             setattr(cur_module, attr_name, _attr)
    return prepared
