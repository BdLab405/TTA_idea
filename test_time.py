import os
import torch
import logging
import numpy as np
import methods

from models.model import get_model
from utils.misc import print_memory_info
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq

# 获取logger对象，方便记录运行过程中的日志信息
logger = logging.getLogger(__name__)
# 定义主函数 evaluate，参数 description 通常是命令行传入的配置描述
def evaluate(description):
    # 从命令行或外部传入的参数中加载配置文件，设置 cfg 全局配置
    load_cfg_from_args(description)

    # 定义支持的评估设置类型，用于断言校验
    valid_settings = ["reset_each_shift",           # 每遇到一个新的domain shift就重置模型
                      "continual",                  # 连续domain shift，不重置模型
                      "gradual",                    # 缓慢变化的domain shift（如腐蚀严重程度从1到5）
                      "mixed_domains",              # 样本来自不同域，无固定顺序
                      "correlated",                 # 样本排序按类别
                      "mixed_domains_correlated",   # 混合domain + 类别相关排序
                      "gradual_correlated",         # 缓慢腐蚀 + 类别相关排序
                      "reset_each_shift_correlated" # 类别相关 + 每次都重置模型
                      ]

    # 断言当前设置合法，防止使用未支持的策略
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"
    # 设置使用的设备，优先使用GPU（cuda）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 根据配置中指定的数据集名称，获取类别总数
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    print(num_classes)
    # 获取基础模型（例如 ResNet18）及其输入所需的预处理函数（例如 Normalize）
    base_model, model_preprocess = get_model(cfg, num_classes, device)
    # 将预处理函数绑定到模型上，便于后续调用
    base_model.model_preprocess = model_preprocess
    # 获取所有已注册的 test-time adaptation 方法名称
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    # 校验配置中指定的方法在注册列表中
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    # 根据方法名称实例化相应的适应方法对象
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(
        cfg=cfg, model=base_model, num_classes=num_classes
    )
    # 打印提示：已成功构造 test-time adaptation 方法
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # 根据不同的数据集决定 test-time 使用的“腐蚀类型”或“域”序列
    if cfg.CORRUPTION.DATASET == "domainnet126":
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    # 输出当前正在使用的数据集和其腐蚀或域序列
    logger.info(f"Using {cfg.CORRUPTION.DATASET} with the following domain sequence: {domain_sequence}")

    # 对于 "mixed_domains" 模式，只使用"mixed"作为唯一的domain
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # 如果设置为 gradual（渐变腐蚀），且腐蚀列表中只有一个，则生成完整的腐蚀等级序列
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] \
       and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]  # 渐变腐蚀序列：从轻到重再到轻
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY  # 否则直接使用配置文件给出的腐蚀强度

    # 初始化记录误差的列表
    accs = []      # 所有腐蚀条件下的错误率列表
    accs_5 = []    # 腐蚀强度为5时的错误率
    domain_dict = {}  # 用于保存每个域的详细结果

    # 主评估循环：遍历每一个域（或混合模式）
    for i_dom, domain_name in enumerate(domain_seq_loop):
        # 是否需要重置模型状态
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")  # 如果模型不支持 reset()
        else:
            logger.warning("not resetting model")

        # 对每个腐蚀强度等级进行评估
        for severity in severities:
            # 构建测试数据加载器（会根据当前域和腐蚀级别构建相应数据）
            test_data_loader = get_test_loader(
                setting=cfg.SETTING,
                adaptation=cfg.MODEL.ADAPTATION,
                dataset_name=cfg.CORRUPTION.DATASET,
                preprocess=model_preprocess,
                data_root_dir=cfg.DATA_DIR,
                domain_name=domain_name,
                domain_names_all=domain_sequence,
                severity=severity,
                num_examples=cfg.CORRUPTION.NUM_EX,
                rng_seed=cfg.RNG_SEED,
                use_clip=cfg.MODEL.USE_CLIP,
                n_views=cfg.TEST.N_AUGMENTATIONS,
                delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
            )

            # 第一次打印数据预处理方式
            if i_dom == 0:
                logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")


    #         # 开始评估模型准确率
    #         acc, domain_dict, num_samples = get_accuracy(
    #             model,
    #             data_loader=test_data_loader,
    #             dataset_name=cfg.CORRUPTION.DATASET,
    #             domain_name=domain_name,
    #             setting=cfg.SETTING,
    #             domain_dict=domain_dict,
    #             print_every=cfg.PRINT_EVERY,
    #             device=device
    #         )
    #
    #         # 记录准确率
    #         accs.append(acc)
    #
    #         # 如果腐蚀强度是5（最严重），记录在 accs_5 中
    #         if severity == 5 and domain_name != "none":
    #             accs_5.append(acc)
    #
    #         # 打印该 domain + 腐蚀等级下的准确率
    #         logger.info(
    #             f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}{severity}][#samples={num_samples}]: {acc:.2%}")
    #
    # # 如果记录了 severity=5 的准确率，打印两个平均准确率；否则只打印一个
    # if len(accs_5) > 0:
    #     logger.info(f"mean accuracy: {np.mean(accs):.2%}, mean accuracy at 5: {np.mean(accs_5):.2%}")
    # else:
    #     logger.info(f"mean accuracy: {np.mean(accs):.2%}")

        # 开始评估模型准确率
        acc, domain_dict, num_samples, first_batches_acc, first_batches_reval_acc, last_batches_acc = get_accuracy(
            model,
            data_loader=test_data_loader,
            dataset_name=cfg.CORRUPTION.DATASET,
            domain_name=domain_name,
            setting=cfg.SETTING,
            domain_dict=domain_dict,
            print_every=cfg.PRINT_EVERY,
            device=device
        )

        # 记录准确率
        accs.append(acc)

        # 如果腐蚀强度是5（最严重），记录在 accs_5 中
        if severity == 5 and domain_name != "none":
            accs_5.append(acc)

        # 打印该 domain + 腐蚀等级下的整体准确率
        logger.info(
            f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}{severity}][#samples={num_samples}]: {acc:.2%}"
        )

        # 打印前5个 batch 的准确率（第一次 & 再次评估）
        first_pass_list = " ".join([f"{acc1:.2%}" for acc1 in first_batches_acc])
        reval_list = " ".join([f"{acc2:.2%}" for acc2 in first_batches_reval_acc])

        logger.info(
            f"{cfg.CORRUPTION.DATASET} first pass\taccuracy % "
            f"[{domain_name}{severity}]: {first_pass_list}"
        )
        logger.info(
            f"{cfg.CORRUPTION.DATASET} re-eval   \taccuracy % "
            f"[{domain_name}{severity}]: {reval_list}"
        )

        # 打印最后5个 batch 的准确率
        last_pass_list = " ".join([f"{acc_last:.2%}" for acc_last in last_batches_acc])
        logger.info(
            f"{cfg.CORRUPTION.DATASET} last 5 batches\taccuracy % "
            f"[{domain_name}{severity}]: {last_pass_list}"
        )

    # 打印平均准确率
    if len(accs_5) > 0:
        logger.info(f"mean accuracy: {np.mean(accs):.2%}, mean accuracy at 5: {np.mean(accs_5):.2%}")
    else:
        logger.info(f"mean accuracy: {np.mean(accs):.2%}")


    # 如果是混合domain设置，打印各个domain下的详细评估结果
    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)

    # 如果开启debug选项，打印内存信息
    if cfg.TEST.DEBUG:
        print_memory_info()


# Python脚本入口点，执行主函数 evaluate
if __name__ == '__main__':
    evaluate('"Evaluation.')

