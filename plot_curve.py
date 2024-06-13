import datetime
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import scipy.stats as sci
import pandas as pd
import matplotlib.image as mpimg
import mmcv


def save_distribution(data, title, pedge=1.1, hlim=0.02):
    plt.figure()
    number = data.size
    edge = np.linspace(-pedge, pedge, 160)
    bins = [0] * (len(edge) - 1)
    for i in range(len(edge) - 1):
        total = np.sum(np.array(edge[i] < data) == np.array(data < edge[i + 1]))
        bins[i] = total

    bins = np.array(bins) / number  # 做一个归一化

    plt.stem(edge[:-1], bins, markerfmt='C3.')  # 不能直接用hist，因为hist的区间是动态的！（最小刻度会变，很傻吊）
    plt.title(title)
    plt.xlim(-1.1 * pedge, 1.1 * pedge)
    plt.ylim(0, hlim)
    plt.grid()

    plt.savefig('y_data_fig/{}.jpg'.format(title))
    plt.close()
    
    
def plot_loss_and_lr(train_loss, learning_rate, output_dir):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('{}/loss_and_lr{}.png'.format(output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP, output_dir):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('{}/mAP.png'.format(output_dir))
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)
        
def generate_axis(rows,cols):
    return [(row,col) for row in range(rows) for col in range(cols)]
        
def plot_ious_and_confi_per_class(tp_special_data, classes_list):
    det_labels = tp_special_data[:,0].astype(np.int16)
    num_classes = len(classes_list)
    label_statistics = [[] for i in range(num_classes)]
    
    for it, label in enumerate(det_labels):
        label_statistics[label].append(tp_special_data[it,1:])  # score IOU
    
    plt.figure(figsize=(20,6))
    plt.subplots_adjust(left=0.05,bottom=None,right=0.95,top=None,wspace=.55,hspace=.35) # 设置子图间距

    for it, statistics in enumerate(label_statistics):
        statistics = np.array(statistics)
        statistics[:, 1] = iou_remap(statistics[:, 1])  # IOU数据 重新缩放
        
        plt.subplot(2,10,it+1)
        plt.title(f"{classes_list[it]}")
        plt.boxplot(statistics)
        plt.grid()  # 生成网格
        
    title_font = {'weight': 'bold', 'size': 18}
    plt.suptitle('Score and IOU in TP',fontdict=title_font)

    # plt.show()
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32.png")
    plt.savefig("/workspace/code/Quant/MQBench/test/test_img/w3a3.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/w4a4.png")
    plt.close()

def loop_plot_ious_and_confi_per_class(loop_data, classes_list, x_script):
    differ = 0.6  # 用来偏移画图的
    loop_label_statistics = []
    num_classes = len(classes_list)
    for tp_special_data in loop_data:
        det_labels = tp_special_data[:,0].astype(np.int16)
        label_statistics = [[] for i in range(num_classes)]
    
        for it, label in enumerate(det_labels):
            label_statistics[label].append(tp_special_data[it,1:])  # score IOU
        loop_label_statistics.append(label_statistics)
    
    plt.figure(figsize=(36,15))
    plt.subplots_adjust(left=0.04,bottom=0.10,right=0.96,top=0.90,wspace=.15,hspace=.3) # 设置子图间距

    for it in range(num_classes):  # 画20张子图
        all_position = []
    
        plt.subplot(4,5,it+1)
        plt.title(f"{classes_list[it]}", fontsize=22)
        for idx, label_statistics in enumerate(loop_label_statistics):  # 遍历文件数据
            statistics = label_statistics[it]  # 引出对应类下的 socre数据或IOU数据
            statistics = np.array(statistics)
            statistics[:, 1] = iou_remap(statistics[:, 1])  # IOU数据 重新缩放
            # confidence 或 IOU重新排序，只取前K个结果
            topk = 200
            confi_data = np.sort(statistics[:, 0])[-topk:]
            iou_data = np.sort(statistics[:, 1])[-topk:]
            
            positions = [1+idx*differ, 5.4+idx*differ]
            all_position.extend(positions)
            plt.boxplot([confi_data, iou_data], positions=positions)
            
        # 接下来着手处理x坐标信息
        axis_names = x_script*2
        all_position.sort()
        plt.xticks(ticks=all_position,        #设置要显示的x轴刻度，若指定空列表则去掉x轴刻度
            # , 
            labels=axis_names,#设置x轴刻度显示的文字，要与ticks对应   
            fontsize=12,        #设置刻度字体大小
            rotation=0,        #设置刻度文字旋转角度
            ha='center', va='center',        #刻度文字对齐方式，当rotation_mode为’anchor'时，对齐方式决定了文字旋转的中心。ha也可以写成horizontalalignment，va也可以写成verticalalignment。
        )
        plt.xlabel(f"Score Statistical Results{' '*30}IOU Statistical Results")
        
        plt.grid()  # 生成网格
        
    plt.suptitle('Score and IOU in TP',fontsize=28)

    # plt.show()
    plt.savefig("/workspace/code/Quant/MQBench/test/test_img/BOX-fp32-8-6-5-4-3.png")
    plt.close()

def plot_ious_and_confi(tp_special_data, classes_list, target_dir):
    det_labels = tp_special_data[:,0].astype(np.int16)  # label score IOU
    tp_special_data[:, 2] = iou_remap(tp_special_data[:, 2])  # IOU数据 重新缩放
    
    plt.figure(figsize=(12,4))
    plt.boxplot(tp_special_data[:,1:])
    plt.grid()  # 生成网
    title_font = {'weight': 'bold', 'size': 14}
    plt.title('Score and IOU in TP, summarization',fontdict=title_font)
    # plt.show()
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32_summarization.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/w4a4_summarization.png")
    plt.savefig(f"/workspace/code/Quant/MQBench/test/test_img/{target_dir}/w3a3_summarization.png")
    plt.close()

def statistics_by_percentage_bin(loop_data, classes_list, x_script, target_dir):
    # mode = 'iou' # iou > 0.9 下的讨论  这图也好看
    mode = 'cls' # cls > 0.9 下的讨论  # 这图好看
    fig = plt.figure(figsize=(12,8))
    # plt.subplots_adjust(left=0.08,bottom=0.10,right=0.9,top=0.92,wspace=.2,hspace=.2) # 设置子图间距
    title_font = {'weight': 'bold', 'size': 22}
    label_font = {'weight': 'bold', 'size': 16}
    suptitle_font = {'weight': 'bold', 'size': 40}
    
    loop_number = len(loop_data)
    
    if mode == 'iou':
        bin_number = 10
        edge = np.linspace(0., 1.0, bin_number + 1)
    elif mode == 'cls':
        # bin_number = 6
        # edge = np.array([0., 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        bin_number = 5
        edge = np.linspace(0.5, 1.0, bin_number + 1)
    else:
        raise NotImplementedError
    
    edge = np.round(edge, decimals=3)
    
    bins = np.zeros(bin_number)
    bar_width = 0.16
    mid_position = np.linspace(1, bin_number, bin_number) * 0.75
    base_position = mid_position - bar_width * (loop_number - 1) / 2
    x_show = []
    for i in range(bin_number):
        x_show.append(f'[{edge[i]}, {edge[i+1]}]')
    
    for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
        position = base_position + idx * bar_width
        
        confi_data = tp_special_data[:, 0]
        iou_data = tp_special_data[:, 1]
        if mode == 'iou':
            percent9_data = iou_data
            slicing_data = confi_data
        elif mode == 'cls':
            percent9_data = confi_data
            slicing_data = iou_data
        else:
            raise NotImplementedError
            
        # det_labels = tp_special_data[:,0].astype(np.int16)  # label score IOU
        percentage = 0.9
        focused_percent9_data = percent9_data[percent9_data > percentage]
        focused_slicing_data = slicing_data[percent9_data > percentage]
        elem_number = focused_slicing_data.size
        
        for i in range(bin_number):
            total = np.sum(np.array(edge[i] <= focused_slicing_data) == np.array(focused_slicing_data < edge[i + 1]))
            bins[i] = total / elem_number  # 顺便做一个归一化
            
        plt.bar(position, bins, width=bar_width, label=x_script[idx])
    plt.xticks(mid_position, x_show)   # 替换横坐标x的刻度显示内容
    plt.legend()   # 给出图例
    if mode == 'iou':
        plt.xlabel(f"Classification Score Range", fontdict=label_font)
        plt.ylabel(f"Percentage", fontdict=label_font)
        plt.ylim(0., 0.9)
    elif mode == 'cls':
        plt.xlabel(f"IOU Score Range", fontdict=label_font)
        plt.ylabel(f"Percentage", fontdict=label_font)
        # plt.ylim(0.275, 0.725)
        plt.ylim(0., 0.5)
        
        
    plt.tick_params(labelsize=14)
    # TODO 把百分比标到头上
        
    title_font = {'weight': 'bold', 'size': 14}
    # plt.title('Score and IOU in TP, summarization',fontdict=title_font)
    # plt.show()
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32_summarization.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/w4a4_summarization.png")
    plt.savefig(f"test_img/{target_dir}/statistics_bar_summarization_{mode}0.9.png")
    plt.close()
    
    
    
    # plt.figure()
    # number = data.size
    # edge = np.linspace(-pedge, pedge, 360)
    # bins = [0] * (len(edge) - 1)
    # for i in range(len(edge) - 1):
    #     total = np.sum(np.array(edge[i] < data) == np.array(data < edge[i + 1]))
    #     bins[i] = total

    # bins = np.array(bins) / number  # 做一个归一化

    # plt.stem(edge[:-1], bins, markerfmt='C3.')  # 不能直接用hist，因为hist的区间是动态的！（最小刻度会变，很傻吊）
    # plt.title(title)
    # plt.xlim(-1.1 * pedge, 1.1 * pedge)
    # plt.ylim(0, hlim)
    # plt.grid()

    # plt.savefig('./fig/{}.jpg'.format(title))
    # plt.close()

def statistics_for_iou_bin(loop_data, classes_list, x_script, target_dir):
    fig = plt.figure(figsize=(12,8))
    # plt.subplots_adjust(left=0.08,bottom=0.10,right=0.9,top=0.92,wspace=.2,hspace=.2) # 设置子图间距
    title_font = {'weight': 'bold', 'size': 22}
    label_font = {'weight': 'bold', 'size': 16}
    suptitle_font = {'weight': 'bold', 'size': 40}
    
    loop_number = len(loop_data)
    

    bin_number = 10
    # edge = np.linspace(0.5, 1.0, bin_number + 1)
    edge = np.linspace(0., 1.0, bin_number + 1)
    
    edge = np.round(edge, decimals=3)
    
    bins = np.zeros(bin_number)
    bar_width = 0.16
    mid_position = np.linspace(1, bin_number, bin_number) * 0.75
    base_position = mid_position - bar_width * (loop_number - 1) / 2
    x_show = []
    for i in range(bin_number):
        x_show.append(f'[{edge[i]}, {edge[i+1]}]')
    
    for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
        position = base_position + idx * bar_width
        
        confi_data = tp_special_data[:, 0]
        iou_data = tp_special_data[:, 1]
        
            
        # det_labels = tp_special_data[:,0].astype(np.int16)  # label score IOU
        elem_number = iou_data.size
        
        for i in range(bin_number):
            total = np.sum(np.array(edge[i] <= iou_data) == np.array(iou_data < edge[i + 1]))
            bins[i] = total / elem_number  # 顺便做一个归一化
            # bins[i] = total  # 
            
        plt.bar(position, bins, width=bar_width, label=x_script[idx])
    plt.xticks(mid_position, x_show)   # 替换横坐标x的刻度显示内容
    plt.legend()   # 给出图例
    
    # Total 
    plt.xlabel(f"IOU Score Range", fontdict=label_font)
    plt.ylabel(f"Percentage", fontdict=label_font)
    # plt.ylim(0., 0.6)
        
        
    plt.tick_params(labelsize=14)
    # TODO 把百分比标到头上
        
    title_font = {'weight': 'bold', 'size': 14}
    # plt.title('Score and IOU in TP, summarization',fontdict=title_font)
    # plt.show()
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32_summarization.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/w4a4_summarization.png")
    plt.savefig(f"test_img/{target_dir}/statistics_bar_summarization_all_iou_variation.png")
    plt.close()
    
    
    
    # plt.figure()
    # number = data.size
    # edge = np.linspace(-pedge, pedge, 360)
    # bins = [0] * (len(edge) - 1)
    # for i in range(len(edge) - 1):
    #     total = np.sum(np.array(edge[i] < data) == np.array(data < edge[i + 1]))
    #     bins[i] = total

    # bins = np.array(bins) / number  # 做一个归一化

    # plt.stem(edge[:-1], bins, markerfmt='C3.')  # 不能直接用hist，因为hist的区间是动态的！（最小刻度会变，很傻吊）
    # plt.title(title)
    # plt.xlim(-1.1 * pedge, 1.1 * pedge)
    # plt.ylim(0, hlim)
    # plt.grid()

    # plt.savefig('./fig/{}.jpg'.format(title))
    # plt.close()

def loop_box_ious_and_confi(loop_data, classes_list, x_script, target_dir):
    all_position = []
    differ = 0.6  # 用来偏移画图的
    plt.figure(figsize=(12,4))
    for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
    
        # det_labels = tp_special_data[:,0].astype(np.int16)  # label score IOU
        # tp_special_data[:, 2] = iou_remap(tp_special_data[:, 2])  # IOU数据 重新缩放
        # confidence 或 IOU重新排序，只取前K个结果
        if target_dir =='Retina50':
            topk = round(0.5 * tp_special_data.shape[0])  # 妙！这才对啊
        elif target_dir =='SSD300': 
            topk = round(0.5 * tp_special_data.shape[0])  # 妙！这才对啊
        sorted_idxs = np.argsort(tp_special_data[:, 2])
        tp_special_data = tp_special_data[sorted_idxs]  # 进行从小到大排序
        confi_data = tp_special_data[-topk:][:, 1]
        iou_data = tp_special_data[-topk:][:, 2]
        
        # iou_data = np.sort(tp_special_data[:, 2])
        
        positions = [1+idx*differ, 5+idx*differ]
        all_position.extend(positions)
        plt.boxplot([confi_data, iou_data], positions=positions, sym='')  # NOTE 注意，隐藏了离群点
    # 接下来着手处理x坐标信息
    axis_names = x_script*2
    all_position.sort()
    plt.xticks(ticks=all_position,        #设置要显示的x轴刻度，若指定空列表则去掉x轴刻度
        # , 
        labels=axis_names,#设置x轴刻度显示的文字，要与ticks对应   
        fontsize=10,        #设置刻度字体大小
        rotation=0,        #设置刻度文字旋转角度
        ha='center', va='center',        #刻度文字对齐方式，当rotation_mode为’anchor'时，对齐方式决定了文字旋转的中心。ha也可以写成horizontalalignment，va也可以写成verticalalignment。
    )  
    
    plt.xlabel(f"Score Statistical Results{' '*70}IOU Statistical Results")
    
    plt.ylim([0.1, 1.])
    
    plt.grid()  # 生成网
    
    title_font = {'weight': 'bold', 'size': 12}
    plt.title('Score and IOU in TP, summarization',fontdict=title_font)
    # plt.show()
    plt.savefig(f"/workspace/code/Quant/MQBench/test_img/{target_dir}/BOX-fp32-8-6-5-4-3-Summarization.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32-8-6-5-4-3-Summarization1.png")
    plt.close()

def loop_plot_ious_and_confi(loop_data, classes_list, x_script, target_dir):
    all_position = []
    differ = 0.6  # 用来偏移画图的
    plt.figure(figsize=(12,4))
    count_list = []
    for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
        count_list.append(len(tp_special_data))
        # # det_labels = tp_special_data[:,0].astype(np.int16)  # label score IOU
        # tp_special_data[:, 2] = iou_remap(tp_special_data[:, 2])  # IOU数据 重新缩放
        # # confidence 或 IOU重新排序，只取前K个结果
        # topk = 10000
        # confi_data = np.sort(tp_special_data[:, 1])[-topk:]
        # iou_data = np.sort(tp_special_data[:, 2])[-topk:]
        
        positions = [1+idx*differ]
        all_position.extend(positions)
    plt.plot(x_script, count_list, )
    # 接下来着手处理x坐标信息
    axis_names = x_script
    all_position.sort()
    plt.xticks(
        # ticks=all_position,        #设置要显示的x轴刻度，若指定空列表则去掉x轴刻度
        # # , 
        # labels=axis_names,#设置x轴刻度显示的文字，要与ticks对应   
        fontsize=10,        #设置刻度字体大小
        rotation=0,        #设置刻度文字旋转角度
        ha='center', va='center',        #刻度文字对齐方式，当rotation_mode为’anchor'时，对齐方式决定了文字旋转的中心。ha也可以写成horizontalalignment，va也可以写成verticalalignment。
    )  
    plt.grid()  # 生成网
    plt.legend('TP')
    title_font = {'weight': 'bold', 'size': 12}
    plt.title('Score and IOU in TP, summarization',fontdict=title_font)
    # plt.show()
    plt.savefig(f"/workspace/code/Quant/MQBench/test/test_img/{target_dir}/PLOT-fp32-8-6-5-4-3-Summarization.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32-8-6-5-4-3-Summarization1.png")
    plt.close()


def loop_hexbin_ious_and_confi(loop_data, classes_list, x_script, target_dir):
    fig = plt.figure(figsize=(20,18))
    plt.subplots_adjust(left=0.08,bottom=0.10,right=0.9,top=0.88,wspace=0,hspace=.32) # 设置子图间距
    title_font = {'weight': 'bold', 'size': 22}
    label_font = {'weight': 'bold', 'size': 20}
    suptitle_font = {'weight': 'bold', 'size': 40}
    
    # 定制一个y=x曲线
    ideal_dot = np.linspace(.5, 1.0, 20)
    
    for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
        ax = fig.add_subplot(2,2,idx+1)
        plt.title(f"{x_script[idx]}", fontdict=title_font)
        
        # # 按照类别去绘画
        # tp_special_data = tp_special_data[tp_special_data[:,0] == 0]
        
        # # 按照IOU重新排序，只取前K个结果
        # topk = 10000
        # iou_data = tp_special_data[:, 2]
        # index = np.lexsort((iou_data, ))
        # tp_special_data = tp_special_data[index][-topk:]
        # confi_data = tp_special_data[:, 1]
        # iou_data = tp_special_data[:, 2]
        
        confi_data = tp_special_data[:, 0]
        iou_data = tp_special_data[:, 1]
        
        # df = pd.DataFrame(tp_special_data[:, 1:], columns = ['Classification Score','IOU Score'])
        # sns.jointplot(x=df['Classification Score'], y=df['IOU Score'], # 设置xy轴，显示columns名称
        #       data = df,  #设置数据
        #       color = 'b', #设置颜色
        #     #   s = 50, edgecolor = 'w', linewidth = 1,#设置散点大小、边缘颜色及宽度(只针对scatter)
        #     #   stat_func=sci.pearsonr,
        #       kind = 'hex',#设置类型：'scatter','reg','resid','kde','hex'
        #       #stat_func=<function pearsonr>,
        #       space = 0.1, #设置散点图和布局图的间距
        #     #   size = 4, #图表大小(自动调整为正方形))
        #       ratio = 5, #散点图与布局图高度比，整型
        #     #   marginal_kws = dict(bins=15, rug =True), #设置柱状图箱数，是否设置rug
        #       )
        # df.plot.hexbin(x='Classification Score', y='IOU Score', gridsize=30)
        # plt.scatter(confi_data, iou_data, s=1)  # 注意，x轴是classification，y轴是iou
        # hb = plt.hexbin(confi_data, iou_data, gridsize=40, cmap='Blues', vmin=0, vmax=4)  # min max指的是color的上下限
        hb = plt.hexbin(confi_data, iou_data, gridsize=40, cmap='Blues', vmin=0, vmax=10)  # min max指的是color的上下限
        plt.plot(ideal_dot, ideal_dot, 'r:', linewidth=5)
        # cb = plt.colorbar(hb)  
        # cb.set_label(z)
        plt.xlabel(f"Classification Score", fontdict=label_font)
        plt.ylabel(f"IOU Score", fontdict=label_font)
        plt.tick_params(labelsize=14)
        # plt.xlim([0.1, 1.0])
        # plt.ylim([0.65, 1.0])
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    # # 做成一张主图
    # fig, axarr = plt.subplots(2, 3, figsize=(25, 16))
    # subaxis = generate_axis(2,3)
    # for it, ax_idx in enumerate(subaxis):
    #     axarr[ax_idx].imshow(mpimg.imread(f"/workspace/code/Quant/MQBench/test/test_img/hex_temp/{it}.png"))
    
    # # 去掉 x和 y轴
    # [ax.set_axis_off() for ax in axarr.ravel()]
    # plt.suptitle('Score and IOU in TP, summarization', fontsize=22)
    # plt.tight_layout()
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/HEXSCATTER-fp32-8-6-5-4-3-Summarization.png")
    position = fig.add_axes([0.92, 0.12, 0.015, .78 ])#位置[左,下,右,上]
    cb = fig.colorbar(hb, cax=position)
    # cb.ax.set_yticklabels(['0','5','10','15','20','25','30', '35'], fontsize=18)
    cb.ax.set_yticklabels(['0', '', '1', '', '2', '', '3', '', '≥4'], fontsize=24)
    plt.suptitle('Score and IOU in TP+FP, summarization', fontsize=40)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(f"test_img/{target_dir}/HEXBIN-fp32-8-6-5-4-3-Summarization.png")
    # plt.savefig("test_img/fp32-8-6-5-4-3-Summarization1.png")
    plt.close()

def loop_dot_ious_and_confi(loop_data, classes_list, x_script, target_dir):
    fig = plt.figure(figsize=(24,16))
    plt.subplots_adjust(left=0.08,bottom=0.10,right=0.9,top=0.92,wspace=.2,hspace=.2) # 设置子图间距
    title_font = {'weight': 'bold', 'size': 22}
    label_font = {'weight': 'bold', 'size': 20}
    suptitle_font = {'weight': 'bold', 'size': 40}
    
    # 定制一个y=x曲线
    ideal_dot = np.linspace(.5, 1.0, 20)
    
    for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
        ax = fig.add_subplot(2,3,idx+1)
        plt.title(f"{x_script[idx]}", fontdict=title_font)
        
        # # 按照类别去绘画
        # tp_special_data = tp_special_data[tp_special_data[:,0] == 0]
        
        # # 按照IOU重新排序，只取前K个结果
        # topk = 10000
        # iou_data = tp_special_data[:, 2]
        # index = np.lexsort((iou_data, ))
        # tp_special_data = tp_special_data[index][-topk:]
        # confi_data = tp_special_data[:, 1]
        # iou_data = tp_special_data[:, 2]
        
        confi_data = tp_special_data[:, 0]
        iou_data = tp_special_data[:, 1]
        
        # df = pd.DataFrame(tp_special_data[:, 1:], columns = ['Classification Score','IOU Score'])
        # sns.jointplot(x=df['Classification Score'], y=df['IOU Score'], # 设置xy轴，显示columns名称
        #       data = df,  #设置数据
        #       color = 'b', #设置颜色
        #     #   s = 50, edgecolor = 'w', linewidth = 1,#设置散点大小、边缘颜色及宽度(只针对scatter)
        #     #   stat_func=sci.pearsonr,
        #       kind = 'hex',#设置类型：'scatter','reg','resid','kde','hex'
        #       #stat_func=<function pearsonr>,
        #       space = 0.1, #设置散点图和布局图的间距
        #     #   size = 4, #图表大小(自动调整为正方形))
        #       ratio = 5, #散点图与布局图高度比，整型
        #     #   marginal_kws = dict(bins=15, rug =True), #设置柱状图箱数，是否设置rug
        #       )
        # df.plot.hexbin(x='Classification Score', y='IOU Score', gridsize=30)
        # plt.scatter(confi_data, iou_data, s=1)  # 注意，x轴是classification，y轴是iou
        plt.plot(confi_data, iou_data, 'b.', linewidth=1)  # min max指的是color的上下限
        plt.plot(ideal_dot, ideal_dot, 'r:', linewidth=4)
        # cb = plt.colorbar(hb)  
        # cb.set_label(z)
        plt.xlabel(f"Classification Score", fontdict=label_font)
        plt.ylabel(f"IOU Score", fontdict=label_font)
        plt.tick_params(labelsize=14)
        # plt.xlim([0.1, 1.0])
        # plt.ylim([0.65, 1.0])
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    # # 做成一张主图
    # fig, axarr = plt.subplots(2, 3, figsize=(25, 16))
    # subaxis = generate_axis(2,3)
    # for it, ax_idx in enumerate(subaxis):
    #     axarr[ax_idx].imshow(mpimg.imread(f"/workspace/code/Quant/MQBench/test/test_img/hex_temp/{it}.png"))
    
    # # 去掉 x和 y轴
    # [ax.set_axis_off() for ax in axarr.ravel()]
    # plt.suptitle('Score and IOU in TP, summarization', fontsize=22)
    # plt.tight_layout()
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/HEXSCATTER-fp32-8-6-5-4-3-Summarization.png")

    # cb.ax.set_yticklabels(['0','5','10','15','20','25','30', '35'], fontsize=18)
    plt.suptitle('Score and IOU in TP, summarization', fontsize=40)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(f"test_img/{target_dir}/dot-fp32-8-6-5-4-3-Summarization.png")
    # plt.savefig("test_img/fp32-8-6-5-4-3-Summarization1.png")
    plt.close()

# def loop_violinplot_ious_and_confi(loop_data, classes_list, x_script, target_dir):  # TODO 改成分开绘图吧，合在一起其实不方便？或者着重解决双坐标轴的情况
#     all_position = []
#     differ = 0.6  # 用来偏移画图的
#     plt.figure(figsize=(12,4))
#     for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
    
        
#         # IOU数据 重新缩放
#         # tp_special_data[:, 2] = iou_remap(tp_special_data[:, 2])
        
#         # confidence 或 IOU重新排序，只取前K个结果
#         if target_dir =='Retina50':
#             topk = round(0.5 * tp_special_data.shape[0])  # 妙！这才对啊
#         elif target_dir =='SSD300': 
#             topk = round(0.5 * tp_special_data.shape[0])
#         sorted_idxs = np.argsort(tp_special_data[:, 2])
#         tp_special_data = tp_special_data[sorted_idxs]  # 进行从小到大排序
#         confi_data = tp_special_data[-topk:][:, 1]
#         iou_data = tp_special_data[-topk:][:, 2]
        
#         positions = [1+idx*differ, 5+idx*differ]
#         all_position.extend(positions)
#         plt.violinplot([confi_data, iou_data], positions=positions)
#     # 接下来着手处理x坐标信息
#     axis_names = x_script*2
#     all_position.sort()
#     plt.xticks(ticks=all_position,        #设置要显示的x轴刻度，若指定空列表则去掉x轴刻度
#         # , 
#         labels=axis_names,#设置x轴刻度显示的文字，要与ticks对应   
#         fontsize=10,        #设置刻度字体大小
#         rotation=0,        #设置刻度文字旋转角度
#         ha='center', va='center',        #刻度文字对齐方式，当rotation_mode为’anchor'时，对齐方式决定了文字旋转的中心。ha也可以写成horizontalalignment，va也可以写成verticalalignment。
#     )  
    
#     plt.xlabel(f"Classification Statistical Results{' '*64}IOU Statistical Results{' '*8}")
#     plt.ylim([0.1, 1.])
#     # 构建右侧坐标
#     plt.grid()  # 生成网
    
#     title_font = {'weight': 'bold', 'size': 12}
#     plt.title('Score and IOU in TP, summarization',fontdict=title_font)
#     # plt.show()
#     plt.savefig(f"/workspace/code/Quant/MQBench/test_img/{target_dir}/VIOLIN-fp32-8-6-5-4-3-Summarization.png")
#     # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32-8-6-5-4-3-Summarization1.png")
#     plt.close()


def loop_violinplot_ious_and_confi(loop_data, classes_list, x_script, target_dir):  # TODO 改成分开绘图吧，合在一起其实不方便？或者着重解决双坐标轴的情况
    all_position = []
    differ = 0.6  # 用来偏移画图的
    plt.figure(figsize=(12,4))
    for idx, tp_special_data in enumerate(loop_data):  # 循环绘制每一个数据文件
    
        
        # IOU数据 重新缩放
        # tp_special_data[:, 2] = iou_remap(tp_special_data[:, 2])
        
        # sorted_idxs = np.argsort(tp_special_data[:, 2])
        # tp_special_data = tp_special_data[sorted_idxs]  # 进行从小到大排序
        # confi_data = tp_special_data[-topk:][:, 1]
        # iou_data = tp_special_data[-topk:][:, 2]
        confi_data = tp_special_data[:, 0]
        iou_data = tp_special_data[:, 1]
        
        positions = [1+idx*differ, 5+idx*differ]
        all_position.extend(positions)
        plt.violinplot([confi_data, iou_data], positions=positions)
    # 接下来着手处理x坐标信息
    axis_names = x_script*2
    all_position.sort()
    plt.xticks(ticks=all_position,        #设置要显示的x轴刻度，若指定空列表则去掉x轴刻度
        # , 
        labels=axis_names,#设置x轴刻度显示的文字，要与ticks对应   
        fontsize=10,        #设置刻度字体大小
        rotation=0,        #设置刻度文字旋转角度
        ha='center', va='center',        #刻度文字对齐方式，当rotation_mode为’anchor'时，对齐方式决定了文字旋转的中心。ha也可以写成horizontalalignment，va也可以写成verticalalignment。
    )  
    
    plt.xlabel(f"Classification Statistical Results{' '*64}IOU Statistical Results{' '*8}")
    plt.ylim([0.1, 1.])
    # 构建右侧坐标
    plt.grid()  # 生成网
    
    title_font = {'weight': 'bold', 'size': 12}
    plt.title('Score and IOU in TP, summarization',fontdict=title_font)
    # plt.show()
    plt.savefig(f"test_img/{target_dir}/VIOLIN-fp32-8-6-5-4-3-Summarization.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32-8-6-5-4-3-Summarization1.png")
    plt.close()


def loop_bar_fp(loop_data, classes_list, x_script, target_dir):
    # 绘制FP中background和cls型的占比
    all_position = []
    differ = 0.6  # 用来偏移画图的
    plt.figure(figsize=(12,4))
    for idx, fp_data in enumerate(loop_data):  # 循环绘制每一个数据文件
    
        # det_labels = tp_special_data[:,0].astype(np.int16)  # label score IOU
        # cls_fp = fp_data[0]
        # cls_num = len(cls_fp)
        background_fp1 = fp_data[1]  # cls正确的
        background1_num = len(background_fp1)
        background_fp_total = fp_data[2]  # 总的
        background_total_num = len(background_fp_total)
        background1_percentage = (background_total_num - background1_num) / background_total_num  # 正确的比例
        background2_percentage = 1 - background1_percentage  # 纯背景的比例
        
        positions = [1+idx*differ]
        all_position.extend(positions)
        plt.bar(positions, [background1_percentage], color="b", width=0.4, )
        plt.bar(positions, [background2_percentage], color="g", bottom=[background1_percentage], width=0.4, )
    # 接下来着手处理x坐标信息
    axis_names = x_script
    all_position.sort()
    plt.xticks(ticks=all_position,        #设置要显示的x轴刻度，若指定空列表则去掉x轴刻度
        # , 
        labels=axis_names,#设置x轴刻度显示的文字，要与ticks对应   
        fontsize=10,        #设置刻度字体大小
        rotation=0,        #设置刻度文字旋转角度
        ha='center', va='center',        #刻度文字对齐方式，当rotation_mode为’anchor'时，对齐方式决定了文字旋转的中心。ha也可以写成horizontalalignment，va也可以写成verticalalignment。
    )  
    
    plt.xlabel(f"Statistical Results in FP")
    plt.ylabel(f"Number")
    plt.legend(('cls right','real background'))
    plt.grid()  # 生成网
    
    title_font = {'weight': 'bold', 'size': 12}
    plt.title('FP, summarization',fontdict=title_font)
    # plt.show()
    plt.savefig(f"/workspace/code/Quant/MQBench/test/test_img/{target_dir}/BAR-fp32-8-6-5-4-3-FP-Summarization.png")
    # plt.savefig("/workspace/code/Quant/MQBench/test/test_img/fp32-8-6-5-4-3-Summarization1.png")
    plt.close()

def maxminnorm(array):  # 全局归一化
    max_value=array.max()
    min_value=array.min()
    return (array - min_value)/(max_value-min_value)

def iou_remap(array):  # 重新缩放iou数据的区间[0.5,1]->[0,1]
    return 2*(array - 0.5) 

if __name__ in "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    
    # plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    # # plt.rcParams['font.family'] = 'simhei'
    # plt.rcParams['axes.unicode_minus'] = False
    
    # 加载class json 文件
    # json_file = '/workspace/code/Quant/MQBench/application/pytorch_object_detection/retinaNet/pascal_voc_classes.json'
    # assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    # with open(json_file, 'r') as f:
    #     class_dict = json.load(f)
    #     classes_list = list(class_dict.keys())
    classes_list = None

    # analytic_type = 'FP'  # TP/FP
    # target = 'SSDLite_voc'  # SSD300
    target = 'Retina50_voc'  # SSD300
    # target = 'Retina18_voc'  # SSD300
    total_fp_path = None
    total_tp_path =None
    total_pos_sample_path = None
    
    if target == 'Retina50_voc':
        total_tp_path = [
                    'work_dirs/retinanet_r50_fpn_1x_voc/tp_special_list.npy',
                    'work_dirs/retinanet_r50_fpn_voc_w4a4_LSQ/tp_special_list.npy',
                    'work_dirs/retinanet_r50_fpn_voc_w2a2_LSQ_le/tp_special_list.npy',
        ]
        # total_pos_sample_path = [
        #             'work_dirs/retinanet_r18_fpn_1x_voc/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/all_pos_sample_data.npy',
        # ]
        # total_fp_path = [
        #             'work_dirs/retinanet_r18_fpn_1x_voc/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/all_pos_sample_data.npy',
        # ]
        
        
        # x_script = ['fp32']
        x_script = ['fp32', 'LSQ w4a4', 'LSQ w2a2']    
    elif target == 'Retina18_voc':
        total_tp_path = [
                    'work_dirs/retinanet_r18_fpn_1x_voc/tp_special_list.npy',
                    'work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ/tp_special_list.npy',
                    'work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/tp_special_list.npy',
        ]
        # total_pos_sample_path = [
        #             'work_dirs/retinanet_r18_fpn_1x_voc/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/all_pos_sample_data.npy',
        # ]
        # total_fp_path = [
        #             'work_dirs/retinanet_r18_fpn_1x_voc/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w4a4_LSQ/all_pos_sample_data.npy',
        #             'work_dirs/retinanet_r18_fpn_voc_w2a2_LSQ/all_pos_sample_data.npy',
        # ]
        
        
        # x_script = ['fp32']
        x_script = ['fp32', 'LSQ w4a4', 'LSQ w2a2']
    elif target == 'SSDLite_voc':
        # SSD300
        
        # total_fp_path = [
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/all_pos_sample_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4/all_pos_sample_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp/all_pos_sample_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp_qloss/all_pos_sample_data.npy',  # 就是纯eval的时候有问题
        # ]    
        # total_tp_path = [
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/tp_special_list.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4/tp_special_list.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp_qloss/tp_special_list.npy',  # 就是纯eval的时候有问题
        # ]          
        # total_fp_path = [
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/cls_wrong_fp_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4/cls_wrong_fp_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp_qloss/cls_wrong_fp_data.npy',  # 就是纯eval的时候有问题
        # ]          

        
        # 只用在statistica for iou
        total_tp_path = [
            'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/tp_special_list.npy',  # 就是纯eval的时候有问题
            'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4/tp_special_list.npy',  # 就是纯eval的时候有问题
            'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp/tp_special_list.npy',  # 就是纯eval的时候有问题
            'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp_qloss/tp_special_list.npy',  # 就是纯eval的时候有问题
        ]          
        # total_fp_path = [
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc/cls_wrong_fp_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4/cls_wrong_fp_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp/cls_wrong_fp_data.npy',  # 就是纯eval的时候有问题
        #     'work_dirs/ssdlite_mobilenetv2_scratch_600e_voc_quant_w4a4_mypro_temp_qloss/cls_wrong_fp_data.npy',  # 就是纯eval的时候有问题
        # ]        
        
        x_script = ['fp32', 'w4a4', 'w4a4_HQOD_noq', 'w4a4_HQOD']
        # x_script = ['fp32', 'w4a4', 'w4a4_HQOD']
        # x_script = ['w4a4', 'w4a4_HQOD']
        # x_script = ['fp32', '4Bit', '4Bit_HarDet', '4Bit_HQOD']
        
    else:
        raise NotImplementedError
    
    mmcv.mkdir_or_exist(f'test_img/{target}')
    
    data_list = []
    # for path in total_tp_path:
    #     data = np.load(path, allow_pickle=True)
    #     data_list.append(data)
    if None not in [total_tp_path, total_fp_path]:
        print('\n HexBin tp+fp Mode!!!!\n ')
        for tp_path, fp_path in zip(total_tp_path, total_fp_path):
            tp_data = np.load(tp_path, allow_pickle=True)
            fp_data = np.load(fp_path, allow_pickle=True)
            data = np.concatenate([tp_data[:,1:], 
                                   fp_data, # cls错误型FP
                                #    np.array(fp_data[1]), # 分类正确但IOU小于阈值型FP
                                   ])
            data_list.append(data)
    elif total_tp_path is not None:
        print('\n tp Mode!!!!\n ')
        for path in total_tp_path:
            data = np.load(path, allow_pickle=True)
            pccs = np.corrcoef(data[:, 1], data[:, 2])
            print(f'{pccs[0][1]}\n')
            data_list.append(data[:, 1:])
            # data_list.append(data)
    elif total_fp_path is not None:
        print('\n fp Mode!!!!\n ')
        for path in total_fp_path:
            data = np.load(path, allow_pickle=True)
            data_list.append(data)
    elif total_pos_sample_path is not None:
        print('\n pos sample Mode!!!!\n ')
        for path in total_pos_sample_path:
            data = np.load(path, allow_pickle=True)
            data_list.append(data)
        
    # plot_ious_and_confi_per_class(tp_special_data, classes_list)
    # plot_ious_and_confi(tp_special_data, classes_list)
    # analytic_type == 'TP':
    # loop_plot_ious_and_confi_per_class(data_list, classes_list, x_script, target)
    
    # loop_plot_ious_and_confi(data_list, classes_list, x_script, target)
    
    # loop_box_ious_and_confi(data_list, classes_list, x_script, target)
    # loop_violinplot_ious_and_confi(data_list, classes_list, x_script, target)  # 这个不行
    
    
    loop_hexbin_ious_and_confi(data_list, classes_list, x_script, target)  # 这个稍微行，可以只展示lsq和HQOD的.
    # loop_dot_ious_and_confi(data_list, classes_list, x_script, target)  # 这个没必要了
    # statistics_by_percentage_bin(data_list, classes_list, x_script, target)  # 这个不行
    # statistics_for_iou_bin(data_list, classes_list, x_script, target)  # 
    
        
        
    # analytic_type == 'FP':
        # loop_bar_fp(data_list, classes_list, x_script, target)
    
    # -----------log
    '''
    1. 其实还有一个点，因为bit是统一w和a的，如果w8a16；w8a32又会怎样呢？
    2. 讨论的是confi还是classification socre？在RetinaNet中虽然没区别
    3. hexbin可以理解为热力图的一种.colorbar其实代表了下限，其实实际是超过140的
    
    待解决：假BiDet有猫腻，根本不对劲；但是明白他想表达的意思了，即关注nms后存在的问题，就是nms后仍有很多框，而其中的大多数框都是
    会被判定为FP——因为IOU达不到阈值。我们其实应该关注TP和分类正确的bg型FP   搞清楚FN是怎么来的？！
    
    已定制：
        a. VIOLIN，观察恶化情况；可以看到分类结果先恶化；
        b. HEXBIN，观察imbalance&disharmonious；可以看到一开始就存在不和谐现象；int8加剧了这个现象；因为mAP的计算机制，所以这影响了mAP的结果。
    '''