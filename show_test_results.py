import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

img = mmcv.imread('/data/liguanlin/Datasets/Visdrone/VisDrone2019-DET-test-dev/images/9999938_00000_d_0000428.jpg') #9999938_00000_d_0000428
#img = mmcv.imread('/data/liguanlin/Datasets/Visdrone/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg')
#img = mmcv.imread('/data/liguanlin/Datasets/Visdrone/VisDrone2019-DET-test-dev/images/0000074_02723_d_0000005.jpg')
# 获取基本配置文件参数

#cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

#cfg.load_from = 'work_dir_custom/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'

#cfg.load_from = 'work_dir_custom/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'
# 构建数据集
#datasets = [build_dataset(cfg.data.train)]

config_file = '../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
checkpoint_file = '../checkpoints/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'

#config_file = 'work_dir_custom/customformat.py'
#checkpoint_file = 'customformat/xxxx.pth'

# 构建检测模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# 添加类别文字属性提高可视化效果

result = inference_detector(model, img)
show_result_pyplot(model, img, result)