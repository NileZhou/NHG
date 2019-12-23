from django.shortcuts import render
from django.views.generic import View
from django.http import JsonResponse, HttpResponse
from django import forms
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.forms.models import model_to_dict

import json
# Create your views here.
# from .pipeline import pipeline
from pipeline import pipeline # 所有的一切，只需要小小的做一个__init__.py

import subprocess


def index(request):

    # content = """5月6日，广州超大城市综合气象观测试验2019年增强观测期启动会召开。来自中国气象局综合观测司、中国气象局气象探测中心、广东省气象局、上海市气象局、广州市科技局、广州市气象局等单位领导和专家共100余人参加启动会。
    # 　　按照中国气象局部署，广东省气象局将开展为期三年的超大城市综合气象观测试验，致力于解决大城市临近预报和环境气象服务中关键性核心技术问题。试验主要内容包括开展城市冠层观测试验；开展雷雨大风高时空分辨率、高覆盖天气雷达观测试验，构建高分辨率、高（全）覆盖天气雷达探测格点数据库；开展综合观测资料的融合性分析与应用和资料同化分析试验，构建覆盖城市的高分辨率实时三维实况场；开展强降水观测试验，提升数值模式对台风强风强降水以及沿海地区强降水的预报能力；开展灰霾观测试验，提高珠三角灰霾与空气质量预报水平。
    # 　　2019年广州超大城市综合气象观测试验，将开展大气综合廓线站观测网建设和进行增强期观测试验。目前已做好了各项准备工作，已选取了广州市局和黄埔区局2个站点开展试验，龙门站作为对比站，已完成了5条垂直廓线观测设备的布设。下来将结合典型天气过程获取温湿、风、水凝物、气溶胶5条廓线数据，通过数据分析将揭示广州超大城市及城市群对气象环境，尤其是对大气边界层的影响；并通过建立观测预报的良性互动机制试验，探索未来观测与预报一体化的业务和业务流程。"""
    #
    #
    #     summary, title = pipeline(content, 'CNN_RNN')
    #
    #     print(os.getcwd())
    #     return HttpResponse(summary+'\n'+title)

    # article_path = '/media/nile/study/repositorys/autosumma/myweb/gentitle/tmp.txt'
    # model_prefix = 'CNN_RNN'
    #
    # res = subprocess.check_output(['./gen.sh', article_path, model_prefix]).decode('utf-8')
    # summary = res[res.index('15036464') + 9: res.index('==============================')]
    # title = res[res.index('==============================') + 30:]
    #
    # return render(request, 'gentitle/index.html', context={'title': title, 'summary': summary})

    return render(request, 'gentitle/index.html', context={'title': '', 'summary': ''})


@csrf_exempt
def gen(request):
    # article_path = '/media/nile/study/repositorys/autosumma/myweb/gentitle/tmp.txt'
    # article = request.POST.get('article')
    # model_prefix = request.POST.get('model_prefix')
    # with open(article_path, 'w', encoding='utf-8') as f:
    #     f.writelines(article)

    # res = subprocess.check_output(['./gen.sh', article_path, model_prefix]).decode('utf-8')
    # summary = res[res.index('==============================') + 30: res.rindex('==============================')]
    # title = res[res.rindex('==============================') + 30:]

    article = request.POST.get('article')
    model_prefix = request.POST.get('model_prefix')
    summary, title = pipeline(article, model_prefix)

    return_data = {'title': title.strip(), 'summary': summary.strip()}
    return HttpResponse(json.dumps(return_data), content_type='application/json')
