from django.contrib import admin
from .models import ArticlePost # 把这个注册到这里，是为了让后台能进行管理

# Register your models here.
# 注册ArticlePost到admin中
admin.site.register(ArticlePost)
