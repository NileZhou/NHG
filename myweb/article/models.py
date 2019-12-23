"""
   每当对数据库进行了更改（添加、修改、删除等）操作，都需要进行数据迁移。
"""
from django.db import models
# 导入内建的User模型
from django.contrib.auth.models import User
# timezone用来处理时间相关的事务
from django.utils import timezone
# Create your models here.


class ArticlePost(models.Model):
    # 每一个Field都是一个字段
    author = models.ForeignKey(User, on_delete=models.CASCADE) # 外键是用来解决 一对多 的  ,比如一个作者可能有多篇文章
    title = models.CharField(max_length=100)
    body = models.TextField()
    created = models.DateTimeField(default=timezone.now)
    updated = models.DateTimeField(auto_now=True) # 自动更新为当前的时间

    # 这是内部类
    # 元数据是 任何不是字段的东西.eg: 排序选项ordering，数据库表名db_table，单数和复数名称verbose_name和verbose_name_plural
    class Meta:
        ordering = ('-created', ) # 每当取文章出来做一个列表时，按照-created(文章创建时间，负号代表倒序，保证最新文章在顶部)

    def __str__(self):
        return self.title


