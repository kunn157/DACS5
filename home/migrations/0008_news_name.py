# Generated by Django 3.2.13 on 2022-12-28 20:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0007_news'),
    ]

    operations = [
        migrations.AddField(
            model_name='news',
            name='name',
            field=models.CharField(default='', max_length=300),
        ),
    ]