# Generated by Django 3.2.13 on 2022-12-18 23:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0005_auto_20221219_0618'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='category_id',
            new_name='category',
        ),
        migrations.RenameField(
            model_name='product',
            old_name='type_id',
            new_name='type',
        ),
    ]
