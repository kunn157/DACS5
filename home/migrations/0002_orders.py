# Generated by Django 3.2.13 on 2022-12-18 22:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Orders',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name_customer', models.CharField(max_length=100)),
                ('phone_customer', models.CharField(max_length=10)),
                ('add_customer', models.CharField(max_length=200)),
                ('addDetail_customer', models.CharField(max_length=200)),
                ('total_order', models.IntegerField()),
                ('message', models.CharField(max_length=300, null=True)),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(auto_now=True)),
                ('payment', models.CharField(max_length=50)),
            ],
        ),
    ]
