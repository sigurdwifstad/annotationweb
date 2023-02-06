# Generated by Django 2.2.28 on 2023-02-06 11:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('annotationweb', '0002_auto_20230131_1353'),
    ]

    operations = [
        migrations.AlterField(
            model_name='task',
            name='network_config_path',
            field=models.CharField(blank=True, default='', help_text='Path to config file defining neural network parameters (Auto segmentation task only)', max_length=1000),
        ),
        migrations.AlterField(
            model_name='task',
            name='type',
            field=models.CharField(choices=[('classification', 'Classification'), ('boundingbox', 'Bounding box'), ('landmark', 'Landmark'), ('cardiac_segmentation', 'Cardiac apical segmentation'), ('cardiac_plax_segmentation', 'Cardiac PLAX segmentation'), ('cardiac_alax_segmentation', 'Cardiac ALAX segmentation'), ('spline_segmentation', 'Spline segmentation'), ('spline_line_point', 'Splines, lines & point segmentation'), ('auto_segmentation', 'Auto segmentation')], max_length=50),
        ),
    ]
