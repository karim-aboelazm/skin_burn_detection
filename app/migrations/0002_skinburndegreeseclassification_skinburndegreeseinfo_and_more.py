# Generated by Django 4.2 on 2023-05-04 12:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="SkinBurnDegreeseClassification",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("skin_image", models.ImageField(upload_to="skin_process/")),
                ("skin_burn_classification", models.CharField(max_length=225)),
                ("skin_burn_accuracy", models.CharField(max_length=225)),
            ],
            options={
                "verbose_name_plural": "Skin Burn Degreese Classification",
            },
        ),
        migrations.CreateModel(
            name="SkinBurnDegreeseInfo",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("title", models.CharField(max_length=255)),
                ("definition", models.TextField()),
                ("causes", models.TextField()),
                ("treatment", models.TextField()),
            ],
            options={
                "verbose_name_plural": "Skin Burn Degreese Info",
            },
        ),
        migrations.DeleteModel(
            name="SkinImagePrediction",
        ),
    ]
