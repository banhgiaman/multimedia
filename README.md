# Multimedia
Master multimedia subject's project

## Install requirements libraries
```
pip install -r requirements.txt
```

## Train the EfficientNet model based on tensorflow built-in
```
At project root, go to EfficentNetwork directory: 
    cd EfficentNetwork

## To train the model, you need to place the flowers dataset into EfficientNetwork/data ##
## You can download the dataset at: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ ##

Format the dataset structure before train the model:
    python dataset_converter.py

## You can change the number of epoches or batch_size or the network B_ version in the `efficientnet_102flowers.py` file based on your GPU strength ##
Train the model:
    python efficientnet_102flowers.py
```

## Run and predict flowers in the website with Django
```
At project root, go to FlowersClassification directory:
    cd FlowersClassification

## Copy/Replace your model instead after being successfully trained into FlowersClassification/dl_models ##
Run the Django server:
    python manage.py runserver
``` 
