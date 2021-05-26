import os
import glob
import numpy as np
from scipy.io import loadmat
from shutil import copyfile, rmtree

flowers_name = [
    'pink primrose',
    'hard-leaved pocket orchid',
    'canterbury bells',
    'sweet pea',
    'english marigold',
    'tiger lily',
    'moon orchid',
    'bird of paradise',
    'monkshood',
    'globe thistle',
    'snapdragon',
    "colt's foot",
    'king protea',
    'spear thistle',
    'yellow iris',
    'globe-flower',
    'purple coneflower',
    'peruvian lily',
    'balloon flower',
    'giant white arum lily',
    'fire lily',
    'pincushion flower',
    'fritillary',
    'red ginger',
    'grape hyacinth',
    'corn poppy',
    'prince of wales feathers',
    'stemless gentian',
    'artichoke',
    'sweet william',
    'carnation',
    'garden phlox',
    'love in the mist',
    'mexican aster',
    'alpine sea holly',
    'ruby-lipped cattleya',
    'cape flower',
    'great masterwort',
    'siam tulip',
    'lenten rose',
    'barbeton daisy',
    'daffodil',
    'sword lily',
    'poinsettia',
    'bolero deep blue',
    'wallflower',
    'marigold',
    'buttercup',
    'oxeye daisy',
    'common dandelion',
    'petunia',
    'wild pansy',
    'primula',
    'sunflower',
    'pelargonium',
    'bishop of llandaff',
    'gaura',
    'geranium',
    'orange dahlia',
    'pink-yellow dahlia',
    'cautleya spicata',
    'japanese anemone',
    'black-eyed susan',
    'silverbush',
    'californian poppy',
    'osteospermum',
    'spring crocus',
    'bearded iris',
    'windflower',
    'tree poppy',
    'gazania',
    'azalea',
    'water lily',
    'rose',
    'thorn apple',
    'morning glory',
    'passion flower',
    'lotus',
    'toad lily',
    'anthurium',
    'frangipani',
    'clematis',
    'hibiscus',
    'columbine',
    'desert-rose',
    'tree mallow',
    'magnolia',
    'cyclamen',
    'watercress',
    'canna lily',
    'hippeastrum',
    'bee balm',
    'ball moss',
    'foxglove',
    'bougainvillea',
    'camellia',
    'mallow',
    'mexican petunia',
    'bromelia',
    'blanket flower',
    'trumpet creeper',
    'blackberry lily'
]
flowers_name = np.array(flowers_name)

data_path = os.path.join('data')
setid_path = os.path.join(data_path, 'setid.mat')
image_labels_path = os.path.join(data_path, 'imagelabels.mat')
data_dir = os.path.join(data_path, 'sorted')

setid = loadmat(setid_path)
idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

image_labels = loadmat(image_labels_path)['labels'][0]
image_labels -= 1
image_labels = flowers_name[image_labels]

files = sorted(glob.glob(os.path.join(data_path, '102flowers', '*.jpg')))
labels = np.array([i for i in zip(files, image_labels)])

if os.path.exists(data_dir):
    rmtree(data_dir, ignore_errors=True)
os.mkdir(data_dir)

def move_files(dir_name, labels):
    cur_dir_path = os.path.join(data_dir, dir_name)
    
    if not os.path.exists(cur_dir_path):
        os.mkdir(cur_dir_path)
        
    for i in flowers_name:
        class_dir = os.path.join(cur_dir_path, str(i))
        os.mkdir(class_dir)

    for label in labels:
        src = str(label[0])
        dst = os.path.join(cur_dir_path, label[1], src.split(os.sep)[-1])
        copyfile(src, dst)

move_files('test', labels[idx_test, :])
move_files('train', labels[idx_train, :])
move_files('valid', labels[idx_valid, :])
