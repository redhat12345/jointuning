import jointuning.utils.data as data
import jointuning.config as config
from jointuning.alexnet import preprocess_input


data_dir = '/home/fniu/Documents/dog_proj/data/train'
model = 'alexnet'
layer_names = ['conv_1', 'conv_2_1']
target_size = (227, 227)
pytables_path = '/home/fniu/Desktop/train.hdf5'
group_name = 'train'
batch_size = 1000

data.get_image_descriptor_from_dir(data_dir, model,
                                   layer_names=layer_names,
                                   nbins=100,
                                   target_size=target_size,
                                   batch_size=batch_size,
                                   preprocessing_function=preprocess_input,
                                   image_ext_list=config.IMAGE_EXTENSIONS,
                                   save_to_pytables=True,
                                   pytables_path=pytables_path,
                                   group_name=group_name,
                                   feature_table_name='features',
                                   path_table_name='paths')
