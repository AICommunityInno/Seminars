from keras.preprocessing.image import *

from skimage.io import *

import os


class SegmentationIterator(Iterator):
    def __init__(self, images_directory,
                 masks_directory,
                 image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None,
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 follow_links=False, interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

        # first, count the number of samples and classes
        image_samples = self._count_valid_files_in_directory(images_directory, white_list_formats)
        mask_samples = self._count_valid_files_in_directory(masks_directory, white_list_formats)
        
        assert image_samples == mask_samples, 'Amount of images and masks must be the same'
        
        self.samples = image_samples

        print('Found %d images.' % (self.samples,))

        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        
        image_paths = list(sorted(os.listdir(images_directory)))
        mask_paths = list(sorted(os.listdir(masks_directory)))
        
        assert image_paths == mask_paths, 'Image names must match mask names'
        
        for i, fname in enumerate(image_paths):
            if self._is_in_white_list(fname, white_list_formats):
                self.filenames.append(fname)
        
        super(SegmentationIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        
    def _is_in_white_list(self, path, white_list_formats):
        for ext in white_list_formats:
            if path.lower().endswith('.' + ext):
                return True
        
        return False
    
    def _count_valid_files_in_directory(self, directory, white_list_formats):
        count = 0
        for fname in os.listdir(directory):
            if self._is_in_white_list(fname, white_list_formats):
                count += 1
        
        return count

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array),) + self.image_shape[:2] + (1,), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.images_directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            
            self.image_data_generator.random_transform(x)
            x = self.image_data_generator.apply_augmentation(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x / 255.
            
            mask = load_img(os.path.join(self.masks_directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            mask = img_to_array(mask, data_format=self.data_format)
            
            mask = self.image_data_generator.apply_augmentation(mask)
            mask = self.image_data_generator.standardize(mask)
            batch_y[i] = mask / 255.
            
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    
class SegmentationGenerator(ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(SegmentationGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format
        )
        
    def flow_from_directory(self, images_directory,
                            masks_directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None,
                            batch_size=32, shuffle=True, seed=None,
                            data_format=None, save_to_dir=None,
                            save_prefix='', save_format='png',
                            follow_links=False, interpolation='nearest'):
        return SegmentationIterator(
            images_directory,
            masks_directory,
            self,
            target_size=target_size, color_mode=color_mode,
            classes=classes,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            interpolation=interpolation)
    
    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        self.img_row_axis = self.row_axis - 1
        self.img_col_axis = self.col_axis - 1
        self.img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[self.img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[self.img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
            
        self.transform_matrix = transform_matrix
        
        if self.horizontal_flip:
            self.horizontal_flip_val = np.random.random()

        if self.vertical_flip:
            self.vertical_flip_val = np.random.random()
    
    def apply_augmentation(self, x):
        if self.transform_matrix is not None:
            h, w = x.shape[self.img_row_axis], x.shape[self.img_col_axis]
            transform_matrix = transform_matrix_offset_center(self.transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, self.img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
        
        if self.horizontal_flip and self.horizontal_flip_val < 0.5:
            x = flip_axis(x, self.img_col_axis)

        if self.vertical_flip and self.vertical_flip_val < 0.5:
            x = flip_axis(x, self.img_row_axis)

        return x
