import keras.layers as kl
import keras.models as km
import keras.preprocessing as kp
import keras.optimizers as ko
import keras.callbacks as kc

def unet(input_size):
  inputs = kl.Input(input_size)
  downsampling_sizes = [128, 256, 512, 1024]
  upsampling_sizes = [512, 256, 128, 64]
  convs = []

  # Conv64 + ReLU + BN
  conv = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv = kl.BatchNormalization()(conv)
  # Conv64 + ReLU + BN
  conv = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
  conv = kl.BatchNormalization()(conv)

  for downsampling_size in downsampling_sizes:
    convs.append(conv)
    # 2x2 Max Pool
    pool = kl.MaxPooling2D(pool_size=(2, 2))(conv)
    # Conv + ReLU + BN
    conv = kl.Conv2D(downsampling_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool)
    conv = kl.BatchNormalization()(conv)
    # Conv + ReLU + BN
    conv = kl.Conv2D(downsampling_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = kl.BatchNormalization()(conv)
    # remember for concatenation

  for upsampling_size in upsampling_sizes:
    print ("JAZZ " + str(upsampling_size) + " JAZZ")
    # 2x2 upsampling
    depool = kl.UpSampling2D(size = (2,2))(conv)
    # Conv
    upconv = kl.Conv2D(upsampling_size, 2, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(depool)
    # Concat
    merge = kl.concatenate([convs.pop(),upconv], axis = 3)
    # Conv + ReLU + BN
    conv = kl.Conv2D(upsampling_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
    conv = kl.BatchNormalization()(conv)
    # Conv + ReLU + BN
    conv = kl.Conv2D(upsampling_size, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = kl.BatchNormalization()(conv)

  # sigmoid
  conv = kl.Conv2D(1, 1, activation = 'sigmoid')(conv)
  model = km.Model(input = inputs, output = conv)
  return model

data_gen_args = dict(
rotation_range=30,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.3,horizontal_flip=True)
image_datagen = kp.image.ImageDataGenerator(preprocessing_function=lambda x: x/255.,**data_gen_args)
mask_datagen = kp.image.ImageDataGenerator(**data_gen_args)

target_size = (240,320)
batch_size = 8

image_generator = image_datagen.flow_from_directory(
  './imgs_train',
  class_mode=None,
  seed=0,target_size=target_size,batch_size=batch_size)

mask_generator = mask_datagen.flow_from_directory(
  './masks_train',
  class_mode=None,
  seed=0,target_size=target_size,color_mode='grayscale',batch_size=batch_size)

image_generator_test = image_datagen.flow_from_directory(
  './imgs_test',
  class_mode=None,
  seed=0,target_size=target_size,batch_size=batch_size)

mask_generator_test = mask_datagen.flow_from_directory(
  './masks_test',
  class_mode=None,
  seed=0,target_size=target_size,color_mode='grayscale',batch_size=batch_size)

train_generator = zip(image_generator, mask_generator)
test_generator = zip(image_generator_test,mask_generator_test)

model = unet((240,320,3))

model.compile(optimizer = ko.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

filepath = './checkpoints_2.h5'

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = kc.ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

def lr_schedule(epoch):
  lr = 1e-4
  if epoch > 80:
    lr /=16.
  elif epoch > 40:
      lr /= 8.
  elif epoch > 25:
    lr /= 4.
  elif epoch > 10:
    lr /= 2.
  print('Learning rate: ', lr)
  return lr

lr_scheduler = kc.LearningRateScheduler(lr_schedule)

# model.load_weights('checkpoints_2.h5')
model.fit_generator(train_generator, steps_per_epoch=(3041-305)//batch_size,epochs=200,validation_data=test_generator,validation_steps=301//8,callbacks=[checkpoint,lr_scheduler])

# Make a prediction on the test set

imgs = image_generator_test.next()
masks_true = mask_generator_test.next()
masks = model.predict(imgs)
fig,axs = plt.subplots(nrows=batch_size,figsize=(4,4*batch_size))
for img,mask,ax in zip(imgs,masks,axs):
  ax.imshow(img.squeeze())
  ax.imshow(mask.squeeze(),alpha=0.6,vmin=0,vmax=1)
