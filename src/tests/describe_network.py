from keras.applications import VGG19, MobileNet
from keras.engine import Model

base_model = MobileNet(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv2').output)

lay = base_model.layers

print(lay)
for ll in lay:
    print(ll.name)
