# Solving PDEs with CNNs

This is a little proof of concept project to see whether partial differential equations can be solved by convolutional neural networks.

In particular, we try to solve the navier stokes + continuity equation with a small UNet.

## setting up the environment:

to get this Code to run, you should have installed the following packages:

- numpy
- pytorch
- tensorboardX
- visdom
- opencv-python
- statsmodels
- natsort
- matplotlib

... I hope this list is more or less complete

## train a model:

to train a model, call:

> ./run.sh train.py --n_epochs=300

this command will print all the outputs of train.py into the /Logs folder.
Furthermore, it will put the learning metrics into the /Logger/tensorboard folder. To view them, call:

> tensorboard --logdir=Logger/tensorboard

Finally, after each epoch, it will store the model parameters inside the /Logger folder.

## test a model:

to test the model and get a little "opencv-movie", call:

> python test_cv2.py

to get some test snapshots with matplotlib, call:

> python test.py
