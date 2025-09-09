'''
This file save all of hyperparameters
'''

# WIP - Remove this line if the path below works works  "example_testing_root": "/home/nugh75/Git/archeology-fragment-reconstruction/dataset/250331-clean/set24/images/fragments/Pillars-of-St-George-Serbia1282.jpg/s17",
# NOTE: directory must end at '/', because we use this path to call external c++ program
WorkSpacePath = {
    "training_dataset_root": "",
    "testing_dataset_root": "your path",
    "example_testing_root": "../Examples/s17",
    "checkpoint_dir": "JigsawNet/checkpoints/",
}

# for CNN
NNHyperparameters = {
    "width": 160,       # image width
    "height": 160,      # image height
    "depth" : 3,        # action candidates + original image
    "batch_size": 64,
    "weight_decay": 1e-4,
    "learning_rate": 1e-4,
    "total_training_step": 30000,
    "learner_num": 5
}