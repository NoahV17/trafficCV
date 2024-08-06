import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the dataset
dataset, info = tfds.load('coco/2017', split='train', with_info=True)

# List of desired classes
desired_classes = ['traffic light', 'stop sign', 'car', 'person', 'street sign']
desired_class_indices = []
for cls in desired_classes:
    try:
        index = info.features['objects']['label'].str2int(cls)
        desired_class_indices.append(index)
    except KeyError:
        print(f"Warning: '{cls}' is not a valid class name in the COCO 2017 dataset.")


def filter_fn(data):
    """Filter function to include only desired classes."""
    objects = data['objects']
    valid_indices = tf.where(tf.reduce_any([objects['label'] == idx for idx in desired_class_indices], axis=0))
    filtered_bboxes = tf.gather(objects['bbox'], valid_indices)
    filtered_labels = tf.gather(objects['label'], valid_indices)
    return {
        'image': data['image'],
        'objects': {
            'bbox': tf.reshape(filtered_bboxes, [-1, 4]),
            'label': tf.reshape(filtered_labels, [-1])
        }
    }

filtered_dataset = dataset.map(filter_fn)

# Function to visualize a sample
def visualize_sample(data):
    image = data['image'].numpy()
    bboxes = data['objects']['bbox'].numpy()
    labels = data['objects']['label'].numpy()

    plt.imshow(image)
    ax = plt.gca()
    for bbox, label in zip(bboxes, labels):
        y_min, x_min, y_max, x_max = bbox
        rect = plt.Rectangle((x_min * image.shape[1], y_min * image.shape[0]),
                             (x_max - x_min) * image.shape[1],
                             (y_max - y_min) * image.shape[0],
                             fill=False, color='red')
        ax.add_patch(rect)
        plt.text(x_min * image.shape[1], y_min * image.shape[0], 
                 info.features['objects']['label'].int2str(label), 
                 color='red', fontsize=12)
    plt.show()

# Visualize some samples
for data in filtered_dataset.take(5):
    visualize_sample(data)
