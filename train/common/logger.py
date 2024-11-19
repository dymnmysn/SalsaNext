import tensorflow as tf
import numpy as np
import io
from PIL import Image
import torch


class Logger:
    def __init__(self, log_dir):
        """
        Create a summary writer logging to log_dir.
        """
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        Log a scalar variable.
        Converts PyTorch tensors to NumPy or scalar values if necessary.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()  # Convert to a Python scalar
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


    def image_summary(self, tag, images, step):
        """
        Log a list of images.
        """
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert the image to a format TensorFlow can handle
                img = np.asarray(img, dtype=np.uint8)  # Ensure the image is uint8
                if img.ndim == 2:  # Grayscale to RGB
                    img = np.stack([img] * 3, axis=-1)
                elif img.ndim == 3 and img.shape[-1] == 1:  # Single channel to RGB
                    img = np.squeeze(img, axis=-1)
                    img = np.stack([img] * 3, axis=-1)

                # Log the image
                tf.summary.image(f"{tag}/{i}", [img], step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """
        Log a histogram of the tensor of values.
        """
        with self.writer.as_default():
            counts, bin_edges = np.histogram(values, bins=bins)
            bin_edges = bin_edges[:-1]  # Remove the last edge for consistency

            # Create histogram data for TensorFlow
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()

