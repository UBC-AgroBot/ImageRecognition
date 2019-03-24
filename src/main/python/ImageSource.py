import queue
import os

class ImageSource:
    directory = None
    _image_queue = queue.Queue()

    def __init__(self, directory):
        self.directory = directory
        self._setup_images()

    def _setup_images(self):
        root_dir = self.directory
        for dir_name, subdir_list, file_list in os.walk(root_dir):
            for file in file_list:
                if ".jpg" in file:
                    file_path = root_dir + "/" + dir_name + "/" + file
                    self._image_queue.put(file_path)

    def next(self):
        return self._image_queue.get()

    def has_next(self):
        return self._image_queue.empty