from ImageClassifier import ImageClassifier
from ImageSource import ImageSource

class Controller:
    image_source = None
    flower_classifier = None

    def __init__(self):
        flower_model_path = "../outputs/flower_model/output_graph.pb"
        flower_labels_path = "../outputs/flower_model/output_labels.txt"
        self.flower_classifier = ImageClassifier(flower_model_path, flower_labels_path)
        self.image_source = ImageSource("../flower_photos")

    def run(self):
        while True:
            if self.image_source.has_next:
                result = self.flower_classifier.classify(self.image_source.next)
                print(result)
                

    