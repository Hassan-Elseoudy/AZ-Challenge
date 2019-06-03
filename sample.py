from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import os
import time
import tqdm

ENDPOINT = "https://westeurope.api.cognitive.microsoft.com"

# Replace with a valid key
training_key = "ba19559b3388427888f213e7cdc8c203"
prediction_key = "fc857b03c0ca4eed97e230836e9c11c0"
prediction_resource_id = "/subscriptions/b6e1399d-f923-48c8-8556-12606973af0d/resourceGroups/Azure/providers/Microsoft.CognitiveServices/accounts/Azure_prediction"

publish_iteration_name = "classifyModel"

trainer = CustomVisionTrainingClient(training_key, endpoint=ENDPOINT)

# Create a new project
print ("Creating project...")
project = trainer.create_project("My New Project")

base_image_url = "/home/smsm/PycharmProjects/Azure_Challenge_/cognitive-services-python-sdk-samples/samples/vision/images/huh"

# Make tags in the new project
print('Adding tags...')
tags = {}
for class_ in os.listdir(base_image_url):
    tags[class_] = trainer.create_tag(project.id, class_)
    print("Tag:",tags[class_],"Class:",class_)

# function
def function_kda(path, tag):
    for file_name in (os.listdir(path)):
        print('Getting images ...')
        with open(os.path.join(path, file_name), "rb") as image_contents:
            image_list.append(ImageFileCreateEntry(name = file_name, contents = image_contents.read(), tag_ids = [tag]))


# make imgaes in the new project
print("Adding images...")
image_list = []
for class_ in os.listdir(base_image_url):
    function_kda(os.path.join(base_image_url, class_), class_)

print("Length:",len(image_list))
upload_result = trainer.create_images_from_files(project.id, images=image_list)
if not upload_result.is_batch_successful:
    print("Image batch upload failed.")
    for image in upload_result.images:
        print("Image status: ", image.status)
    exit(-1)

print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")

# Now there is a trained endpoint that can be used to make a prediction
predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

with open(base_image_url + "images/Test/test_image.jpg", "rb") as image_contents:
    results = predictor.classify_image(project.id, publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))