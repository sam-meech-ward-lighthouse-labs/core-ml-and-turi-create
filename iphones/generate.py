import turicreate as tc

# Import the data
annotations = tc.SFrame.read_json('data.json', orient='records')
images = tc.load_images('images/')
data = images.join(annotations)

# Split the data for testing
train_data, test_data = data.random_split(0.8)

# "expolore" the data
# this part work only on Mac.
# data['image_with_ground_truth'] = \
#    tc.object_detector.util.draw_bounding_boxes(data['image'], data['annotation'])
# data.explore()

# Start training, this will take a while
print("Start Training")
model = tc.object_detector.create(train_data, max_iterations=1)

# Save the model for later use in Turi Create
model.save('mymodel.model')

# Export for use in Core ML
model.export_coreml('MyCustomObjectDetector.mlmodel')