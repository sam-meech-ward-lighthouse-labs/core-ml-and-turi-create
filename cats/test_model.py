import turicreate as tc

# Import the data
# annotations = tc.SFrame.read_json('file_resized.json', orient='records')
data = tc.load_images('test.jpg')
# data = images.join(annotations)

# "expolore" the data
# this part work only on Mac.
# data['image_with_ground_truth'] = \
#    tc.object_detector.util.draw_bounding_boxes(data['image'], data['annotation'])
# data.explore()

model = tc.load_model('their_model.model')

# Test the model
predictions = model.predict(data)
data['predicated_image'] = \
  tc.object_detector.util.draw_bounding_boxes(data['image'], predictions)

data['predicated_image'][0].show()
data.explore()
