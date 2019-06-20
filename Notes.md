## Python Download

https://www.python.org/downloads/release/python-368/

## Project Setup

https://github.com/apple/turicreate

## Object Detection

https://github.com/apple/turicreate/tree/master/userguide/object_detection

## Transfered Learning

https://github.com/apple/turicreate/blob/master/userguide/object_detection/how-it-works.md#transfer-learning

## Code

### Object Detection

```python
import turicreate as tc

# Import the data
annotations = tc.SFrame.read_json('file_resized.json', orient='records')
images = tc.load_images('images_resized/')
data = images.join(annotations)

# Split the data for testing
train_data, test_data = data.random_split(0.8)

# "expolore" the data
# this part work only on Mac.
data['image_with_ground_truth'] = \
   tc.object_detector.util.draw_bounding_boxes(data['image'], data['annotation'])
data.explore()

# Start training, this will take a while
print("Start Training")
model = tc.object_detector.create(train_data, max_iterations=1)

# Test the model
predictions = model.predict(test_data)
test_data['predicated_image'] = tc.object_detector.util.draw_bounding_boxes(test_data['image'], predictions)
test_data['predicated_image'][0].show()
test_data.explore()
model.evaluate(test_data)

# Save the model for later use in Turi Create
model.save('mymodel.model')

# Export for use in Core ML
model.export_coreml('MyCustomObjectDetector.mlmodel')
```

### Loading In A Model

```python
model = tc.load_model('their_model.model')
```


##

**Loss is the penalty for a bad prediction.**

```
Start Training
Using 'image' as feature column
Using 'annotation' as annotations column
Setting 'batch_size' to 32
Using GPU to create model (AMD Radeon Pro 560)
+--------------+--------------+--------------+
| Iteration    | Loss         | Elapsed Time |
+--------------+--------------+--------------+
| 1            | 4.262        | 9.7          |
| 12           | 4.312        | 20.7         |
| 23           | 3.818        | 31.6         |
| 34           | 2.962        | 42.5         |
| 45           | 2.350        | 53.4         |
| 56           | 1.915        | 64.2         |
| 67           | 1.689        | 75.1         |
| 78           | 1.471        | 86.0         |
| 89           | 1.307        | 96.9         |
| 100          | 1.216        | 107.7        |
| 111          | 1.213        | 118.7        |
| 122          | 1.137        | 129.5        |
| 133          | 1.029        | 140.4        |
| 144          | 1.058        | 151.2        |
| 155          | 1.008        | 162.2        |
| 166          | 0.992        | 173.1        |
| 177          | 0.946        | 184.0        |
| 188          | 0.881        | 194.8        |
| 199          | 0.853        | 205.7        |
| 210          | 0.836        | 216.6        |
| 221          | 0.818        | 227.5        |
| 232          | 0.795        | 238.3        |
| 243          | 0.715        | 249.3        |
| 254          | 0.676        | 260.1        |
| 265          | 0.735        | 271.0        |
| 276          | 0.673        | 281.9        |
| 287          | 0.684        | 292.8        |
| 298          | 0.654        | 303.6        |
| 309          | 0.666        | 314.5        |
| 320          | 0.656        | 325.3        |
| 331          | 0.604        | 336.3        |
| 342          | 0.596        | 347.1        |
| 353          | 0.606        | 358.1        |
| 364          | 0.617        | 368.9        |
| 375          | 0.595        | 379.8        |
| 386          | 0.586        | 390.6        |
| 397          | 0.589        | 401.6        |
| 408          | 0.581        | 412.4        |
| 419          | 0.554        | 423.3        |
| 430          | 0.511        | 434.1        |
| 441          | 0.507        | 445.0        |
| 452          | 0.511        | 455.9        |
| 463          | 0.537        | 466.9        |
| 474          | 0.537        | 477.7        |
| 485          | 0.525        | 488.6        |
| 496          | 0.552        | 499.4        |
| 507          | 0.661        | 510.3        |
| 518          | 0.772        | 521.2        |
| 529          | 0.638        | 532.1        |
| 540          | 0.530        | 542.9        |
| 551          | 0.479        | 553.8        |
| 562          | 0.446        | 564.6        |
| 573          | 0.454        | 575.6        |
| 584          | 0.405        | 586.4        |
| 595          | 0.394        | 597.3        |
| 606          | 0.399        | 608.2        |
| 617          | 0.403        | 619.1        |
| 628          | 0.392        | 629.9        |
| 639          | 0.396        | 640.9        |
| 650          | 0.390        | 651.7        |
| 661          | 0.377        | 662.6        |
| 672          | 0.379        | 673.4        |
| 683          | 0.365        | 684.3        |
| 694          | 0.372        | 695.2        |
| 705          | 0.384        | 706.2        |
| 716          | 0.375        | 717.0        |
| 727          | 0.364        | 727.9        |
| 738          | 0.367        | 738.7        |
| 749          | 0.373        | 749.6        |
| 760          | 0.360        | 760.5        |
| 771          | 0.375        | 771.4        |
| 782          | 0.377        | 782.2        |
| 793          | 0.365        | 793.2        |
| 804          | 0.359        | 804.0        |
| 815          | 0.389        | 814.9        |
| 826          | 0.388        | 825.8        |
| 837          | 0.372        | 836.7        |
| 848          | 0.375        | 847.5        |
| 859          | 0.352        | 858.4        |
| 870          | 0.374        | 869.2        |
| 881          | 0.378        | 880.2        |
| 892          | 0.355        | 891.0        |
| 903          | 0.367        | 901.9        |
| 914          | 0.396        | 912.8        |
| 925          | 0.393        | 923.7        |
| 936          | 0.374        | 934.5        |
| 947          | 0.364        | 945.5        |
| 958          | 0.373        | 956.3        |
| 969          | 0.372        | 967.3        |
| 980          | 0.370        | 978.0        |
| 991          | 0.344        | 989.0        |
| 1000         | 0.355        | 997.8        |
+--------------+--------------+--------------+
Class                                    : ObjectDetector

Schema
------
Model                                    : darknet-yolo
Number of classes                        : 1
Non-maximum suppression threshold        : 0.45
Input image shape                        : (3, 416, 416)

Training summary
----------------
Training time                            : 16m 40s
Training epochs                          : 4000
Training iterations                      : 1000
Number of examples (images)              : 8
Number of bounding boxes (instances)     : 8
Final loss (specific to model)           : 0.355
```

## Links

### ML

- [Videos on the technical details](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [](https://www.youtube.com/watch?v=R9OHn5ZF4Uo)

### Python

- [Python Crash Course](https://machinelearningmastery.com/crash-course-python-machine-learning-developers/)

### Core ML

- [Ray Wenderlich](https://www.raywenderlich.com/164213/coreml-and-vision-machine-learning-in-ios-11-tutorial)
- [turicreate](https://github.com/apple/turicreate)
- [Vision with Core ML](https://developer.apple.com/videos/play/wwdc2018/717)
- [Understanding Images in Vision Framework](https://developer.apple.com/videos/play/wwdc2019/222)
- [Recognizing Objects in Live Capture](https://developer.apple.com/documentation/vision/recognizing_objects_in_live_capture)
- [Classifying Images with Vision and Core ML](https://developer.apple.com/documentation/vision/classifying_images_with_vision_and_core_ml)
- [coreml](https://developer.apple.com/documentation/coreml)
