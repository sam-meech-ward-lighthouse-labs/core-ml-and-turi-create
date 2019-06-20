// /Applications/Adobe Photoshop CC 2019/Presets/Scripts
const layer = app.activeDocument.layers[0];

const frame = {
  x: parseInt(layer.bounds[0]),
  y: parseInt(layer.bounds[1]),
  width: parseInt(layer.bounds[2] - layer.bounds[0]),
  height: parseInt(layer.bounds[3] - layer.bounds[1]),
};
frame.centerX = parseInt(frame.x + frame.width/2);
frame.centerY = parseInt(frame.y + frame.height/2);

const output = '"coordinates": {"height": '+
frame.height+', "width": '+
frame.width+', "x": '+
frame.centerX+', "y": '+
frame.centerY+'}';

alert(output);
