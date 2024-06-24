extends Node2D


@onready var texture_rect = $CanvasLayer/HBoxContainer/TextureRect
@onready var prediction_label = $CanvasLayer/HBoxContainer/VBoxContainer/PredictionLabel

var images_file_path = "res://data/train-images.idx3-ubyte"
var labels_file_path = "res://data/train-labels.idx1-ubyte"

var image_data
var label_data

var training_image_data
var training_label_data

var image_batches
var label_batches

var testing_image_data
var testing_label_data

var image_index = 0

var batch_size: int = 100
var learn_rate: float = 0.5
var epochs: int = 5
var training_split: float = 1
var momentum: int = 0.8
var hidden_layer_activation = ActivationFactory.new_activation(ActivationFactory.type.SIGMOID)
var output_layer_activation = ActivationFactory.new_activation(ActivationFactory.type.SOFTMAX)
var cost = CostFactory.new_cost(CostFactory.type.CROSS_ENTROPY)

var net: NeuralNetwork
func _ready():
	
	image_data = load_images(images_file_path, 2500)
	label_data = load_labels(labels_file_path, 2500)
	
	training_image_data = image_data.slice(0, snapped(training_split*image_data.size(), batch_size))
	testing_image_data = image_data.slice(training_image_data.size(), image_data.size())
	
	training_label_data = label_data.slice(0, snapped(training_split*label_data.size(), batch_size))
	testing_label_data = label_data.slice(training_label_data.size(), label_data.size())
	
	shuffle_data(image_data, label_data)
	image_batches = create_mini_batches(training_image_data, batch_size)
	label_batches = create_mini_batches(training_label_data, batch_size)
	
	#net = NeuralNetwork.new(784, 10, 16, 2, hidden_layer_activation, output_layer_activation, cost)
	net = NeuralNetwork.load_from_file("res://saves/neural_network_save.json")
var start = false
func _on_train_button_pressed():
	start = not start

func _process(delta):
	while start:
		train_one_batch()
	
var batch_index = 0
func train_one_batch():
	return
	net.train(image_batches[batch_index], label_batches[batch_index], learn_rate, epochs, momentum)
	print("Total Cost: " + str(net.calculate_average_cost(image_data, label_data)))
	batch_index += 1
	if batch_index == image_batches.size():
		batch_index = 0
	net.save_to_file("res://saves/neural_network_save.json")


const SPRITE_0001 = preload("res://sprites/Sprite-0001.png")
var test_index = 0
func _on_next_button_pressed():
	var d = convert_texture_to_data(SPRITE_0001)
	set_texture_on_rect(d)
	update_prediction_label(net.forward_propagate(d))
	return
	set_texture_on_rect(image_data[test_index])
	update_prediction_label(net.forward_propagate(image_data[test_index]))
	test_index += 1
	if test_index == image_data.size():
		test_index = 0

func set_texture_on_rect(data_point: Array):
	texture_rect.texture = create_texture_from_data(data_point, 28, 28)
	

func shuffle_data(image_data: Array, label_data: Array):
	
	var size = image_data.size()
	for i in size:
		
		var j = randi() % size
		
		var temp_image = image_data[i]
		image_data[i] = image_data[j]
		image_data[j] = temp_image
		
		var temp_label = label_data[i]
		label_data[i] = label_data[j]
		label_data[j] = temp_label
		
	
func create_mini_batches(data: Array, batch_size: int):
	var batches = []
	for i in range(0, data.size(), batch_size):
		batches.append(data.slice(i, i+batch_size))
	return batches

func convert_texture_to_data(texture: Texture2D):
	var image = texture.get_image()
	var data: Array = []
	for y in 28:
		for x in 28:
			data.append(image.get_pixel(x, y).get_luminance())
	return data

func create_texture_from_data(pixel_data: Array, width: int, height: int):

	var image = Image.create(width, height, false, Image.FORMAT_L8)
	for y in height:
		for x in width:
			var v = pixel_data[y * width + x]
			image.set_pixel(x, y, Color(v,v,v))
	var texture = ImageTexture.new()
	texture.set_image(image)
	return texture


func update_prediction_label(predicted_outputs: Array):
	prediction_label.clear()
	var predicted_outputs_with_label: Array[Vector2] = []
	for i in predicted_outputs.size():
		predicted_outputs_with_label.append(Vector2(i, predicted_outputs[i]))
	predicted_outputs_with_label.sort_custom(sort_predictions)
	
	for i in predicted_outputs_with_label.size():
		if i == 0:
			prediction_label.append_text("[b]" + str(predicted_outputs_with_label[i].x) + ": " + str(snapped(100*predicted_outputs_with_label[i].y, 0.01))  + "%[/b]\n\n")
		else:
			prediction_label.append_text(str(predicted_outputs_with_label[i].x) + ": " + str(snapped(100*predicted_outputs_with_label[i].y, 0.01)) + "%\n\n")
			
func sort_predictions(a, b):
	return a.y > b.y

func load_images(images_file_path, image_count = -1):
	
	var file = FileAccess.open(images_file_path, FileAccess.READ)
	
	file.seek(16)
	
	var rows = 28
	var columns = 28
	var images = []
	var count = 0
	
	for i in (file.get_length() - file.get_position()) / (rows * columns):
		var image = []
		for r in rows:
			for c in columns:
				image.append(file.get_8() / 255.0)
		images.append(image)
		count += 1
		if count == image_count:
			break

	file.close()
	return images
	
func load_labels(labels_file_path, label_count = -1):
	
	var file = FileAccess.open(labels_file_path, FileAccess.READ)
	
	file.seek(8)
	
	var labels = []
	var count = 0

	for i in file.get_length() - file.get_position():
		labels.append(file.get_8())
		count += 1
		if count == label_count:
			break
	file.close()
	
	var new_labels = []
	for label in labels:
		var temp = []
		for i in 10:
			if i == label:
				temp.append(1)
			else:
				temp.append(0)
		new_labels.append(temp)
		
	return new_labels


