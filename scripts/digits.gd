class_name Digits

extends Node2D

@onready var sub_viewport = $SubViewport
@onready var input_sprite = $SubViewport/InputSprite

@onready var texture_rect = $CanvasLayer/HBoxContainer/TextureRect
@onready var prediction_label = $CanvasLayer/HBoxContainer/VBoxContainer/PredictionLabel

@onready var training_accuracy_label = $CanvasLayer/HBoxContainer/VBoxContainer2/TrainingAccuracyLabel
@onready var testing_accuracy_label = $CanvasLayer/HBoxContainer/VBoxContainer2/TestingAccuracyLabel
@onready var epoch_completion_label = $CanvasLayer/HBoxContainer/VBoxContainer2/EpochCompletionLabel

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

var batch_size: int = 500
var learn_rate: float = 0.3
var epochs: int = 1
var training_split: float = 1
var momentum: int = 0.8
var hidden_layer_activation = ActivationFactory.new_activation(ActivationFactory.type.SIGMOID)
var output_layer_activation = ActivationFactory.new_activation(ActivationFactory.type.SOFTMAX)
var cost = CostFactory.new_cost(CostFactory.type.CROSS_ENTROPY)

var noise_strength: float = 0.0

var net: NeuralNetwork

var layers: Array = [784, 16, 16, 10]

func _ready():
	
	label_data = load_labels_ubyte(labels_file_path, 10000)
	#image_data = load_from_json("res://data/image_data_noisy.json")
	image_data = load_images_ubyte(images_file_path, 10000)
	#for i in image_data.size():
		#image_data[i] = await apply_noise(image_data[i])
	#save_to_json("res://data/image_data_noisy.json", image_data)
	
	#image_data = [[0,0], [0,1], [1,0], [1,1]]
	#label_data = [[0,1], [1,0], [1,0], [0,1]]
	
	training_image_data = image_data.slice(0, snapped(training_split*image_data.size(), batch_size))
	testing_image_data = image_data.slice(training_image_data.size(), image_data.size())
	
	training_label_data = label_data.slice(0, snapped(training_split*label_data.size(), batch_size))
	testing_label_data = label_data.slice(training_label_data.size(), label_data.size())
	
	shuffle_data(image_data, label_data)
	image_batches = create_mini_batches(training_image_data, batch_size)
	label_batches = create_mini_batches(training_label_data, batch_size)
	
	net = NeuralNetwork.new(layers, hidden_layer_activation, output_layer_activation, cost)
	#net = NeuralNetwork.load_from_file("res://saves/neural_network_save.json")
	print("Initialization Complete")
	
func _on_train_button_pressed():
	if net == null:
		return
	start_training()
	
func start_training():
	for epoch in epochs:
		print("Start of Epoch " + str(epoch+1))
		for i in image_batches.size():
			var start_time = Time.get_ticks_msec()
			net.train(image_batches[i], label_batches[i], learn_rate, momentum)
			var end_time = Time.get_ticks_msec()
			var elapsed_time = end_time - start_time
			print("Time taken: %d ms" % elapsed_time)
			epoch_completion_label.text = "Epoch: " + str(snapped(float(i+1)/image_batches.size()*100, 0.1)) + "%"
			print(epoch_completion_label.text)
			save_network()
			#await get_tree().process_frame
			#update_statistics()

func update_statistics():
	
	var testing_correct := 0
	var training_correct := 0
	for i in testing_image_data.size():
		testing_correct += 1 if is_prediction_correct(net.forward_propagate(testing_image_data[i]), testing_label_data[i]) else 0
	for i in training_image_data.size():
		training_correct += 1 if is_prediction_correct(net.forward_propagate(training_image_data[i]), training_label_data[i]) else 0
	training_accuracy_label.text = "Training: " + str(snapped(float(training_correct)/training_image_data.size()*100, 0.1)) + "%"
	testing_accuracy_label.text = "Testing: " + str(snapped(float(testing_correct)/testing_image_data.size()*100, 0.1)) + "%"
	print("Total Cost: " + str(net.calculate_average_cost(image_data.slice(0,10), label_data.slice(0,10))))
	
	
func is_prediction_correct(prediction, expected_outputs):
	var max_index = 0
	var max = prediction[0]
	for i in range(1, prediction.size(), 1):
		if prediction[i] > max:
			max = prediction[i]
			max_index = i
	return expected_outputs[max_index] == 1

func save_network():
	if net == null:
		return
	net.save_to_file("res://saves/neural_network_save.json")
	print("Saved")
	var a = []
	for i in 784:
		a.append(0.0)
	set_texture_on_rect(a)
	update_prediction_label(net.forward_propagate(a))
	
func load_network():
	net = NeuralNetwork.load_from_file("res://saves/neural_network_save.json")
	print("Loaded")
	
var test_index = 0
func _on_next_button_pressed():
	if net == null:
		return
	print(test_index)
	print("Total Cost: " + str(net.calculate_average_cost(image_data.slice(0,10), label_data.slice(0,10))))
	set_texture_on_rect(image_data[test_index])
	update_prediction_label(net.forward_propagate(image_data[test_index]))
	
	test_index += 1
	if test_index == image_data.size():
		test_index = 0

func set_texture_on_rect(data_point: Array):
	texture_rect.texture = convert_data_to_texture(data_point)
	
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

func convert_data_to_texture(pixel_data: Array):
	var width = 28
	var height = 28
	var image = Image.create(width, height, false, Image.FORMAT_L8)
	for y in height:
		for x in width:
			var v = pixel_data[y * width + x]
			image.set_pixel(x, y, Color(v, v, v))
	var texture = ImageTexture.new()
	texture.set_image(image)
	return texture
	
func apply_noise(pixel_data: Array):
	var texture = convert_data_to_texture(pixel_data)
	input_sprite.texture = texture
	input_sprite.material.set_shader_parameter("angle", deg_to_rad(randf_range(-20, 20)))
	input_sprite.material.set_shader_parameter("noise_strength", noise_strength)
	input_sprite.material.set_shader_parameter("offset", Vector2(randf_range(-3.0/28, 3.0/28), randf_range(-3.0/28, 3.0/28)))
	await RenderingServer.frame_post_draw
	return convert_texture_to_data(sub_viewport.get_texture())

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

func load_images_ubyte(images_file_path, image_count = -1):
	
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
	
func load_labels_ubyte(labels_file_path, label_count = -1):
	
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

func save_to_json(file_path: String, data: Array):
	var file = FileAccess.open(file_path, FileAccess.WRITE)
	var json = JSON.new()
	file.store_string(json.stringify(data))
	file.close()

func load_from_json(file_path: String):
	var file = FileAccess.open(file_path, FileAccess.READ)
	var json = JSON.new()
	var data = json.parse_string(file.get_as_text())
	file.close()
	return data

func _on_new_network_pressed():
	net = NeuralNetwork.new(layers, hidden_layer_activation, output_layer_activation, cost)
	print("New Network Initialized")
