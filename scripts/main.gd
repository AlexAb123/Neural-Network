extends Node2D

const BLUE_DOT = preload("res://sprites/Blue_Dot.png")
const BLUE_BOX = preload("res://sprites/Blue_Box.png")

const RED_DOT = preload("res://sprites/Red_dot.png")
const RED_BOX = preload("res://sprites/Red_Box.png")

var dots: Array[Array] = []

# Red is [1,0] Blue is [0,1]
var data = []
var labels = []

var data_batches
var label_batches

var net: NeuralNetwork

var hidden_layer_activation = ActivationFactory.new_activation(ActivationFactory.type.SIGMOID)
var output_layer_activation = ActivationFactory.new_activation(ActivationFactory.type.SOFTMAX)
var cost = CostFactory.new_cost(CostFactory.type.CROSS_ENTROPY)

var boxes: Array[Array] = []

func _ready():
	initialize_grid()
	for x in range(100):
		if x < 30 or x > 70:
			continue
		for y in range(100):
			if y > 30 and y < 70:
				continue
			data.append([x/100.0,y/100.0])
			if y < 50:
				labels.append([1,0])
			else:
				labels.append([0,1])
	shuffle_data(data, labels)
	net = NeuralNetwork.new(2, 2, 3, 1, hidden_layer_activation, output_layer_activation, cost)
	update_boxes()
	data_batches = create_mini_batches(data, 50)
	label_batches = create_mini_batches(labels, 50)
func update_boxes():
	for x in range(100):
		for y in range(100):
			var output = net.forward_propagate([x/100.0,y/100.0])
			if output[0] > output[1]:
				boxes[x][y].texture = RED_BOX
			else:
				boxes[x][y].texture = BLUE_BOX
func initialize_grid():
	for x in range(100):
		if x < 30 or x > 70:
			continue
		var temp = []
		for y in range(100):
			if y > 30 and y < 70:
				continue
			var sprite: Sprite2D = Sprite2D.new()
			if y < 50:
				sprite.texture = RED_DOT
			else:
				sprite.texture = BLUE_DOT
			sprite.global_position = Vector2(16*x,16*y)
			add_child(sprite)
	for x in range(100):
		var temp = []
		
		for y in range(100):
			var box = Sprite2D.new()
			box.global_position = Vector2(16*x,16*y)
			add_child(box)
			temp.append(box)
		boxes.append(temp)
	for x in boxes.size():
		for y in boxes[x].size():
			boxes[x][y].global_position = Vector2(16*x,16*y)

func create_mini_batches(data: Array, batch_size: int):
	var batches = []
	for i in range(0, data.size(), batch_size):
		batches.append(data.slice(i, i+batch_size))
	return batches
	
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
		
var start = false
func _on_train_button_pressed():
	start = not start

var frame = 0
var data_index = 0

func _process(delta):
	
	if start:
		frame += 1
		for i in data_batches.size():
			net.train(data_batches[i], label_batches[i], 0.5, 10, 0.8)
			net.save_to_file("res://saves/neural_network_save.json")
			update_boxes()
		if frame == 1:
			print("Total Cost:")
			print(net.calculate_average_cost(data, labels))
			frame = 0
